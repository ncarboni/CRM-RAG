# SPARQL Optimization Plan for Document Generation

## Problem Statement

The MAH dataset has **866,885 entities** requiring document generation. The current bottleneck is not embedding (2-3 hours with OpenAI API for $3.42) but **document generation** from RDF data.

## Current Approaches

### Option 1: Individual SPARQL Queries (`universal_rag_system.py`)

```
For each entity:
    SPARQL Endpoint → [SELECT query] → Entity context → Document
```

**Code path**: `create_enhanced_document()` → `get_entity_context()` (lines 1155-1300)

**Usage**: `python main.py --generate-docs-only`

**Pros**:
- Simple, no intermediate files
- Works with any SPARQL endpoint

**Cons**:
- 866K entities × 1 query each = 866K network round-trips
- Very slow for large datasets

### Option 2: Bulk TTL Export + rdflib (`bulk_generate_documents.py`)

```
SPARQL Endpoint → [CONSTRUCT dump] → TTL file → [rdflib parse] → Entity Documents
```

**Code path**: `export_from_sparql()` → `load_from_file()` → `generate_all_documents()`

**Usage**: `python scripts/bulk_generate_documents.py --dataset mah`

**Pros**:
- Single network request for all data
- In-memory processing after load

**Cons**:
- Large TTL files (potentially GBs)
- rdflib parsing can be slow/memory-intensive
- Intermediate file storage needed

## Proposed Strategy: Batch SPARQL Queries

A third approach combining benefits of both:

```
SPARQL Endpoint → [Batch SELECT with VALUES] → Entity Documents (directly)
```

This avoids both per-entity queries AND large TTL files.

## Batch SPARQL Query Strategy

Instead of querying one entity at a time, use `VALUES` clause to query multiple entities per request.

### Single Entity Query (slow)
```sparql
SELECT ?p ?o WHERE {
  <http://example.org/entity/123> ?p ?o .
}
```
- 866K entities × 1 query each = 866K queries
- At 0.18s/query (QLever best case) = 43+ hours

### Batch Query with VALUES (fast)
```sparql
SELECT ?entity ?p ?o WHERE {
  VALUES ?entity {
    <http://example.org/entity/1>
    <http://example.org/entity/2>
    ...
    <http://example.org/entity/1000>
  }
  ?entity ?p ?o .
}
```
- 866K entities ÷ 1000 per batch = 867 queries
- At 0.18s/query (QLever) = ~2.6 minutes for all entity data

### Alternative: SPARQL CONSTRUCT for Batches
```sparql
CONSTRUCT {
  ?entity ?p ?o .
} WHERE {
  VALUES ?entity { <uri1> <uri2> ... }
  ?entity ?p ?o .
}
```
Returns RDF triples directly, useful if we want to keep some RDF processing.

## Pros and Cons of Batch SPARQL

### Pros

1. **Speed**: Dramatically fewer round-trips to the endpoint
   - 867 queries vs 866,885 queries
   - Network latency paid once per batch, not per entity

2. **Memory efficiency**: Process results in streaming fashion
   - No need to load entire TTL file into memory
   - Generate documents as results arrive

3. **Resume capability**: Easy to checkpoint progress
   - Track which batches completed
   - Resume from last successful batch

4. **Flexibility**: Can adjust batch size dynamically
   - Larger batches for fast endpoints
   - Smaller batches if queries timeout

5. **No intermediate files**: Skip TTL export entirely
   - Less disk I/O
   - No large temp files

### Cons

1. **Query complexity**: VALUES clause has limits
   - Some endpoints limit VALUES size (1000-10000 URIs)
   - Very long URIs increase query size

2. **Endpoint load**: Batch queries are heavier per-request
   - May trigger rate limiting
   - Could impact other users on shared endpoints

3. **Error handling**: One bad entity can fail entire batch
   - Need robust error handling
   - May need to retry with smaller batches

4. **Result size**: Large batches return large result sets
   - Memory pressure on client side
   - May need streaming result parsing

5. **Endpoint compatibility**: Not all endpoints optimize VALUES equally
   - Fuseki: May not optimize well
   - QLever: Designed for this pattern

## Testing Strategy

### Benchmark Infrastructure

A dedicated benchmark script has been created:

```
benchmarks/
├── benchmark_document_generation.py   # Main benchmark script
└── results/                           # Output directory for reports
    └── <dataset>_<timestamp>/
        ├── report.json                # Machine-readable results
        └── report.md                  # Human-readable summary
```

### Start with Asinou (Small Dataset)

**Why asinou first:**
- ~692 entities (vs 866K for mah)
- Quick iteration on benchmark methodology
- Validate measurements before long-running tests
- Same CIDOC-CRM structure as mah

### Phase 0: Benchmark Results (Fuseki)

#### Asinou Dataset (605 entities, depth=2 bidirectional traversal)

| Approach | Total Time | Queries | Throughput | Speedup |
|----------|------------|---------|------------|---------|
| Individual Queries | 144.75s | 201,574 | 4.2 ent/s | baseline |
| Bulk Export + rdflib | 0.24s | 1 | 2,527 ent/s | 603x |
| **Batch Queries** | **0.09s** | **6** | **6,699 ent/s** | **1,602x** |

**Key findings:**
- Individual queries: ~333 SPARQL queries per entity (outgoing + incoming × depth)
- Batch queries with VALUES clause are **1,602x faster** than individual queries
- Bulk export is fast for small datasets but downloads entire graph

#### Run More Benchmarks

Run the benchmark script to measure all three approaches:

```bash
# Full benchmark on asinou (all approaches, all entities)
python benchmarks/benchmark_document_generation.py --dataset asinou

# Quick test with limited entities
python benchmarks/benchmark_document_generation.py --dataset asinou --limit 100

# Test specific approach
python benchmarks/benchmark_document_generation.py --dataset asinou --approach individual
python benchmarks/benchmark_document_generation.py --dataset asinou --approach bulk
python benchmarks/benchmark_document_generation.py --dataset asinou --approach batch

# Custom batch sizes for batch approach
python benchmarks/benchmark_document_generation.py --dataset asinou --approach batch --batch-sizes 50,100,200,500
```

**Expected output:**
- `benchmarks/results/asinou_<timestamp>/report.md` - Summary comparison
- `benchmarks/results/asinou_<timestamp>/report.json` - Detailed timing data

**Metrics collected:**
- Total time per approach
- Throughput (entities/second)
- Per-query timing statistics (avg, median, min, max, stddev)
- Error rates
- Memory/data size (for bulk export)

### Phase 1: Analyze Results and Scale Test

After asinou benchmark completes:

1. **Review report.md** - Identify fastest approach
2. **Check error rates** - Any failures with batch queries?
3. **Optimal batch size** - Which size gives best throughput?

If batch queries work well on asinou, scale test on mah:

```bash
# Test on mah with limited entities
python benchmarks/benchmark_document_generation.py --dataset mah --limit 1000
python benchmarks/benchmark_document_generation.py --dataset mah --limit 10000
```

**Decision point:**
- If Fuseki + batch queries achieve > 100 entities/sec → Proceed with Fuseki
- If < 50 entities/sec or high error rate → Consider QLever

### Phase 2: QLever Comparison (If Needed)

Only proceed to QLever if Fuseki performance is insufficient.

**Why QLever might be needed:**
- Benchmark data shows 85x faster than Fuseki (0.18s vs 15s geometric mean)
- Better VALUES clause optimization
- Designed for large-scale SPARQL

**Setup required:**
1. Install QLever
2. Load dataset
3. Run same benchmark with `--endpoint` pointing to QLever

```bash
python benchmarks/benchmark_document_generation.py --dataset mah --endpoint http://localhost:7000/sparql
```

**Decision criteria:**

| Metric | Fuseki Acceptable | Switch to QLever |
|--------|-------------------|------------------|
| Throughput | > 100 ent/sec | < 50 ent/sec |
| Error rate | < 1% | > 5% |
| 866K entities time | < 3 hours | > 6 hours |

### Phase 3: Implementation

Based on test results, implement the faster approach in `bulk_generate_documents.py`.

## Implementation Outline

```python
class BatchSPARQLDocumentGenerator:
    def __init__(self, endpoint_url, batch_size=1000):
        self.endpoint = endpoint_url
        self.batch_size = batch_size

    def get_all_entities(self) -> List[str]:
        """Get list of all entity URIs"""
        query = "SELECT DISTINCT ?entity WHERE { ?entity ?p ?o }"
        # ... execute and return URIs

    def get_entity_batch(self, entity_uris: List[str]) -> Dict[str, List[Triple]]:
        """Query properties for a batch of entities"""
        values_clause = " ".join(f"<{uri}>" for uri in entity_uris)
        query = f"""
        SELECT ?entity ?p ?o WHERE {{
          VALUES ?entity {{ {values_clause} }}
          ?entity ?p ?o .
        }}
        """
        # ... execute and group by entity

    def generate_documents(self, output_dir: str):
        """Main entry point"""
        entities = self.get_all_entities()

        for i in range(0, len(entities), self.batch_size):
            batch = entities[i:i + self.batch_size]
            entity_data = self.get_entity_batch(batch)

            for entity_uri, triples in entity_data.items():
                doc = self.create_document(entity_uri, triples)
                self.write_document(doc, output_dir)

            # Checkpoint progress
            self.save_checkpoint(i + len(batch))
```

## Resource Estimates

### Fuseki (estimated)
- Query time: 15-30s per 1000-entity batch (based on benchmarks)
- Total for 867 batches: 3.6 - 7.2 hours
- Plus document generation: ~1 hour
- **Total: 5-8 hours**

### QLever (estimated)
- Query time: 0.2-0.5s per 1000-entity batch
- Total for 867 batches: 3-7 minutes
- Plus document generation: ~1 hour
- **Total: ~1 hour**

### Current TTL Approach (observed)
- TTL export: ? hours
- TTL parsing + doc generation: ? hours
- **Total: "quite a while"** (user's words)

## Next Steps

1. [ ] Create test script for batch SPARQL queries
2. [ ] Run Fuseki tests with 10K entity sample
3. [ ] Collect metrics and evaluate
4. [ ] If Fuseki insufficient, set up QLever
5. [ ] Run comparison tests
6. [ ] Implement chosen approach in pipeline

## Open Questions

1. What is the current TTL export + parse time for MAH?
2. Is QLever already available or needs setup?
3. Are there rate limits on the SPARQL endpoint?
4. What's the acceptable total time for document generation?

---

## QLever Setup Guide

QLever is a high-performance graph database optimized for large-scale SPARQL queries. It's particularly efficient with VALUES clause batching, making it ideal for our batch query strategy.

### Why QLever?

- **85x faster** than Fuseki for complex queries (0.18s vs 15s geometric mean)
- Optimized for VALUES clause queries (our batch strategy)
- Can handle hundreds of billions of triples on a single machine
- Efficient memory usage with external memory algorithms

### Installation

#### macOS (Homebrew) - Recommended

```bash
brew tap qlever-dev/qlever
brew install qlever
```

#### Debian/Ubuntu (apt)

```bash
sudo apt update && sudo apt install -y wget gpg ca-certificates
wget -qO - https://packages.qlever.dev/pub.asc | gpg --dearmor | sudo tee /usr/share/keyrings/qlever.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/qlever.gpg] https://packages.qlever.dev/ $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") main" | sudo tee /etc/apt/sources.list.d/qlever.list
sudo apt update && sudo apt install qlever
```

#### Other Platforms (pip/pipx/uv)

```bash
# Using pip
pip install qlever

# Using pipx (isolated environment)
pipx install qlever

# Using uv
uv tool install qlever
```

Note: pip/pipx/uv installation runs QLever in a container, which has a small performance penalty compared to native installation.

### Setup for MAH Dataset

#### 1. Create Working Directory

```bash
mkdir -p ~/qlever-indices/mah
cd ~/qlever-indices/mah
```

#### 2. Create Qleverfile

Create a file named `Qleverfile` in the working directory:

```ini
[data]
NAME = mah
# Point to the existing TTL dump
# No download needed - we have local data
GET_DATA_CMD = echo "Using local TTL dump"
FORMAT = ttl

[index]
# Path to your TTL dump file
INPUT_FILES = /Users/carboni/Documents/pynotebook/personal/RAG rdf/CRM_RAG/data/exports/mah_dump.ttl
# Memory for index building (adjust based on your RAM)
# For 333MB TTL, 4-8GB should be sufficient
STXXL_MEMORY = 8G
# Index settings for optimal query performance
SETTINGS_JSON = {"num-triples-per-batch": 5000000, "language-priority": "en,de,fr,it"}

[server]
# SPARQL endpoint port (choose any available port)
PORT = 7001
# Memory for query processing
MEMORY_FOR_QUERIES = 8G
# Result cache size
CACHE_MAX_SIZE = 4G
# Query timeout in seconds
TIMEOUT = 300
# Leave empty for public access (local development)
ACCESS_TOKEN =

[runtime]
# Use Docker for containerized execution
SYSTEM = docker
IMAGE = docker.io/adfreiburg/qlever:latest
CONTAINER_NAME = qlever-mah

[ui]
# Optional: Web UI for interactive queries
UI_PORT = 7000
```

#### 3. Build the Index

```bash
cd ~/qlever-indices/mah
qlever index
```

This will:
- Parse the TTL file
- Build optimized index structures
- Create vocabulary and permutation files

For the MAH dataset (333MB, ~6.7M triples), indexing should take **5-15 minutes**.

#### 4. Start the SPARQL Endpoint

```bash
qlever start
```

The endpoint will be available at `http://localhost:7001`

#### 5. Verify It's Working

```bash
# Test with a simple query
qlever query "SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }"

# Or use curl
curl -s "http://localhost:7001" \
  --data-urlencode "query=SELECT (COUNT(*) AS ?count) WHERE { ?s ?p ?o }" \
  --data-urlencode "action=tsv_export"
```

#### 6. (Optional) Start the Web UI

```bash
qlever ui
```

Access the UI at `http://localhost:7000`

### Using QLever in the Pipeline

#### Update datasets.yaml

Add a QLever endpoint configuration:

```yaml
mah_qlever:
  name: "MAH (QLever)"
  description: "MAH dataset with QLever endpoint"
  endpoint: "http://localhost:7001"
  # ... other settings
```

#### Run Benchmark Against QLever

```bash
# Test with QLever endpoint
python benchmarks/benchmark_document_generation.py \
  --dataset mah \
  --endpoint http://localhost:7001 \
  --limit 5000

# Compare with Fuseki
python benchmarks/benchmark_document_generation.py \
  --dataset mah \
  --endpoint http://localhost:3030/mah/sparql \
  --limit 5000
```

### QLever CLI Commands Reference

| Command | Description |
|---------|-------------|
| `qlever setup-config <name>` | Get example Qleverfile for dataset |
| `qlever index` | Build index from data files |
| `qlever start` | Start SPARQL endpoint server |
| `qlever stop` | Stop the server |
| `qlever restart` | Restart the server |
| `qlever status` | Show server status |
| `qlever query "<sparql>"` | Execute a SPARQL query |
| `qlever log` | Show server logs |
| `qlever ui` | Start web UI |
| `qlever --help` | Show all commands |

### Qleverfile Settings Reference

#### [data] Section
| Setting | Description |
|---------|-------------|
| `NAME` | Dataset identifier (used for file naming) |
| `GET_DATA_URL` | URL for downloading data |
| `GET_DATA_CMD` | Shell command to download/prepare data |
| `FORMAT` | RDF format: `ttl`, `nt`, `rdf` |

#### [index] Section
| Setting | Description |
|---------|-------------|
| `INPUT_FILES` | Glob pattern for input files |
| `MULTI_INPUT_JSON` | JSON spec for multiple file processing |
| `STXXL_MEMORY` | External memory for indexing (e.g., `8G`) |
| `SETTINGS_JSON` | JSON object with indexer settings |

#### [server] Section
| Setting | Description |
|---------|-------------|
| `PORT` | HTTP port for SPARQL endpoint |
| `MEMORY_FOR_QUERIES` | RAM for query processing |
| `CACHE_MAX_SIZE` | Result cache size limit |
| `TIMEOUT` | Query timeout in seconds |
| `ACCESS_TOKEN` | Optional authentication token |

#### [runtime] Section
| Setting | Description |
|---------|-------------|
| `SYSTEM` | Execution mode: `docker`, `native`, `podman` |
| `IMAGE` | Docker image URI |
| `CONTAINER_NAME` | Container identifier |

### Troubleshooting

#### Index building fails with memory error
- Increase `STXXL_MEMORY` in Qleverfile
- Ensure Docker has enough memory allocated (Docker Desktop → Settings → Resources)

#### Server won't start
- Check if port is already in use: `lsof -i :7001`
- Check logs: `qlever log`
- Ensure index was built successfully

#### Slow query performance
- Increase `MEMORY_FOR_QUERIES`
- Increase `CACHE_MAX_SIZE`
- Check if the query pattern is optimized for QLever

### Resources

- [QLever Documentation](https://docs.qlever.dev)
- [QLever GitHub](https://github.com/ad-freiburg/qlever)
- [qlever-control CLI](https://github.com/ad-freiburg/qlever-control)
- [Example Qleverfiles](https://deepwiki.com/qlever-dev/qlever-control/7-example-dataset-configurations)
