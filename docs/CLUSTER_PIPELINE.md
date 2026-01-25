# Cluster Pipeline Guide

Complete guide for processing large RDF datasets using the cluster pipeline. This covers exporting from local SPARQL, transferring to a GPU cluster, processing embeddings, and deploying locally.

## Overview

The cluster pipeline (`scripts/cluster_pipeline.py`) orchestrates three processing steps:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: EXPORT                                                              │
│ SPARQL Endpoint → RDF Triples (TTL file)                                    │
│                                                                             │
│ Location: Local machine (with SPARQL access)                                │
│ Output: data/exports/<dataset>_dump.ttl                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: GENERATE DOCUMENTS                                                  │
│ RDF Triples → Entity Documents (Markdown)                                   │
│                                                                             │
│ Location: Local machine or cluster (no network needed)                      │
│ Output: data/documents/<dataset>/entity_documents/*.md                      │
│         data/documents/<dataset>/documents_metadata.json                    │
│                                                                             │
│ Features:                                                                   │
│ - Multiprocessing support (--workers N)                                     │
│ - Event-aware relationship traversal                                        │
│ - Configurable context depth (0, 1, 2)                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: EMBED                                                               │
│ Entity Documents → Embeddings + Document Graph                              │
│                                                                             │
│ Location: GPU cluster (recommended) or local                                │
│ Output: data/cache/<dataset>/document_graph.pkl                             │
│         data/cache/<dataset>/vector_index/                                  │
│                                                                             │
│ Features:                                                                   │
│ - GPU acceleration (CUDA)                                                   │
│ - Batch processing                                                          │
│ - Embedding cache for resumability                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Full Pipeline (Single Machine)

```bash
# Process everything locally
python scripts/cluster_pipeline.py --dataset mah --all --workers 8
```

### Split Pipeline (Local + Cluster)

```bash
# === LOCAL MACHINE (has SPARQL) ===
# Only export - this is fast (single SPARQL query to dump all triples)
python scripts/cluster_pipeline.py --dataset mah --export

# Transfer TTL file to cluster (smaller than generated documents!)
rsync -avz data/exports/mah_dump.ttl user@cluster:CRM_RAG/data/exports/

# === GPU CLUSTER (has GPU + more CPU cores, no SPARQL needed) ===
# Generate docs AND embed on cluster - uses cluster's CPU cores + GPU
python scripts/cluster_pipeline.py --dataset mah --generate-docs --embed --workers 16 --env .env.cluster

# Transfer cache back (document_graph.pkl + vector_index)
rsync -avz user@cluster:CRM_RAG/data/cache/mah/ data/cache/mah/

# === LOCAL MACHINE (serve) ===
python main.py --env .env.local
```

**Why this workflow is better:**
- `--generate-docs` works from the TTL file, not SPARQL - can run on cluster
- TTL file is smaller than thousands of generated document files
- Cluster has more CPU cores for multiprocessing (`--workers 16`)
- Cluster has GPU for fast embedding

## Prerequisites

### Local Machine

- Python 3.9+
- Access to SPARQL endpoint (Fuseki, GraphDB, etc.)
- CRM_RAG repository cloned and configured

### GPU Cluster

- Python 3.9+
- CUDA-capable GPU (16GB+ VRAM recommended)
- CRM_RAG repository cloned
- Same embedding model as local (for query-time)

## Step-by-Step Guide

### 1. Configure Your Environment

#### Local Configuration

Create `config/.env.local`:

```bash
# Copy from example
cp config/.env.local.example config/.env.local
```

Edit `config/.env.local`:

```bash
# LLM for answering questions
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o

# Embedding configuration
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=auto  # Uses MPS on Mac, CPU otherwise
EMBEDDING_BATCH_SIZE=64

# Cache settings
USE_EMBEDDING_CACHE=true

TEMPERATURE=0.7
PORT=5001
```

Create `config/.env.secrets`:

```bash
OPENAI_API_KEY=sk-your-key-here
```

#### Cluster Configuration

Create `config/.env.cluster`:

```bash
# LLM (needed for answer generation if testing on cluster)
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o

# Embedding - optimized for GPU
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=128  # Larger batches for GPU

USE_EMBEDDING_CACHE=true
TEMPERATURE=0.7
PORT=5001
```

#### Dataset Configuration

Ensure your dataset is in `config/datasets.yaml`:

```yaml
default_dataset: mah

datasets:
  mah:
    name: mah
    display_name: "Museum Collection"
    description: "Museum artworks, artists, and exhibitions"
    endpoint: "http://localhost:3030/MAH/sparql"
    interface:
      page_title: "Museum Collection Chat"
      welcome_message: "Ask me about the museum collection..."
      example_questions:
        - "Which pieces from Swiss Artists are in the museum?"
```

### 2. Export RDF Data (Step 1)

Export all triples from your SPARQL endpoint:

```bash
python scripts/cluster_pipeline.py --dataset mah --export
```

**What happens:**
- Sends `CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }` query
- Saves all triples to `data/exports/mah_dump.ttl`
- Timeout: 1 hour (for large datasets)

**Output:**
```
INFO - Exporting triples from http://localhost:3030/MAH/sparql...
INFO - Exported 156.3 MB to data/exports/mah_dump.ttl
```

**Verify:**
```bash
ls -lh data/exports/mah_dump.ttl
# -rw-r--r--  1 user  staff   156M Jan 24 10:30 data/exports/mah_dump.ttl
```

### 3. Generate Entity Documents (Step 2)

Convert RDF triples to entity documents:

```bash
# Single-threaded (safe, slower)
python scripts/cluster_pipeline.py --dataset mah --generate-docs

# Multi-threaded (8 workers, much faster)
python scripts/cluster_pipeline.py --dataset mah --generate-docs --workers 8

# Use existing export file
python scripts/cluster_pipeline.py --dataset mah --generate-docs --from-file data/exports/mah_dump.ttl
```

**Options:**
- `--workers N`: Number of parallel workers (default: 1)
- `--context-depth 0|1|2`: Relationship traversal depth (default: 2)
  - `0`: No relationships (just entity properties)
  - `1`: Direct relationships only
  - `2`: Multi-hop through events (recommended for CIDOC-CRM)

**Output:**
```
INFO - Loading graph from data/exports/mah_dump.ttl...
INFO - Loaded 2,847,392 triples
INFO - Building indexes...
INFO - Indexed 156,234 entities with literals
INFO - Generating documents for 156234 entities (context_depth=2, workers=8)...
Generating documents: 100%|████████████████| 156234/156234 [12:34<00:00, 207.1 entities/sec]
INFO - Generated 156234 documents
INFO - Documents: data/documents/mah/entity_documents
INFO - Metadata: data/documents/mah/documents_metadata.json
```

**Performance (867K entities):**

| Workers | Time | Rate |
|---------|------|------|
| 1 | ~70 min | 200 entities/sec |
| 4 | ~20 min | 700 entities/sec |
| 8 | ~12 min | 1,200 entities/sec |
| 32 | ~5 min | 2,800 entities/sec |

**Verify:**
```bash
ls data/documents/mah/entity_documents/ | wc -l
# 156234

cat data/documents/mah/documents_metadata.json | head -20
```

### 4. Transfer to Cluster

Transfer document files to the GPU cluster:

```bash
# Using rsync (recommended - shows progress, handles interruptions)
rsync -avz --progress data/documents/mah/ user@cluster:CRM_RAG/data/documents/mah/

# Using scp
scp -r data/documents/mah/ user@cluster:CRM_RAG/data/documents/mah/
```

**Also transfer configuration:**
```bash
rsync -avz config/datasets.yaml user@cluster:CRM_RAG/config/
rsync -avz config/.env.cluster user@cluster:CRM_RAG/config/
rsync -avz config/.env.secrets user@cluster:CRM_RAG/config/
rsync -avz data/labels/ user@cluster:CRM_RAG/data/labels/
```

### 5. Compute Embeddings on Cluster (Step 3)

SSH to your cluster and run:

```bash
cd CRM_RAG

# Activate your environment
source venv/bin/activate

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Run embedding computation
python scripts/cluster_pipeline.py --dataset mah --embed --env .env.cluster
```

**Output:**
```
INFO - STEP 3: COMPUTE EMBEDDINGS
INFO - Loading embedding model: BAAI/bge-m3 on cuda
INFO - Model loaded. Embedding dimension: 1024
INFO - Found metadata for 156234 documents
Reading documents: 100%|████████████████| 156234/156234 [00:45<00:00, 3472.1 doc/s]
INFO - Loaded 156234 documents
INFO - Processing embedding batch 1/2442
Batch embedding: 100%|████████████████| 2442/2442 [08:23<00:00, 4.85 batch/s]
INFO - Building document graph edges...
INFO - Added 847392 edges based on document content relationships
INFO - Building vector store...
INFO - Vector store saved to data/cache/mah/vector_index
INFO - Document graph saved to data/cache/mah/document_graph.pkl
```

**Performance:**

| GPU | 50K entities | 150K entities | 500K entities |
|-----|--------------|---------------|---------------|
| A100 (40GB) | 3 min | 8 min | 25 min |
| RTX 4090 | 5 min | 15 min | 45 min |
| RTX 3090 | 8 min | 22 min | 70 min |
| CPU (fallback) | 2 hours | 6 hours | 20 hours |

### 6. Transfer Cache Back to Local

```bash
# Transfer embeddings and graph back
rsync -avz --progress user@cluster:CRM_RAG/data/cache/mah/ data/cache/mah/
```

**Verify:**
```bash
ls -lh data/cache/mah/
# document_graph.pkl  (50-200 MB)
# vector_index/       (100-400 MB)
```

### 7. Run the Server

```bash
python main.py --env .env.local
```

**Output:**
```
INFO - Document graph loaded from data/cache/mah/document_graph.pkl with 156234 documents
INFO - Vector store loaded successfully
INFO - Starting Flask application...
 * Running on http://localhost:5001
```

Open `http://localhost:5001` in your browser.

## Pipeline Commands Reference

### Run Steps

```bash
# Full pipeline
python scripts/cluster_pipeline.py --dataset <id> --all

# Individual steps
python scripts/cluster_pipeline.py --dataset <id> --export
python scripts/cluster_pipeline.py --dataset <id> --generate-docs
python scripts/cluster_pipeline.py --dataset <id> --embed

# Combined steps
python scripts/cluster_pipeline.py --dataset <id> --export --generate-docs
python scripts/cluster_pipeline.py --dataset <id> --generate-docs --embed
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset <id>` | Dataset ID (required) | - |
| `--env <file>` | Environment config file | `.env` |
| `--workers <n>` | Parallel workers for doc generation | 1 |
| `--context-depth <0\|1\|2>` | Relationship traversal depth | 2 |
| `--batch-size <n>` | Embedding batch size | 64 |
| `--from-file <path>` | Use existing TTL file | - |

### Utilities

```bash
# Check pipeline status
python scripts/cluster_pipeline.py --dataset mah --status

# Clean all intermediate files
python scripts/cluster_pipeline.py --dataset mah --clean

# Clean specific files
python scripts/cluster_pipeline.py --dataset mah --clean-export   # Remove TTL export
python scripts/cluster_pipeline.py --dataset mah --clean-docs     # Remove documents
python scripts/cluster_pipeline.py --dataset mah --clean-cache    # Remove embeddings
```

### Status Output

```bash
python scripts/cluster_pipeline.py --dataset mah --status
```

```json
{
  "dataset_id": "mah",
  "endpoint": "http://localhost:3030/MAH/sparql",
  "steps": {
    "export": {
      "complete": true,
      "file": "data/exports/mah_dump.ttl",
      "size_mb": 156.3,
      "modified": "2025-01-24T10:30:00"
    },
    "generate": {
      "complete": true,
      "document_count": 156234,
      "directory": "data/documents/mah/entity_documents",
      "generated_at": "2025-01-24T10:45:00"
    },
    "embed": {
      "complete": true,
      "graph_file": "data/cache/mah/document_graph.pkl",
      "graph_size_mb": 89.2,
      "vector_dir": "data/cache/mah/vector_index",
      "modified": "2025-01-24T11:00:00"
    }
  }
}
```

## SLURM Job Scripts

### Document Generation (CPU Job)

`bulk_docs.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=crm_docs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/docs_%j.out
#SBATCH --error=logs/docs_%j.err

module load python/3.11
source ~/venv/bin/activate
cd ~/CRM_RAG

python scripts/cluster_pipeline.py \
    --dataset mah \
    --generate-docs \
    --from-file data/exports/mah_dump.ttl \
    --workers 32
```

Submit:
```bash
sbatch bulk_docs.sbatch
```

### Embedding Computation (GPU Job)

`embed.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=crm_embed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

module load python/3.11
module load cuda/12.1
source ~/venv/bin/activate
cd ~/CRM_RAG

python scripts/cluster_pipeline.py \
    --dataset mah \
    --embed \
    --env .env.cluster \
    --batch-size 128
```

Submit:
```bash
sbatch embed.sbatch
```

### Full Pipeline (Combined Job)

`full_pipeline.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=crm_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

module load python/3.11
module load cuda/12.1
source ~/venv/bin/activate
cd ~/CRM_RAG

# Run full pipeline (assumes TTL already transferred)
python scripts/cluster_pipeline.py \
    --dataset mah \
    --generate-docs \
    --embed \
    --from-file data/exports/mah_dump.ttl \
    --workers 32 \
    --env .env.cluster \
    --batch-size 128
```

## Troubleshooting

### Export Fails

**Timeout error:**
```
requests.exceptions.ReadTimeout: HTTPConnectionPool... Read timed out
```

Solution: The default timeout is 1 hour. For very large datasets, export directly:
```bash
curl -X POST "http://localhost:3030/MAH/sparql" \
  -H "Accept: text/turtle" \
  --data-urlencode "query=CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }" \
  -o data/exports/mah_dump.ttl
```

**Endpoint not found:**
```
requests.exceptions.ConnectionError: Connection refused
```

Solution: Check your SPARQL endpoint is running:
```bash
curl http://localhost:3030/MAH/sparql?query=ASK%20%7B%7D
```

### Document Generation Fails

**Memory error with many workers:**
```
MemoryError: Unable to allocate array
```

Solution: Reduce workers or use a machine with more RAM:
```bash
python scripts/cluster_pipeline.py --dataset mah --generate-docs --workers 4
```

**Missing labels file:**
```
FileNotFoundError: data/labels/property_labels.json
```

Solution: Extract ontology labels first:
```bash
python scripts/extract_ontology_labels.py
```

### Embedding Fails

**CUDA out of memory:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

Solution: Reduce batch size:
```bash
python scripts/cluster_pipeline.py --dataset mah --embed --batch-size 32
```

**Model not found:**
```
OSError: BAAI/bge-m3 does not appear to be a valid model
```

Solution: The model will be downloaded on first use. Ensure internet access or pre-download:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
```

### Cache Loading Fails

**Dimension mismatch:**
```
ValueError: Embedding dimension mismatch: expected 1024, got 768
```

Solution: The embedding model on local must match the cluster model exactly. Check both configs use the same `EMBEDDING_MODEL`.

**Pickle error:**
```
ModuleNotFoundError: No module named 'graph_document_store'
```

Solution: Ensure the same codebase version on both machines, or rebuild the cache.

## Best Practices

### For Large Datasets (100K+ entities)

1. **Always use multiprocessing for document generation:**
   ```bash
   --workers 8  # Local laptop
   --workers 32 # Cluster node
   ```

2. **Use GPU for embeddings:**
   ```bash
   EMBEDDING_DEVICE=cuda
   EMBEDDING_BATCH_SIZE=128
   ```

3. **Enable embedding cache for resumability:**
   ```bash
   USE_EMBEDDING_CACHE=true
   ```

4. **Check status before re-running:**
   ```bash
   python scripts/cluster_pipeline.py --dataset mah --status
   ```

### For Multiple Datasets

Process in parallel on cluster:
```bash
# Submit multiple jobs
for dataset in mah asinou museum; do
  sbatch --export=DATASET=$dataset embed.sbatch
done
```

Or sequentially:
```bash
for dataset in mah asinou museum; do
  python scripts/cluster_pipeline.py --dataset $dataset --embed --env .env.cluster
done
```

### Updating Data

When your SPARQL data changes:

```bash
# 1. Re-export
python scripts/cluster_pipeline.py --dataset mah --export

# 2. Re-generate documents
python scripts/cluster_pipeline.py --dataset mah --generate-docs --workers 8

# 3. Transfer and re-embed
rsync -avz data/documents/mah/ user@cluster:CRM_RAG/data/documents/mah/
# On cluster:
python scripts/cluster_pipeline.py --dataset mah --embed --env .env.cluster

# 4. Transfer cache and restart
rsync -avz user@cluster:CRM_RAG/data/cache/mah/ data/cache/mah/
python main.py --env .env.local
```

## File Sizes Reference

| Component | Size per 100K entities |
|-----------|------------------------|
| TTL export | 20-50 MB |
| Documents | 100-300 MB |
| Document graph | 50-150 MB |
| Vector index | 100-400 MB |
| Embedding cache | 200-600 MB |

## Related Documentation

- [CLUSTER_EMBEDDINGS.md](CLUSTER_EMBEDDINGS.md) - Detailed GPU cluster setup
- [LOCAL_EMBEDDINGS.md](LOCAL_EMBEDDINGS.md) - Local embedding configuration
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
