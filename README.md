# RAG Architecture for CIDOC-CRM

Graph-based RAG (Retrieval-Augmented Generation) system for querying CIDOC-CRM RDF data. Supports multiple datasets with lazy loading and per-dataset caching.

## Demo

https://github.com/user-attachments/assets/692a69ff-c25f-40b1-8a36-1a660e810060



## Repository Structure

```
CRM_RAG/
├── config/              Configuration files
│   ├── .env.openai.example
│   ├── .env.claude.example
│   ├── .env.r1.example
│   ├── .env.ollama.example
│   ├── .env.local.example    # Local embeddings (fast, no API)
│   ├── .env.secrets.example
│   ├── datasets.yaml         # Multi-dataset configuration
│   ├── event_classes.json    # CIDOC-CRM event classes for graph traversal
│   ├── interface.yaml        # Chat interface customization
│   └── README.md             # Configuration guide
├── data/                All data files
│   ├── ontologies/      CIDOC-CRM, VIR, CRMdig ontology files
│   ├── labels/          Extracted labels (shared across datasets)
│   ├── cache/           Per-dataset caches (auto-generated)
│   │   ├── asinou/          # Dataset-specific cache
│   │   │   ├── document_graph.pkl
│   │   │   ├── vector_index/
│   │   │   └── embeddings/  # Embedding cache for resumability
│   │   └── museum/          # Another dataset cache
│   │       ├── document_graph.pkl
│   │       ├── vector_index/
│   │       └── embeddings/
│   └── documents/       Per-dataset entity documents (auto-generated)
│       ├── asinou/entity_documents/
│       └── museum/entity_documents/
├── docs/                Documentation
│   ├── ARCHITECTURE.md
│   ├── LOCAL_EMBEDDINGS.md   # Local embeddings guide
│   ├── CLUSTER_PIPELINE.md   # GPU cluster processing guide
│   └── REORGANIZATION_PLAN.md
├── scripts/             Utility scripts
│   ├── extract_ontology_labels.py
│   └── bulk_generate_documents.py  # Fast bulk export for large datasets
├── logs/                Application logs
├── static/              Web interface CSS and JavaScript
├── templates/           Web interface HTML templates
├── main.py              Flask application entry point
├── universal_rag_system.py  Core RAG logic
├── graph_document_store.py  Graph-based document storage
├── llm_providers.py     LLM abstraction (OpenAI, Claude, local embeddings)
├── embedding_cache.py   Embedding cache for resumability
├── dataset_manager.py   Multi-dataset management
└── config_loader.py     Configuration loading
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create configuration files from templates in the `config/` directory:

```bash
# Copy the secrets template (for API keys)
cp config/.env.secrets.example config/.env.secrets

# Edit config/.env.secrets and add your actual API keys
OPENAI_API_KEY=your_actual_openai_key_here
ANTHROPIC_API_KEY=your_actual_anthropic_key_here
R1_API_KEY=your_actual_r1_key_here

# Copy the provider configuration you want to use
cp config/.env.openai.example config/.env.openai
# OR
cp config/.env.claude.example config/.env.claude
# OR
cp config/.env.r1.example config/.env.r1
# OR
cp config/.env.ollama.example config/.env.ollama
```

### 3. Configure Datasets

Create `config/datasets.yaml` to define your SPARQL datasets:

```yaml
# config/datasets.yaml
default_dataset: asinou  # Which dataset to load by default

datasets:
  asinou:
    name: asinou
    display_name: "Asinou Church"
    description: "Asinou church dataset with frescoes and iconography"
    endpoint: "http://localhost:3030/asinou/sparql"
    # Optional: use local embeddings for this small dataset
    embedding:
      provider: local
      model: BAAI/bge-m3
    interface:  # Optional: override interface.yaml settings
      page_title: "Asinou Dataset Chat"
      welcome_message: "Ask me about Asinou church..."
      example_questions:
        - "Where is Panagia Phorbiottisa located?"
        - "What frescoes are in the church?"

  museum:
    name: museum
    display_name: "Museum Collection"
    description: "Museum artworks, artists, and exhibitions"
    endpoint: "http://localhost:3030/museum/sparql"
    # Optional: use OpenAI embeddings for this dataset (inherits from .env if not specified)
    embedding:
      provider: openai
    interface:
      page_title: "Museum Collection Chat"
      example_questions:
        - "Which pieces from Swiss Artists are in the museum?"
```

Each dataset gets its own cache directory under `data/cache/<dataset_id>/`.

**Per-dataset embedding configuration:**

You can configure different embedding providers for each dataset:

```yaml
datasets:
  small_dataset:
    endpoint: "http://localhost:3030/small/sparql"
    embedding:
      provider: local              # Use local embeddings (fast)
      model: BAAI/bge-m3
      batch_size: 64

  large_dataset:
    endpoint: "http://localhost:3030/large/sparql"
    embedding:
      provider: openai             # Use OpenAI embeddings
      # model inherited from .env config
```

Available embedding options per dataset:
- `provider`: `local`, `sentence-transformers`, `openai`, `ollama`
- `model`: Embedding model name
- `batch_size`: Batch size for local embeddings (default: 64)
- `device`: `auto`, `cuda`, `mps`, `cpu`
- `use_cache`: `true` or `false`

### 4. Extract Ontology Labels

Extract English labels from ontology files (required on first run):

```bash
python scripts/extract_ontology_labels.py
```

This creates label files in `data/labels/` used by the RAG system.

### 5. Configure Event Classes (Optional)

The system uses event-aware graph traversal to build entity documents. In CIDOC-CRM, events (activities, productions, etc.) are the "glue" connecting things, actors, places, and times. Multi-hop context only traverses THROUGH events, preventing unrelated entities from polluting documents.

Event classes are configured in [`config/event_classes.json`](config/event_classes.json):

```json
{
  "_comment": "Add or remove event class URIs as needed",

  "cidoc_crm": [
    "http://www.cidoc-crm.org/cidoc-crm/E5_Event",
    "http://www.cidoc-crm.org/cidoc-crm/E12_Production",
    ...
  ],
  "crmdig": [...],
  "crmsci": [...],
  "vir": [...],
  "crminf": [...]
}
```

To customize:
- Add URIs to existing categories or create new ones
- Keys starting with `_` are ignored (use for comments)
- Changes take effect on next restart

### 6. Customize Chat Interface (Optional)

Customize the chatbot title, welcome message, and example questions by editing `config/interface.yaml`:

```yaml
page_title: "Your Dataset Chat"
header_title: "Your Custom Chatbot"
welcome_message: "Hello! Ask me about your dataset..."
example_questions:
  - "Your first example question?"
  - "Your second example question?"
```

See `config/README.md` for detailed customization options.

### 7. Start Your SPARQL Endpoint

Ensure your SPARQL server is running with your CIDOC-CRM dataset loaded at the configured endpoint.

## Usage

### Basic Usage

```bash
# Run with OpenAI
python main.py --env .env.openai

# Run with Claude
python main.py --env .env.claude

# Run with R1
python main.py --env .env.r1

# Run with Ollama (no API key needed)
python main.py --env .env.ollama

# Force rebuild of document graph and vector store
python main.py --env .env.openai --rebuild
```

Access the chat interface at `http://localhost:5001`

### Local Embeddings (Recommended for Large Datasets)

For datasets with 5,000+ entities, use local embeddings to avoid API rate limits and reduce processing time from days to minutes.

```bash
# Set up local embeddings config
cp config/.env.local.example config/.env.local

# Process a dataset with local embeddings
python main.py --env .env.local --dataset asinou --rebuild --process-only
```

| Method | 50,000 entities | Cost |
|--------|-----------------|------|
| OpenAI API | 2-4 days | ~$10-20 |
| Local (CPU) | 1-2 hours | Free |
| Local (GPU) | 10-20 minutes | Free |

**See [docs/LOCAL_EMBEDDINGS.md](docs/LOCAL_EMBEDDINGS.md) for complete documentation** including:
- Configuration options
- Per-dataset embedding providers
- Model recommendations
- Hardware acceleration (GPU/CPU)
- Troubleshooting

### Bulk Document Generation (Very Large Datasets)

For datasets with 100,000+ entities, the standard per-entity SPARQL queries become a bottleneck. The bulk export script exports all triples in one query and processes locally, reducing processing time from days to minutes.

```bash
# Generate documents using bulk export (reads endpoint from datasets.yaml)
python scripts/bulk_generate_documents.py --dataset mah
```

**Performance comparison for 867,000 entities:**

| Method | Time | Bottleneck |
|--------|------|------------|
| Standard (per-entity SPARQL) | ~113 days | Network round-trips |
| Bulk export + local processing | ~30-45 min | Disk I/O |

**Options:**

```bash
# Export only (creates data/exports/<dataset>_dump.ttl)
python scripts/bulk_generate_documents.py --dataset mah --export-only

# Process from existing export file
python scripts/bulk_generate_documents.py --dataset mah --from-file data/exports/mah_dump.ttl

# Override endpoint from datasets.yaml
python scripts/bulk_generate_documents.py --dataset mah --endpoint http://localhost:3030/other/sparql
```

**Workflow for GPU cluster embedding:**

```bash
# 1. Generate documents locally (fast bulk export)
python scripts/bulk_generate_documents.py --dataset mah

# 2. Transfer to cluster
scp -r data/documents/mah/ user@cluster:~/CRM_RAG/data/documents/mah/

# 3. On cluster: generate embeddings (no SPARQL needed)
python main.py --env .env.cluster --dataset mah --embed-from-docs --process-only

# 4. Transfer cache back
scp -r user@cluster:~/CRM_RAG/data/cache/mah/ ./data/cache/mah/

# 5. Run locally
python main.py --env .env.local
```

**Parallel document generation on cluster:**

For very large datasets (500K+ entities), use multiprocessing:

```bash
# Single machine (e.g., laptop)
python scripts/bulk_generate_documents.py --dataset mah

# Cluster node with 32 cores
python scripts/bulk_generate_documents.py --dataset mah --workers 32

# Memory usage: ~4-8 GB per worker for 867K entities
# With 512 GB RAM, you can safely use 32-64 workers
```

Example SLURM job script (`bulk_docs.sbatch`):
```bash
#!/bin/bash
#SBATCH --job-name=bulk_docs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00

module load python/3.11
source ~/venv/bin/activate
cd ~/CRM_RAG

python scripts/bulk_generate_documents.py \
    --dataset mah \
    --from-file data/exports/mah_dump.ttl \
    --workers 32
```

See [docs/CLUSTER_PIPELINE.md](docs/CLUSTER_PIPELINE.md) for the complete cluster workflow guide.

### Cluster Pipeline (Unified Workflow)

The cluster pipeline script (`scripts/cluster_pipeline.py`) unifies all processing steps into a single command for easier cluster deployment.

**Full pipeline (all steps):**

```bash
python scripts/cluster_pipeline.py --dataset mah --all
```

**Individual steps:**

```bash
# Step 1: Export RDF from SPARQL
python scripts/cluster_pipeline.py --dataset mah --export

# Step 2: Generate entity documents
python scripts/cluster_pipeline.py --dataset mah --generate-docs --workers 8

# Step 3: Compute embeddings
python scripts/cluster_pipeline.py --dataset mah --embed --env .env.cluster
```

**Typical cluster workflow:**

```bash
# LOCAL (has SPARQL access) - just export, fast single query
python scripts/cluster_pipeline.py --dataset mah --export

# Transfer to cluster: TTL file + labels (required for doc generation)
scp data/exports/mah_dump.ttl user@cluster:CRM_RAG/data/exports/
scp -r data/labels/ user@cluster:CRM_RAG/data/labels/

# CLUSTER (has GPU + more CPU cores) - generate docs AND embed
python scripts/cluster_pipeline.py --dataset mah --generate-docs --embed --workers 16 --env .env.cluster

# Transfer cache (embeddings) + documents (metadata) back
scp -r user@cluster:CRM_RAG/data/cache/mah/ data/cache/mah/
scp -r user@cluster:CRM_RAG/data/documents/mah/ data/documents/mah/

# Run locally
python main.py --env .env.local
```

**Check pipeline status:**

```bash
python scripts/cluster_pipeline.py --dataset mah --status
```

**Clean intermediate files:**

```bash
python scripts/cluster_pipeline.py --dataset mah --clean           # Clean all
python scripts/cluster_pipeline.py --dataset mah --clean-export    # Clean export only
python scripts/cluster_pipeline.py --dataset mah --clean-docs      # Clean documents only
python scripts/cluster_pipeline.py --dataset mah --clean-cache     # Clean embeddings only
```

For detailed documentation including SLURM job scripts, troubleshooting, and best practices, see [docs/CLUSTER_PIPELINE.md](docs/CLUSTER_PIPELINE.md).

### Multi-Dataset Mode

When `config/datasets.yaml` is configured, the chat interface displays a dataset selector dropdown. Select a dataset to:
- Load its cached embeddings (or build them on first access)
- Update the interface with dataset-specific titles and example questions
- Query only that dataset's knowledge graph

Datasets are lazily loaded - they initialize only when first selected, saving memory and startup time.

### Clearing Cache

To rebuild a specific dataset's cache:

```bash
# Clear cache for a specific dataset and rebuild
rm -rf data/cache/asinou/
rm -rf data/documents/asinou/
python main.py --env .env.openai --rebuild
```

For single-dataset mode (legacy):
```bash
rm -rf data/cache/document_graph.pkl data/cache/vector_index/
rm -rf data/documents/entity_documents/
python main.py --env .env.openai --rebuild
```

### CLI Reference

**main.py flags:**

| Flag | Description |
|------|-------------|
| `--env <file>` | Path to environment config file (e.g., `.env.openai`) |
| `--dataset <id>` | Dataset ID to process (from datasets.yaml) |
| `--process-only` | Process dataset and exit without starting web server |
| `--rebuild` | Force rebuild of document graph and vector store |
| `--embedding-provider <name>` | Embedding provider: `openai`, `local`, `sentence-transformers`, `ollama` |
| `--embedding-model <model>` | Embedding model name (e.g., `BAAI/bge-m3`) |
| `--no-embedding-cache` | Disable embedding cache (force re-embedding) |
| `--generate-docs-only` | Generate documents from SPARQL without embedding (for cluster workflow) |
| `--embed-from-docs` | Generate embeddings from existing documents (no SPARQL needed) |

**scripts/cluster_pipeline.py flags:**

| Flag | Description |
|------|-------------|
| `--dataset <id>` | Dataset ID (required, from datasets.yaml) |
| `--all` | Run full pipeline (export + generate + embed) |
| `--export` | Step 1: Export RDF from SPARQL endpoint |
| `--generate-docs` | Step 2: Generate entity documents |
| `--embed` | Step 3: Compute embeddings and build graph |
| `--env <file>` | Path to environment config file |
| `--from-file <path>` | Use existing TTL/RDF file instead of exporting |
| `--workers <n>` | Number of parallel workers (default: 1) |
| `--context-depth <0,1,2>` | Relationship traversal depth (default: 2) |
| `--batch-size <n>` | Embedding batch size (default: 64) |
| `--status` | Show pipeline status for dataset |
| `--clean` | Clean all intermediate files |

**scripts/bulk_generate_documents.py flags:**

| Flag | Description |
|------|-------------|
| `--dataset <id>` | Dataset ID (required, reads endpoint from datasets.yaml) |
| `--endpoint <url>` | Override SPARQL endpoint from config |
| `--from-file <path>` | Load from existing RDF export instead of querying |
| `--export-only` | Only export triples, don't generate documents |
| `--workers <n>` | Number of parallel processes (default: 1, use 32+ on cluster) |
| `--context-depth <0,1,2>` | Relationship traversal depth (default: 2 for CIDOC-CRM) |

### Processing Specific Datasets

Process a single dataset from the command line:

```bash
# Process dataset with local embeddings (recommended)
python main.py --env .env.local --dataset asinou --rebuild --process-only

# Process dataset with OpenAI embeddings
python main.py --env .env.openai --dataset museum --rebuild --process-only

# Process and start web server
python main.py --env .env.local --dataset asinou --rebuild
```

See [docs/LOCAL_EMBEDDINGS.md](docs/LOCAL_EMBEDDINGS.md) for detailed examples of processing multiple datasets with different embedding providers.

## API Endpoints

### Dataset Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/datasets` | GET | List all available datasets with their status |
| `/api/datasets/<id>/select` | POST | Initialize and select a dataset, returns interface config |

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send a question. Body: `{"question": "...", "dataset_id": "..."}` |
| `/api/info` | GET | Get system information (LLM provider, model, etc.) |
| `/api/entity/<uri>/wikidata` | GET | Get Wikidata info for an entity |

**Note:** In multi-dataset mode, `dataset_id` is required for `/api/chat`.

## Architecture

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Key Components

- **DatasetManager** (`dataset_manager.py`): Manages multiple RAG system instances with lazy loading
- **UniversalRagSystem** (`universal_rag_system.py`): Core RAG logic with CIDOC-CRM aware retrieval
- **GraphDocumentStore** (`graph_document_store.py`): Graph-based document storage with FAISS vectors
- **LLM Providers** (`llm_providers.py`): Abstraction layer for OpenAI, Anthropic, R1, Ollama, and local embeddings (sentence-transformers)
- **EmbeddingCache** (`embedding_cache.py`): Disk-based embedding cache for resumable processing

### Retrieval Pipeline

1. **Vector Search**: FAISS similarity search for initial candidates
2. **CIDOC-CRM Scoring**: Relationship-aware scoring based on ontology semantics
3. **PageRank**: Graph-based importance scoring
4. **Coherent Subgraph Extraction**: Selects connected documents balancing relevance and connectivity
