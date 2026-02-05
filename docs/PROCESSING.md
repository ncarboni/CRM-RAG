# Processing Guide

How to process RDF datasets for the CRM_RAG system.

## Overview

Processing converts RDF data into searchable documents with embeddings. This can run entirely on one machine or split between a local machine (with SPARQL access) and a GPU cluster (for faster embedding computation).

```
SPARQL Endpoint → TTL Export → Documents → Embeddings → RAG System
```

## Prerequisites

1. Complete basic setup from [README.md](../README.md)
2. Configure your dataset in `config/datasets.yaml`
3. Run `python scripts/extract_ontology_labels.py`

## Single Machine Processing

For small-to-medium datasets or machines with GPU:

```bash
python scripts/cluster_pipeline.py --dataset <id> --all --workers 8
```

This runs all steps: export → document generation → embedding computation.

Start the server:
```bash
python main.py --env .env.local
```

## Split Processing (Local + Cluster)

For large datasets, split processing between machines:

### Step 1: Export (Local)

```bash
python scripts/cluster_pipeline.py --dataset mah --export
```

Output: `data/exports/mah_dump.ttl`

### Step 2: Transfer to Cluster

```bash
scp data/exports/mah_dump.ttl user@cluster:CRM_RAG/data/exports/
scp -r data/labels/ user@cluster:CRM_RAG/data/labels/
```

### Step 3: Process on Cluster

```bash
ssh user@cluster
cd CRM_RAG && source venv/bin/activate

python scripts/cluster_pipeline.py --dataset mah \
  --generate-docs --embed \
  --workers 32 \
  --env .env.cluster
```

For large datasets, use scratch storage:
```bash
python scripts/cluster_pipeline.py --dataset mah \
  --generate-docs --embed \
  --workers 32 \
  --env .env.cluster \
  --data-dir ~/scratch/CRM_RAG_data
```

### Step 4: Transfer Results Back

```bash
scp -r user@cluster:CRM_RAG/data/cache/mah/ data/cache/mah/
scp -r user@cluster:CRM_RAG/data/documents/mah/ data/documents/mah/
```

### Step 5: Run Server (Local)

```bash
python main.py --env .env.local
```

## Cluster Setup

First-time setup on a new cluster:

```bash
# Clone and setup
git clone <repo-url> CRM_RAG
cd CRM_RAG

# Load modules (adjust for your cluster)
module load python/3.11
module load cuda/12.8

# Create environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create directories
mkdir -p data/{exports,labels,cache,documents} logs

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

The cluster config `config/.env.cluster` requires no API keys.

## Dataset Configuration

Define datasets in `config/datasets.yaml`:

```yaml
datasets:
  mah:
    name: mah
    display_name: "Museum Collection"
    endpoint: "http://localhost:3030/MAH/sparql"

    # Optional: image retrieval pattern
    image:
      sparql: |
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        ?entity crm:P138i_has_representation ?img .
        ?img rdf:value ?url .

    # Optional: interface customization
    interface:
      page_title: "Museum Collection Chat"
      welcome_message: "Ask about the museum collection..."
```

### Image Configuration

The `image.sparql` field defines how to find images for entities. Write a SPARQL graph pattern using:
- `?entity` - the entity URI (bound automatically)
- `?url` - the image URL to extract

Examples:

```yaml
# Direct property
image:
  sparql: |
    ?entity <http://schema.org/image> ?url .

# Two-step path
image:
  sparql: |
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
    ?entity crm:P138i_has_representation ?img .
    ?img <http://www.w3.org/1999/02/22-rdf-syntax-ns#value> ?url .

# With type filter
image:
  sparql: |
    PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
    ?entity crm:P138i_has_representation ?img .
    ?img a crm:E36_Visual_Item ;
         <http://www.w3.org/1999/02/22-rdf-syntax-ns#value> ?url .
```

Images are stored in document frontmatter and take priority over Wikidata images.

## Embedding Configuration

Configure in `.env.local` or `.env.cluster`:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_PROVIDER` | `local` or `openai` | local |
| `EMBEDDING_MODEL` | Model name | BAAI/bge-m3 |
| `EMBEDDING_BATCH_SIZE` | Base batch size | 64 |
| `EMBEDDING_DEVICE` | `auto`, `cuda`, `mps`, `cpu` | auto |
| `USE_EMBEDDING_CACHE` | Enable disk cache | true |

### Recommended Models

| Model | Size | Languages | Use Case |
|-------|------|-----------|----------|
| `BAAI/bge-m3` | 2.3GB | 100+ | Multilingual data (recommended) |
| `BAAI/bge-base-en-v1.5` | 440MB | English | English-only data |
| `all-MiniLM-L6-v2` | 90MB | English | Quick testing |

### Batch Size by Hardware

| Hardware | Recommended `EMBEDDING_BATCH_SIZE` |
|----------|-----------------------------------|
| CPU / Apple Silicon | 32-64 |
| GPU 8-16GB | 64 |
| GPU 24-40GB | 128 |
| GPU 80GB (A100) | 128-256 |

The system automatically reduces batch size for longer documents to prevent memory issues.

## Pipeline Commands

### Steps

```bash
# Individual steps
--export           # SPARQL → TTL
--generate-docs    # TTL → Documents
--embed            # Documents → Embeddings

# Combined
--all              # All three steps
--generate-docs --embed   # Skip export (use existing TTL)
```

### Options

| Option | Description |
|--------|-------------|
| `--dataset <id>` | Dataset from datasets.yaml |
| `--env <file>` | Config file (.env.local, .env.cluster) |
| `--workers <n>` | Parallel workers for document generation |
| `--batch-size <n>` | Embedding batch size |
| `--from-file <path>` | Use specific TTL file |
| `--data-dir <path>` | Override data directory |

### Utilities

```bash
# Check status
python scripts/cluster_pipeline.py --dataset mah --status

# Clean files
python scripts/cluster_pipeline.py --dataset mah --clean
python scripts/cluster_pipeline.py --dataset mah --clean-cache  # Embeddings only
```

## Output Files

After processing:

```
data/
├── exports/mah_dump.ttl              # RDF export
├── documents/mah/
│   └── entity_documents/             # Generated markdown files
└── cache/mah/
    ├── document_graph.pkl            # Document graph
    ├── vector_index/                 # FAISS index
    ├── embeddings/                   # Embedding cache
    └── embedding_stats.json          # Processing statistics
```

### Processing Statistics

The `embedding_stats.json` file records:
- Timing (start, end, duration)
- Document counts and lengths
- Throughput metrics
- Hardware information (CPU, RAM, GPU)
- Model details

## SLURM Job Script

For HPC clusters, use `scripts/pipeline.sbatch`:

```bash
sbatch scripts/pipeline.sbatch
squeue -u $USER
tail -f logs/pipeline_*.out
```

Edit the script for your cluster's partition and account settings.

## Updating Data

When SPARQL data changes:

```bash
# Re-export and process
python scripts/cluster_pipeline.py --dataset mah --export
python scripts/cluster_pipeline.py --dataset mah --generate-docs --embed

# Or on cluster (transfer TTL first)
python scripts/cluster_pipeline.py --dataset mah --generate-docs --embed --env .env.cluster
```
