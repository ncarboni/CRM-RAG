# Cluster Pipeline Guide

Guide for processing large RDF datasets on a GPU cluster.

## Overview

```
LOCAL MACHINE                      GPU CLUSTER                       LOCAL MACHINE
(has SPARQL)                       (no API keys needed)              (has API keys)

1. Export RDF          ──scp──>    2. Generate documents
   data/exports/*.ttl              3. Compute embeddings    ──scp──>  4. Run RAG server
   data/labels/*.json                 (local BGE-M3 model)               (uses LLM API)
                                      data/cache/*
                                      data/documents/*
```

**Key point:** The cluster only processes RDF data and computes embeddings using a local model (BGE-M3). No API keys are needed on the cluster.

## Prerequisites

**Local machine:** Complete the basic setup from [README.md](../README.md) first:
- Install dependencies
- Configure `datasets.yaml` with your dataset
- Extract ontology labels (`python scripts/extract_ontology_labels.py`)

## Part 1: Export RDF Locally

Export your dataset from SPARQL:

```bash
python scripts/cluster_pipeline.py --dataset mah --export
```

Output: `data/exports/mah_dump.ttl`

---

## Part 2: Cluster Setup

```bash
ssh user@cluster

# Clone repo
git clone <your-repo-url> CRM_RAG
cd CRM_RAG

# Load modules BEFORE creating venv (must match sbatch)
module load python/3.11
module load cuda/12.8

# Create venv with loaded Python version
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create data directories (gitignored)
mkdir -p data/{exports,labels,cache,documents} logs

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

The cluster config `config/.env.cluster` is already in the repository (no API keys needed).

---

## Part 3: Transfer and Process

```bash
# === LOCAL: Transfer TTL export ===
scp data/exports/mah_dump.ttl user@cluster:CRM_RAG/data/exports/

# === CLUSTER: Generate labels and process ===
ssh user@cluster
cd CRM_RAG && source venv/bin/activate
python scripts/extract_ontology_labels.py
python scripts/cluster_pipeline.py --dataset mah \
  --generate-docs --embed \
  --workers 32 \
  --env .env.cluster

# === LOCAL: Transfer results back ===
scp -r user@cluster:CRM_RAG/data/cache/mah/ data/cache/mah/
scp -r user@cluster:CRM_RAG/data/documents/mah/ data/documents/mah/
```

For large datasets, use `--data-dir` to store outputs on scratch storage:

```bash
python scripts/cluster_pipeline.py --dataset mah \
  --generate-docs --embed \
  --workers 32 \
  --env .env.cluster \
  --data-dir ~/scratch/CRM_RAG_data
```

---

## Part 4: Run Server Locally

```bash
python main.py --env .env.local
```

Open `http://localhost:5001`

---

## CLI Reference

### Pipeline Steps

```bash
# Full pipeline (all steps, single machine with SPARQL)
python scripts/cluster_pipeline.py --dataset <id> --all --workers 8

# Individual steps
python scripts/cluster_pipeline.py --dataset <id> --export           # SPARQL -> TTL
python scripts/cluster_pipeline.py --dataset <id> --generate-docs    # TTL -> Documents
python scripts/cluster_pipeline.py --dataset <id> --embed            # Documents -> Embeddings

# Combined steps (recommended for cluster)
python scripts/cluster_pipeline.py --dataset <id> --generate-docs --embed --workers 16
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset <id>` | Dataset ID from datasets.yaml | Required |
| `--env <file>` | Config file (e.g., .env.cluster) | .env |
| `--workers <n>` | Parallel workers for doc generation | 1 |
| `--context-depth <0\|1\|2>` | Relationship traversal depth | 2 |
| `--batch-size <n>` | Embedding batch size | 64 |
| `--from-file <path>` | Use specific TTL file | Auto-detect |
| `--data-dir <path>` | Override data directory (e.g., for scratch storage) | `data/` |

### Utilities

```bash
# Check status
python scripts/cluster_pipeline.py --dataset mah --status

# Clean files
python scripts/cluster_pipeline.py --dataset mah --clean          # All
python scripts/cluster_pipeline.py --dataset mah --clean-export   # TTL only
python scripts/cluster_pipeline.py --dataset mah --clean-docs     # Documents only
python scripts/cluster_pipeline.py --dataset mah --clean-cache    # Embeddings only
```

---

## SLURM Job Script

For HPC clusters using SLURM, use `scripts/pipeline.sbatch`:

```bash
sbatch scripts/pipeline.sbatch
squeue -u $USER                    # Check queue
tail -f logs/pipeline_*.out        # Monitor output
```

Edit the script to adjust partition, account, and dataset settings for your cluster.

---

## Performance Reference

### Document Generation (CPU)

| Workers | 100K entities | 500K entities |
|---------|---------------|---------------|
| 1 | 8 min | 40 min |
| 8 | 2 min | 8 min |
| 32 | 45 sec | 3 min |

### Embedding Computation (GPU)

| GPU | 100K entities | 500K entities |
|-----|---------------|---------------|
| A100 (40GB) | 5 min | 25 min |
| RTX 4090 | 10 min | 45 min |
| V100 (16GB) | 15 min | 70 min |

### File Sizes

| Component | Per 100K entities |
|-----------|-------------------|
| TTL export | 20-50 MB |
| Documents | 100-300 MB |
| Document graph | 50-150 MB |
| Vector index | 100-400 MB |

---

## Troubleshooting

### Export fails with timeout

```bash
# Export directly with curl (no timeout)
curl -X POST "http://localhost:3030/MAH/sparql" \
  -H "Accept: text/turtle" \
  --data-urlencode "query=CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }" \
  -o data/exports/mah_dump.ttl
```

### Missing labels error

```
FileNotFoundError: data/labels/property_labels.json
```

Run ontology extraction locally and transfer:
```bash
python scripts/extract_ontology_labels.py
scp -r data/labels/ user@cluster:CRM_RAG/data/labels/
```

### CUDA out of memory

Reduce batch size:
```bash
python scripts/cluster_pipeline.py --dataset mah --embed --batch-size 32
```

### Embedding model mismatch

Both local and cluster must use the same `EMBEDDING_MODEL`. Check both `.env` files match.

---

## Updating Data

When your SPARQL data changes:

```bash
# 1. LOCAL: Re-export
python scripts/cluster_pipeline.py --dataset mah --export
scp data/exports/mah_dump.ttl user@cluster:CRM_RAG/data/exports/

# 2. CLUSTER: Re-process
python scripts/cluster_pipeline.py --dataset mah \
  --generate-docs --embed --workers 16 --env .env.cluster

# 3. LOCAL: Transfer and restart
scp -r user@cluster:CRM_RAG/data/cache/mah/ data/cache/mah/
scp -r user@cluster:CRM_RAG/data/documents/mah/ data/documents/mah/
python main.py --env .env.local
```
