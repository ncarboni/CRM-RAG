# Computing Embeddings on a GPU Cluster

This guide explains how to compute embeddings on a GPU cluster and transfer them to your local machine for serving.

## Overview

**Why compute on a cluster?**
- Large embedding models (1-2GB) need significant GPU memory
- Batch processing is 10-100x faster on GPU
- Process once on cluster, serve anywhere

**Two workflows available:**

1. **Standard Workflow** (SPARQL accessible from cluster):
   ```
   [GPU Cluster with SPARQL] → Process embeddings → [Cache] → Copy to local → [Serve]
   ```

2. **No-SPARQL Workflow** (SPARQL only on local machine):
   ```
   [Local] → Generate docs → [Documents] → Copy to cluster → [Embed on GPU] → [Cache] → Copy to local → [Serve]
   ```

Choose **Workflow 2** if your SPARQL endpoint is not accessible from the cluster (e.g., local Fuseki, firewall restrictions).

## Prerequisites

### On the Cluster

- Python 3.9+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- **Workflow 1 only:** Access to your SPARQL endpoint

### On Your Local Machine

- Python 3.9+
- Same embedding model installed (for query-time embedding)

## Step 1: Set Up the Cluster Environment

### Clone the Repository

```bash
# On cluster
git clone <your-repo-url> CRM_RAG
cd CRM_RAG
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA A100-SXM4-40GB
```

## Step 2: Configure for GPU Processing

### Create Cluster Config

```bash
# On cluster
cp config/.env.local.example config/.env.cluster
```

### Edit `config/.env.cluster`

```bash
# LLM Provider (still needed for any LLM calls, can be minimal)
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o

# Embedding Configuration - optimized for GPU cluster
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1  # High quality, 1024 dims
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=128  # Large batches for GPU efficiency

# Cache settings
USE_EMBEDDING_CACHE=true

# Server settings (not used for processing, but required)
TEMPERATURE=0.7
PORT=5001
```

### Create API Secrets (if needed)

```bash
cp config/.env.secrets.example config/.env.secrets
# Edit and add your OPENAI_API_KEY if needed for LLM
```

### Configure Dataset

Make sure `config/datasets.yaml` has your dataset configured:

```yaml
default_dataset: asinou

datasets:
  asinou:
    name: asinou
    display_name: "Asinou Church"
    endpoint: "http://your-sparql-server:3030/asinou/sparql"  # Must be accessible from cluster
    embedding:
      provider: local
      model: mixedbread-ai/mxbai-embed-large-v1
      batch_size: 128
```

**Note:** The SPARQL endpoint must be accessible from the cluster. Options:
- Run Fuseki on the cluster
- Use a publicly accessible endpoint
- Set up SSH tunnel to your local Fuseki

## Step 3: Run Embedding Computation

### Process a Single Dataset

```bash
# On cluster
python main.py --env .env.cluster --dataset asinou --rebuild --process-only
```

### Process Multiple Datasets

```bash
# Process each dataset
python main.py --env .env.cluster --dataset asinou --rebuild --process-only
python main.py --env .env.cluster --dataset museum --rebuild --process-only
```

### Monitor Progress

You should see output like:
```
INFO:llm_providers:CUDA GPU detected: NVIDIA A100-SXM4-40GB
INFO:llm_providers:Loading embedding model: mixedbread-ai/mxbai-embed-large-v1 on cuda
INFO:universal_rag_system:Generating embeddings for 128 documents in batch...
INFO:llm_providers:Batch embedding 128 texts with batch_size=128
Batches: 100%|████████████████████████████████████| 6/6 [00:12<00:00,  2.10s/it]
```

### Expected Processing Times

| Dataset Size | GPU (A100) | GPU (RTX 3090) | CPU |
|--------------|------------|----------------|-----|
| 700 entities | 1-2 min | 2-3 min | 15-30 min |
| 5,000 entities | 5-10 min | 10-15 min | 1-2 hours |
| 50,000 entities | 30-60 min | 1-2 hours | 10-20 hours |

## Step 4: Transfer Cache to Local Machine

### What to Transfer

The cache is stored in `data/cache/<dataset_id>/`:
```
data/cache/asinou/
├── document_graph.pkl      # Document graph with embeddings
├── vector_index/           # FAISS vector index
│   ├── index.faiss
│   └── index.pkl
└── embeddings/             # Individual embedding cache (optional)
```

### Transfer via SCP

```bash
# From your local machine
scp -r user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/
```

### Transfer via rsync (recommended for large datasets)

```bash
# From your local machine
rsync -avz --progress user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/
```

### Transfer Multiple Datasets

```bash
# Transfer all cached datasets
rsync -avz --progress user@cluster:~/CRM_RAG/data/cache/ ./data/cache/
```

## Step 5: Configure Local Machine

### Create Local Config

```bash
# On local machine
cp config/.env.local.example config/.env.local
```

### Edit `config/.env.local`

```bash
# LLM Provider (for answering questions)
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o

# IMPORTANT: Must use SAME embedding model as cluster
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1
EMBEDDING_DEVICE=cpu  # or mps on Apple Silicon
EMBEDDING_BATCH_SIZE=1  # Only single queries, doesn't matter

USE_EMBEDDING_CACHE=true
TEMPERATURE=0.7
PORT=5001
```

**Critical:** The `EMBEDDING_MODEL` must be identical to what was used on the cluster.

### Ensure API Keys

```bash
# config/.env.secrets
OPENAI_API_KEY=sk-your-key-here
```

## Step 6: Run Locally

```bash
python main.py --env .env.local
```

You should see:
```
INFO:universal_rag_system:Document graph loaded from data/cache/asinou/document_graph.pkl with 692 documents
INFO:universal_rag_system:Vector store loaded successfully
Starting Flask application...
Running on http://localhost:5001
```

No re-embedding needed - it loads directly from cache.

---

## Alternative: No-SPARQL Workflow

Use this workflow when your SPARQL endpoint is not accessible from the cluster (e.g., local Fuseki behind a firewall).

### Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ LOCAL MACHINE (with SPARQL access)                                          │
│                                                                             │
│  1. Generate documents from SPARQL                                          │
│     python main.py --env .env.local --dataset asinou \                      │
│       --generate-docs-only --process-only                                   │
│                                                                             │
│  Output: data/documents/asinou/                                             │
│          ├── entity_documents/*.md                                          │
│          └── documents_metadata.json                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Transfer documents
┌─────────────────────────────────────────────────────────────────────────────┐
│ GPU CLUSTER (no SPARQL needed)                                              │
│                                                                             │
│  2. Generate embeddings from document files                                 │
│     python main.py --env .env.cluster --dataset asinou \                    │
│       --embed-from-docs --process-only                                      │
│                                                                             │
│  Output: data/cache/asinou/                                                 │
│          ├── document_graph.pkl                                             │
│          └── vector_index/                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Transfer cache
┌─────────────────────────────────────────────────────────────────────────────┐
│ LOCAL MACHINE (serving)                                                     │
│                                                                             │
│  3. Run the server with cached embeddings                                   │
│     python main.py --env .env.local                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Generate Documents Locally

On your local machine (with SPARQL access):

```bash
# Generate documents only (no embeddings)
python main.py --env .env.local --dataset asinou --generate-docs-only --process-only
```

This creates:
```
data/documents/asinou/
├── entity_documents/
│   ├── entity_001.md
│   ├── entity_002.md
│   └── ...
└── documents_metadata.json    # Entity URIs and relationships
```

The `documents_metadata.json` file contains:
- Entity URIs mapped to document IDs
- Relationship information for building the graph
- Labels and metadata for each entity

### Step 2: Transfer Documents to Cluster

```bash
# From local machine
scp -r data/documents/asinou/ user@cluster:~/CRM_RAG/data/documents/asinou/

# Or use rsync for large datasets
rsync -avz --progress data/documents/asinou/ user@cluster:~/CRM_RAG/data/documents/asinou/
```

### Step 3: Generate Embeddings on Cluster

On the cluster (no SPARQL needed):

```bash
# Embed from existing documents
python main.py --env .env.cluster --dataset asinou --embed-from-docs --process-only
```

You should see:
```
INFO:universal_rag_system:Loading documents from data/documents/asinou/
INFO:universal_rag_system:Found 692 document files
INFO:universal_rag_system:Generating embeddings for 692 documents...
INFO:llm_providers:CUDA GPU detected: NVIDIA A100-SXM4-40GB
Batches: 100%|████████████████████████████████████| 11/11 [00:15<00:00,  1.40s/it]
INFO:universal_rag_system:Building edges from document relationships...
INFO:universal_rag_system:Document graph saved to data/cache/asinou/document_graph.pkl
```

### Step 4: Transfer Cache Back to Local

```bash
# From local machine
scp -r user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/

# Or rsync
rsync -avz --progress user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/
```

### Step 5: Run Locally

```bash
python main.py --env .env.local
```

### Complete Example

```bash
# === LOCAL MACHINE ===

# 1. Generate documents (requires SPARQL)
python main.py --env .env.local --dataset asinou --generate-docs-only --process-only

# 2. Transfer to cluster
rsync -avz data/documents/asinou/ user@cluster:~/CRM_RAG/data/documents/asinou/


# === GPU CLUSTER ===

# 3. Generate embeddings (no SPARQL needed)
python main.py --env .env.cluster --dataset asinou --embed-from-docs --process-only


# === LOCAL MACHINE ===

# 4. Transfer cache back
rsync -avz user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/

# 5. Start server
python main.py --env .env.local
```

### Processing Multiple Datasets

```bash
# Local: Generate docs for all datasets
for dataset in asinou museum special; do
  python main.py --env .env.local --dataset $dataset --generate-docs-only --process-only
done

# Transfer all documents
rsync -avz data/documents/ user@cluster:~/CRM_RAG/data/documents/

# Cluster: Embed all datasets
for dataset in asinou museum special; do
  python main.py --env .env.cluster --dataset $dataset --embed-from-docs --process-only
done

# Transfer all caches back
rsync -avz user@cluster:~/CRM_RAG/data/cache/ ./data/cache/
```

### Updating Data

When your SPARQL data changes:

1. **Regenerate documents locally:**
   ```bash
   python main.py --env .env.local --dataset asinou --generate-docs-only --rebuild --process-only
   ```

2. **Transfer and re-embed on cluster:**
   ```bash
   rsync -avz data/documents/asinou/ user@cluster:~/CRM_RAG/data/documents/asinou/
   # On cluster:
   python main.py --env .env.cluster --dataset asinou --embed-from-docs --process-only
   ```

3. **Transfer cache and restart:**
   ```bash
   rsync -avz user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/
   python main.py --env .env.local
   ```

---

## Recommended Embedding Models

| Model | Dims | Quality | VRAM Needed | Best For |
|-------|------|---------|-------------|----------|
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | Excellent | 6GB | General use |
| `BAAI/bge-large-en-v1.5` | 1024 | Excellent | 6GB | English only |
| `intfloat/e5-large-v2` | 1024 | Excellent | 6GB | General use |
| `BAAI/bge-m3` | 1024 | Excellent | 10GB | Multilingual |
| `intfloat/multilingual-e5-large` | 1024 | Very Good | 8GB | Multilingual |

**For cultural heritage data with multiple languages:** Use `bge-m3` or `multilingual-e5-large`

**For English-only data:** Use `mxbai-embed-large-v1` or `bge-large-en-v1.5`

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
EMBEDDING_BATCH_SIZE=64  # or 32, 16
```

### SPARQL Endpoint Not Accessible

Options:
1. **Use the No-SPARQL Workflow (Recommended):**
   Generate documents locally, transfer to cluster, embed without SPARQL.
   See the "Alternative: No-SPARQL Workflow" section above.

2. **Run Fuseki on cluster:**
   ```bash
   # Upload your data and run Fuseki on cluster
   java -jar fuseki-server.jar --mem /asinou
   ```

3. **SSH tunnel:**
   ```bash
   # On cluster, tunnel to your local Fuseki
   ssh -L 3030:localhost:3030 your-local-machine
   ```

4. **Use public endpoint** (if available)

### Model Mismatch Error

If you see embedding dimension errors, the models don't match:
```
ValueError: Embedding dimension mismatch: expected 1024, got 768
```

Ensure both cluster and local use the exact same `EMBEDDING_MODEL`.

### Cache Transfer Incomplete

Verify all files transferred:
```bash
ls -la data/cache/asinou/
# Should show:
# document_graph.pkl
# vector_index/index.faiss
# vector_index/index.pkl
```

## Quick Reference

### Standard Workflow (SPARQL on Cluster)

```bash
# Process single dataset on cluster
python main.py --env .env.cluster --dataset asinou --rebuild --process-only

# Transfer cache to local
scp -r user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/

# Run locally
python main.py --env .env.local
```

### No-SPARQL Workflow

```bash
# 1. LOCAL: Generate documents
python main.py --env .env.local --dataset asinou --generate-docs-only --process-only

# 2. Transfer documents to cluster
rsync -avz data/documents/asinou/ user@cluster:~/CRM_RAG/data/documents/asinou/

# 3. CLUSTER: Embed from documents (no SPARQL needed)
python main.py --env .env.cluster --dataset asinou --embed-from-docs --process-only

# 4. Transfer cache back to local
rsync -avz user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/

# 5. LOCAL: Run server
python main.py --env .env.local
```

### Transfer Commands

```bash
# Single dataset cache
scp -r user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/

# All cached datasets
rsync -avz user@cluster:~/CRM_RAG/data/cache/ ./data/cache/

# Documents (for no-SPARQL workflow)
rsync -avz data/documents/asinou/ user@cluster:~/CRM_RAG/data/documents/asinou/
```

### Local Commands

```bash
# Start server (uses cached embeddings)
python main.py --env .env.local

# Verify cache is loaded (check logs for "loaded from")
```

## Updating Embeddings

When your data changes:

1. **On cluster:** Re-run processing
   ```bash
   python main.py --env .env.cluster --dataset asinou --rebuild --process-only
   ```

2. **Transfer:** Copy new cache
   ```bash
   rsync -avz user@cluster:~/CRM_RAG/data/cache/asinou/ ./data/cache/asinou/
   ```

3. **Local:** Restart server
   ```bash
   python main.py --env .env.local
   ```

The embedding model must remain the same, or you need to rebuild the entire cache.
