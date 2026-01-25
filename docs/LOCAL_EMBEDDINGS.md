# Local Embeddings Guide

This guide explains how to use local embeddings with sentence-transformers for fast, cost-free processing of large datasets.

## Overview

By default, the system uses API-based embeddings (OpenAI, etc.) which have rate limits and costs. For large datasets (5,000+ entities), this can take days and cost significant money.

**Local embeddings solve this by:**
- Running embeddings on your machine (CPU or GPU)
- No API rate limits - process as fast as your hardware allows
- No costs - completely free
- Privacy - data never leaves your machine
- Resumability - embedding cache allows stopping and continuing

### Performance Comparison

| Method | 50,000 entities | Cost |
|--------|-----------------|------|
| OpenAI API | 2-4 days | ~$10-20 |
| Local (CPU) | 1-2 hours | Free |
| Local (GPU) | 10-20 minutes | Free |

## Quick Start

### 1. Install Dependencies

```bash
pip install sentence-transformers torch
```

Or install all requirements (includes these):
```bash
pip install -r requirements.txt
```

### 2. Set Up Configuration

```bash
# Copy the local embeddings config template
cp config/.env.local.example config/.env.local
```

Edit `config/.env.local` - the key settings are:
```bash
LLM_PROVIDER=openai           # Still need LLM for answering questions
EMBEDDING_PROVIDER=local      # Use local embeddings
EMBEDDING_MODEL=BAAI/bge-m3   # Sentence-transformers model
```

**Note:** SPARQL endpoints are defined in `config/datasets.yaml`, not in `.env` files.

Make sure your API key is in `config/.env.secrets` (needed for chat):
```bash
OPENAI_API_KEY=sk-your-key-here
```

### 3. Process Your Dataset

```bash
# Process a specific dataset
python main.py --env .env.local --dataset asinou --rebuild --process-only

# Or start the server (will process on first access)
python main.py --env .env.local
```

## Configuration Options

### Environment Variables

Add these to your `.env` file or `config/.env.local`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | (same as LLM) | `local`, `sentence-transformers`, `openai`, `ollama` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Model name from HuggingFace |
| `EMBEDDING_BATCH_SIZE` | `64` | Documents per batch (higher = faster, more memory) |
| `EMBEDDING_DEVICE` | `auto` | `auto`, `cuda`, `mps` (Apple Silicon), `cpu` |
| `USE_EMBEDDING_CACHE` | `true` | Cache embeddings to disk for resumability |

### CLI Flags

Override config settings from command line:

```bash
python main.py --env .env.local \
  --dataset asinou \
  --embedding-provider local \
  --embedding-model all-MiniLM-L6-v2 \
  --rebuild \
  --process-only
```

| Flag | Description |
|------|-------------|
| `--embedding-provider <name>` | `local`, `sentence-transformers`, `openai`, `ollama` |
| `--embedding-model <model>` | HuggingFace model name |
| `--no-embedding-cache` | Disable caching (force re-embed everything) |
| `--dataset <id>` | Which dataset to process (from `datasets.yaml`) |
| `--process-only` | Exit after processing (don't start web server) |
| `--rebuild` | Clear cache and rebuild from scratch |

## Per-Dataset Configuration

You can configure different embedding providers for each dataset in `config/datasets.yaml`:

```yaml
default_dataset: asinou

datasets:
  # Small dataset - use local embeddings (fast)
  asinou:
    name: asinou
    display_name: "Asinou Church"
    endpoint: "http://localhost:3030/asinou/sparql"
    embedding:
      provider: local
      model: BAAI/bge-m3
      batch_size: 64

  # Large dataset - also use local (or OpenAI if you prefer)
  museum:
    name: museum
    display_name: "Museum Collection"
    endpoint: "http://localhost:3030/museum/sparql"
    embedding:
      provider: local
      model: BAAI/bge-m3

  # Dataset where you want OpenAI quality
  special:
    name: special
    endpoint: "http://localhost:3030/special/sparql"
    embedding:
      provider: openai
      # model inherited from .env config
```

### Processing Multiple Datasets

```bash
# Process each dataset (they'll use their configured embedding provider)
python main.py --env .env.local --dataset asinou --rebuild --process-only
python main.py --env .env.local --dataset museum --rebuild --process-only

# Start web server (both datasets now cached)
python main.py --env .env.local
```

## Embedding Models

### Recommended Models

| Model | Dims | Size | Speed | Quality | Languages |
|-------|------|------|-------|---------|-----------|
| `BAAI/bge-m3` | 1024 | 2.3GB | Medium | Best | 100+ |
| `BAAI/bge-base-en-v1.5` | 768 | 440MB | Fast | Excellent | English |
| `all-MiniLM-L6-v2` | 384 | 90MB | Very Fast | Good | English |
| `all-mpnet-base-v2` | 768 | 420MB | Fast | Very Good | English |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 470MB | Fast | Good | 50+ |

### Model Selection Guide

**For cultural heritage data (recommended):** `BAAI/bge-m3`
- Multilingual support (labels in Latin, Greek, local languages)
- Long context (8192 tokens vs 512 for most models)
- State-of-the-art quality

**For English-only data:** `BAAI/bge-base-en-v1.5`
- Smaller and faster than bge-m3
- Excellent quality for English

**For quick testing:** `all-MiniLM-L6-v2`
- Very small and fast
- Good enough for testing pipelines

### Using a Different Model

```bash
# Via CLI
python main.py --env .env.local --embedding-model all-MiniLM-L6-v2 --rebuild

# Via config (.env.local)
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Hardware Acceleration

### Automatic Detection

By default (`EMBEDDING_DEVICE=auto`), the system detects:
1. **CUDA GPU** - NVIDIA GPUs (fastest)
2. **MPS** - Apple Silicon M1/M2/M3 (fast)
3. **CPU** - Fallback (still much faster than API)

### Force Specific Device

```bash
# In .env.local
EMBEDDING_DEVICE=cuda  # Force NVIDIA GPU
EMBEDDING_DEVICE=mps   # Force Apple Silicon
EMBEDDING_DEVICE=cpu   # Force CPU
```

### Memory Considerations

If you run out of memory:

```bash
# Reduce batch size
EMBEDDING_BATCH_SIZE=32  # Default is 64

# Or use a smaller model
EMBEDDING_MODEL=all-MiniLM-L6-v2  # 90MB vs 2.3GB
```

## Embedding Cache

Embeddings are cached to disk by default, allowing:
- **Resumability** - Stop and continue later
- **Fast rebuilds** - Only new entities need embedding

### Cache Location

```
data/cache/<dataset_id>/embeddings/
```

### Managing the Cache

```bash
# Clear cache for a dataset (force re-embed)
rm -rf data/cache/asinou/embeddings/

# Or use CLI flag
python main.py --env .env.local --dataset asinou --no-embedding-cache --rebuild

# Check cache stats (shown in logs during processing)
# "Embedding cache: 500 cached embeddings (125.5 MB)"
```

## Troubleshooting

### "No sentence-transformers model found"

**Error:**
```
No sentence-transformers model found with name text-embedding-3-small
```

**Cause:** Using OpenAI model name with local embeddings.

**Fix:** The system should auto-correct this now. If not, specify the model:
```bash
python main.py --env .env.local --embedding-model BAAI/bge-m3
```

### Out of Memory

**Error:**
```
CUDA out of memory / Cannot allocate memory
```

**Fix:** Reduce batch size or use smaller model:
```bash
EMBEDDING_BATCH_SIZE=16
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Slow on CPU

**Cause:** No GPU detected, using CPU.

**Check:** Look for log message:
```
INFO:llm_providers:No GPU detected, using CPU for embeddings
```

**Options:**
1. This is normal - CPU is still 10-100x faster than API
2. Install CUDA toolkit for NVIDIA GPU support
3. Use Apple Silicon Mac (MPS is auto-detected)

### Model Download Fails

**Error:**
```
HTTPError: 403 Client Error: Forbidden
```

**Fix:** Some models require accepting terms on HuggingFace:
1. Go to the model page (e.g., https://huggingface.co/BAAI/bge-m3)
2. Accept the terms
3. Login: `huggingface-cli login`

Or use a model that doesn't require acceptance:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Complete Example

### Scenario: Two datasets, different sizes

```yaml
# config/datasets.yaml
default_dataset: church

datasets:
  church:
    name: church
    display_name: "Church Dataset"
    endpoint: "http://localhost:3030/church/sparql"
    embedding:
      provider: local
      model: BAAI/bge-m3

  museum:
    name: museum
    display_name: "Museum (50k entities)"
    endpoint: "http://localhost:3030/museum/sparql"
    embedding:
      provider: local
      model: BAAI/bge-m3
      batch_size: 128  # Larger batches for big dataset
```

```bash
# config/.env.local
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=auto
USE_EMBEDDING_CACHE=true
TEMPERATURE=0.7
PORT=5001
# Note: SPARQL endpoints are defined in datasets.yaml, not here
```

```bash
# Process both datasets
python main.py --env .env.local --dataset church --rebuild --process-only
python main.py --env .env.local --dataset museum --rebuild --process-only

# Start server
python main.py --env .env.local
```

## Comparison with API Embeddings

| Aspect | Local (sentence-transformers) | API (OpenAI) |
|--------|------------------------------|--------------|
| Speed | 10-100x faster | Rate limited |
| Cost | Free | ~$0.0001/1K tokens |
| Privacy | Data stays local | Sent to API |
| Quality | Very good (bge-m3) | Excellent |
| Setup | Install packages | Just API key |
| Resumability | Built-in cache | Need custom solution |
| GPU support | CUDA, MPS, CPU | N/A |

**Recommendation:** Use local embeddings for processing, especially for large datasets. The quality difference is minimal and the speed/cost benefits are significant.

## Processing on a GPU Cluster

For large datasets or if your local machine lacks GPU memory, you can process embeddings on a GPU cluster and transfer the cache to your local machine.

See **[CLUSTER_PIPELINE.md](CLUSTER_PIPELINE.md)** for detailed instructions.
