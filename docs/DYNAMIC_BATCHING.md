# Dynamic Batching for Embeddings

This document explains the dynamic batching system used for embedding computation in CRM_RAG. This feature prevents CUDA out-of-memory (OOM) errors when processing datasets with variable-length documents.

## The Problem

When embedding documents on a GPU, memory usage depends on:

```
Memory ≈ batch_size × max_sequence_length² × constant
```

The critical issue is that **all documents in a batch are padded to match the longest document**. This means:

- If you have 127 short documents (500 tokens each) and 1 long document (8000 tokens)
- All 128 documents get padded to 8000 tokens
- Attention mask memory: `128 × 8000 × 8000 × 4 bytes ≈ 32 GB`

This causes OOM errors even on high-memory GPUs like the A100 (80GB).

## The Solution

Dynamic batching automatically adjusts the batch size based on document length:

1. **Sort documents by length** - Groups similar-sized documents together
2. **Derive thresholds from the model** - Uses `model.max_seq_length` to set thresholds
3. **Scale batch size relatively** - Uses fractions of the configured `EMBEDDING_BATCH_SIZE`
4. **Clear GPU cache** - Frees memory after processing long documents

## How It Works

### Model-Aware Thresholds

Thresholds are calculated as percentages of the model's maximum sequence length:

```
max_chars = model.max_seq_length × 4  (approximate chars per token)

Thresholds:
- very_long:   75% of max_chars
- long:        50% of max_chars
- medium_long: 25% of max_chars
- medium:      12.5% of max_chars
```

For **bge-m3** (max_seq_length=8192 tokens):
```
max_chars = 8192 × 4 = 32,768 chars

Thresholds:
- very_long:   24,576 chars
- long:        16,384 chars
- medium_long:  8,192 chars
- medium:       4,096 chars
```

For **all-MiniLM-L6-v2** (max_seq_length=256 tokens):
```
max_chars = 256 × 4 = 1,024 chars

Thresholds:
- very_long:   768 chars
- long:        512 chars
- medium_long: 256 chars
- medium:      128 chars
```

### Relative Batch Sizing

Batch sizes are calculated as fractions of the configured `EMBEDDING_BATCH_SIZE`:

| Document Length      | Batch Size Divisor | Example (batch=128) | Example (batch=64) |
|----------------------|--------------------|---------------------|--------------------|
| Short (<12.5% max)   | ÷ 1                | 128                 | 64                 |
| Medium (12.5-25%)    | ÷ 2                | 64                  | 32                 |
| Medium-long (25-50%) | ÷ 4                | 32                  | 16                 |
| Long (50-75%)        | ÷ 8                | 16                  | 8                  |
| Very long (>75%)     | ÷ 16               | 8                   | 4                  |

## Configuration

The main configuration parameter is `EMBEDDING_BATCH_SIZE` in your `.env` file:

### config/.env.local

```bash
# Base batch size for short documents
# Dynamic batching reduces this for longer documents
EMBEDDING_BATCH_SIZE=64
```

### config/.env.cluster

```bash
# Higher base batch size for GPU clusters
# Dynamic batching prevents OOM on long documents
EMBEDDING_BATCH_SIZE=128
```

## Choosing the Right Batch Size

### For Local Development (CPU/MPS)

```bash
# Conservative - works on most systems
EMBEDDING_BATCH_SIZE=32

# Faster - requires 16GB+ RAM
EMBEDDING_BATCH_SIZE=64
```

### For GPU Servers

| GPU Memory | Recommended Base Batch Size |
|------------|----------------------------|
| 8 GB       | 32                         |
| 16 GB      | 64                         |
| 24 GB      | 96                         |
| 40 GB      | 128                        |
| 80 GB      | 128-256                    |

Note: These are base sizes for short documents. Long documents will automatically use smaller batches.

## Example: Processing the MAH Dataset

The MAH (Musée d'Art et d'Histoire) dataset contains ~6,700 documents ranging from 1KB to 30KB:

**Without dynamic batching:**
```
Batch 2026/6773 - OOM Error
Tried to allocate 26.93 GiB
All 128 documents padded to max length in batch
```

**With dynamic batching:**
```
Processing batch of 128 texts (max_len=2500 chars, batch_size=128)
Processing batch of 128 texts (max_len=3200 chars, batch_size=128)
Processing batch of 64 texts (max_len=5800 chars, batch_size=64)
Processing batch of 32 texts (max_len=12000 chars, batch_size=32)
Processing batch of 8 texts (max_len=28000 chars, batch_size=8)
Completed embedding 6773 texts in 87 batches
```

## Monitoring

The system logs detailed information about each batch:

```
INFO - Processing batch of 128 texts (max_len=2500 chars, batch_size=128)
INFO - Processing batch of 8 texts (max_len=28000 chars, batch_size=8)
INFO - Completed embedding 6773 texts in 87 batches
```

Key metrics to watch:
- `max_len`: Maximum document length in the batch (characters)
- `batch_size`: Effective batch size used
- Total batches: More batches with long documents is normal

## Troubleshooting

### Still Getting OOM Errors

1. **Reduce base batch size:**
   ```bash
   EMBEDDING_BATCH_SIZE=64  # Instead of 128
   ```

2. **Check for extremely long documents:**
   ```bash
   find data/documents -name "*.md" -type f -printf '%s %p\n' | sort -rn | head -20
   ```

3. **Enable expandable CUDA segments** (in sbatch script):
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

### Processing is Slower Than Expected

This is normal when you have many long documents. The tradeoff:
- Smaller batches = less memory, slower processing
- Larger batches = more memory, faster processing

If most documents are short but a few are very long, consider:
1. Reviewing why some documents are so long (data quality issue?)
2. Processing long documents separately with a smaller batch size

### Batch Sizes Seem Too Conservative

The default divisors (÷2, ÷4, ÷8, ÷16) are conservative to handle worst-case scenarios. If you have lots of GPU memory and want to be more aggressive:

1. Increase the base batch size:
   ```bash
   EMBEDDING_BATCH_SIZE=256  # More aggressive
   ```

2. The dynamic batching will still protect against OOM for the longest documents.

## Technical Details

### Implementation Location

The dynamic batching logic is in `llm_providers.py`:

- `SentenceTransformersProvider.get_embeddings_batch()` - Main entry point
- `SentenceTransformersProvider._get_length_thresholds()` - Calculates model-aware thresholds
- `SentenceTransformersProvider._get_effective_batch_size()` - Determines batch size

### Algorithm Pseudocode

```python
def get_embeddings_batch(texts):
    # 1. Get model-aware thresholds
    thresholds = calculate_thresholds(model.max_seq_length)

    # 2. Sort texts by length (preserving original order for output)
    sorted_texts = sort_by_length_with_indices(texts)

    # 3. Process in adaptive batches
    while texts_remaining:
        max_len = get_max_length_in_next_batch()
        batch_size = get_effective_batch_size(max_len, thresholds)

        batch = extract_batch(sorted_texts, batch_size)
        embeddings = model.encode(batch)

        # Clear GPU cache for long documents
        if max_len > threshold_medium:
            torch.cuda.empty_cache()

    # 4. Reorder embeddings to match original input order
    return reorder_embeddings(embeddings)
```

### Memory Estimation Formula

For transformer attention with padding:

```
Attention Memory = batch_size × seq_length × seq_length × dtype_size

Where:
- batch_size: Number of documents in batch
- seq_length: Length of LONGEST document (due to padding)
- dtype_size: 4 bytes for float32, 2 bytes for float16

Example (worst case):
- batch_size = 128
- seq_length = 8192 tokens (bge-m3 max)
- dtype_size = 4 bytes

Memory = 128 × 8192 × 8192 × 4 = 34.4 GB (just for attention mask!)
```

This is why dynamic batching is essential for variable-length documents.

## References

- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [BGE-M3 Model Card](https://huggingface.co/BAAI/bge-m3)
