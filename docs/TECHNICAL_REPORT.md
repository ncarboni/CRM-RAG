# Technical Report: Document Processing and Embedding

This report explains the technical processes used in CRM_RAG for document generation and embedding computation.

## 1. Document Generation

### 1.1 RDF to Document Conversion

The system converts RDF triples into natural language documents. Each entity in the knowledge graph becomes a markdown document containing:

- **Header**: Entity label and URI
- **Types**: CIDOC-CRM classes the entity belongs to
- **Properties**: Literal values (labels, descriptions, dates)
- **Relationships**: Connections to other entities, expressed in natural language

Example transformation:

```turtle
# RDF Triples
ex:painting1 rdf:type crm:E22_Human-Made_Object ;
    rdfs:label "The Night Watch" ;
    crm:P62_depicts ex:person1 ;
    crm:P52_has_current_owner ex:museum1 .
```

```markdown
# The Night Watch

URI: http://example.org/painting1

## Types
- Human-Made Object

## Properties
- **Label**: The Night Watch

## Relationships
- The Night Watch depicts Rembrandt
- The Night Watch has current owner Rijksmuseum
```

### 1.2 Relationship Traversal

Documents include relationships up to a configurable depth (default: 2 hops). The traversal uses event-aware logic based on CIDOC-CRM semantics:

- **Events** (E5 subclasses) act as connection points between entities
- For non-event entities, direct relationships are included
- For event entities, the traversal continues through connected events

This captures the temporal and causal structure common in cultural heritage data, where events mediate relationships between objects, people, and places.

### 1.3 Image Indexing

When configured, the system extracts image URLs using a SPARQL query executed against the in-memory RDF graph:

```python
query = f"""
{user_defined_prefixes}
SELECT ?entity ?url WHERE {{
    {user_defined_pattern}
}}
"""
results = graph.query(query)
```

Images are stored in document frontmatter:

```yaml
---
URI: http://example.org/painting1
Label: The Night Watch
Images:
  - https://iiif.example.org/painting1/full/max/0/default.jpg
---
```

During retrieval, local images take priority over Wikidata images (P18 property).

## 2. Embedding Computation

### 2.1 Sentence Transformers

The system uses sentence-transformers models to convert document text into dense vector embeddings. The recommended model is BAAI/bge-m3:

- **Dimensions**: 1024
- **Max sequence length**: 8192 tokens
- **Languages**: 100+
- **Architecture**: XLM-RoBERTa based

Embeddings capture semantic meaning, enabling similarity-based retrieval even when query terms don't exactly match document text.

### 2.2 Dynamic Batching

GPU memory usage during embedding computation follows this relationship:

```
Memory ≈ batch_size × sequence_length² × constant
```

The quadratic dependency on sequence length is due to the attention mechanism. When documents in a batch have different lengths, all are padded to match the longest document. This can cause memory issues:

- 128 documents, all short (500 tokens): ~1 GB
- 128 documents, one long (8000 tokens): ~32 GB (all padded to 8000)

The system addresses this with dynamic batching:

**Step 1: Sort by length**

Documents are sorted by character length, grouping similar-sized documents together. This minimizes padding waste.

**Step 2: Calculate thresholds**

Thresholds are derived from the model's maximum sequence length:

```
max_chars = model.max_seq_length × 4  (approx. chars per token)

very_long:   75% of max_chars → batch_size ÷ 16
long:        50% of max_chars → batch_size ÷ 8
medium_long: 25% of max_chars → batch_size ÷ 4
medium:      12.5% of max_chars → batch_size ÷ 2
short:       < 12.5% of max_chars → full batch_size
```

**Step 3: Adaptive processing**

Each batch uses an appropriate batch size based on the longest document in that batch:

```
Processing batch of 128 texts (max_len=2500 chars, batch_size=128)
Processing batch of 64 texts (max_len=5800 chars, batch_size=64)
Processing batch of 8 texts (max_len=28000 chars, batch_size=8)
```

**Step 4: Memory management**

After processing long document batches, the GPU cache is cleared to free memory for subsequent batches.

### 2.3 Embedding Cache

Embeddings are cached to disk using a content-addressable scheme:

```
cache_key = hash(document_text + model_name)
```

This enables:
- **Resumability**: Processing can stop and continue later
- **Incremental updates**: Only new or changed documents need embedding
- **Cross-session reuse**: Embeddings persist between runs

Cache location: `data/cache/<dataset_id>/embeddings/`

## 3. Document Graph

### 3.1 Graph Structure

Documents are organized as a weighted graph where:
- **Nodes**: Documents (one per entity)
- **Edges**: Relationships between entities
- **Weights**: Based on CIDOC-CRM relationship semantics

Edge weights reflect relationship importance:

| Relationship | Weight | Rationale |
|--------------|--------|-----------|
| P89 falls within (spatial) | 0.9 | Strong spatial context |
| P55 has current location | 0.9 | Strong spatial context |
| P46 is composed of | 0.8 | Part-whole relationship |
| P108i was produced by | 0.7 | Creation relationship |
| P2 has type | 0.6 | Classification |
| Default | 0.5 | Unknown relationship |

### 3.2 Multi-hop Adjacency

The retrieval system builds a multi-hop adjacency matrix to find connected documents:

```
A_multihop = A + (A²)/2 + (A³)/3 + ...
```

Where A is the direct adjacency matrix. The division by hop count reduces the influence of distant connections.

The matrix is then normalized symmetrically:

```
A_normalized = D^(-1/2) × A × D^(-1/2)
```

Where D is the degree matrix. This prevents high-degree nodes from dominating the retrieval.

### 3.3 Coherent Subgraph Extraction

When retrieving documents for a query, the system balances:
- **Relevance**: How well the document matches the query (vector similarity)
- **Connectivity**: How well the document connects to other selected documents

The selection algorithm:

1. Retrieve initial candidates via vector similarity
2. Score candidates using PageRank on the document graph
3. Combine scores: `α × vector_score + (1-α) × pagerank_score`
4. Greedy selection balancing relevance and connectivity

This produces a coherent set of documents that are both relevant to the query and contextually connected to each other.

## 4. Processing Statistics

The system records processing statistics to `embedding_stats.json`:

```json
{
  "timing": {
    "start": "2026-02-04T12:30:00",
    "end": "2026-02-04T12:45:30",
    "elapsed_seconds": 930.5
  },
  "documents": {
    "total": 6773,
    "from_cache": 500,
    "newly_embedded": 6273,
    "length_chars": {
      "min": 1200,
      "max": 28000,
      "avg": 4500
    }
  },
  "performance": {
    "throughput_docs_per_sec": 7.28
  },
  "model": {
    "name": "BAAI/bge-m3",
    "max_seq_length": 8192,
    "embedding_dimension": 1024
  },
  "hardware": {
    "cpu": {"model": "AMD EPYC 7763", "cores_physical": 64},
    "ram_gb": 512,
    "gpu": [{"name": "NVIDIA A100-SXM4-80GB", "memory_gb": 80}]
  }
}
```

This enables:
- Reproducibility documentation
- Performance benchmarking across configurations
- Capacity planning for new datasets

## 5. Implementation References

| Component | File | Key Functions |
|-----------|------|---------------|
| Document generation | `scripts/bulk_generate_documents.py` | `create_document()`, `build_indexes()` |
| Image indexing | `scripts/bulk_generate_documents.py` | `_build_image_index()` |
| Dynamic batching | `llm_providers.py` | `get_embeddings_batch()`, `_get_effective_batch_size()` |
| Embedding cache | `llm_providers.py` | `EmbeddingCache` class |
| Document graph | `graph_document_store.py` | `GraphDocumentStore`, `create_adjacency_matrix()` |
| Retrieval | `universal_rag_system.py` | `cidoc_aware_retrieve()`, `extract_coherent_subgraph()` |
| Statistics | `universal_rag_system.py` | `embed_from_documents()` |
