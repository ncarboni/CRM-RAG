# Architecture Review and Cleanup Proposal

**Date**: 2026-02-09
**Scope**: Structural cleanup only — no logic, algorithm, or behavior changes

## Current State

### File Inventory

| File | Lines | Responsibility |
|------|------:|----------------|
| `universal_rag_system.py` | 5,084 | Everything: config loading, SPARQL queries, document generation, FR doc generation, embedding/indexing, retrieval, answer generation, Wikidata integration, validation reporting |
| `scripts/bulk_generate_documents.py` | 1,148 | Standalone document generation with RDF export, FR traversal, multiprocessing |
| `llm_providers.py` | 1,133 | LLM/embedding provider abstraction (5 providers) |
| `fr_traversal.py` | 646 | FR path matching and document formatting |
| `scripts/cluster_pipeline.py` | 557 | Pipeline orchestration (export, generate, embed) |
| `scripts/extract_ontology_labels.py` | 436 | Ontology metadata extraction |
| `main.py` | 428 | Flask web app, CLI, initialization |
| `graph_document_store.py` | 289 | Graph storage + FAISS vector retrieval |
| `embedding_cache.py` | 224 | Disk-based embedding cache |
| `dataset_manager.py` | 196 | Multi-dataset lazy loading |
| `config_loader.py` | 168 | .env and YAML config loading |

---

## Identified Problems

### 1. `universal_rag_system.py` is a 5,084-line god object

It handles 7 distinct pipeline stages that have no reason to be in the same file:

| Stage | Methods | Approx Lines | Coupling |
|-------|---------|----------:|----------|
| Ontology/config loading | `_load_property_labels`, `_load_ontology_classes`, `_load_class_labels`, `_load_inverse_properties`, `_load_event_classes`, `get_event_classes` | ~280 | Read-only JSON loading, no deps on class state |
| SPARQL batch queries | `_batch_query_tsv`, `_escape_uri_for_values`, `_batch_fetch_types`, `_batch_query_outgoing`, `_batch_query_incoming`, `_batch_fetch_literals`, `_batch_fetch_type_labels`, `_batch_fetch_wikidata_ids`, `get_entities_context_batch`, `_build_image_index` | ~750 | Only needs `self.sparql` (SPARQLWrapper instance) |
| Document generation | `generate_documents_only`, `_generate_documents_batch`, `_generate_documents_individual`, `_identify_satellites_from_prefetched`, `create_enhanced_document`, `_create_document_from_prefetched`, `_create_fr_document_from_prefetched`, `save_entity_document`, `_log_generation_complete` | ~850 | Calls SPARQL batch methods + FR traversal |
| Embedding/indexing | `embed_from_documents`, `process_rdf_data`, `_process_batch_embeddings`, `_process_sequential_embeddings`, `build_vector_store_batched`, `_save_edges_parquet`, `_build_edges_from_parquet`, `_load_triples_index` | ~900 | Reads/writes files, calls embedding provider |
| Document formatting utilities | `is_schema_predicate`, `is_technical_class_name`, `process_cidoc_relationship`, `get_relationship_weight` | ~150 | Pure functions on strings/URIs |
| Retrieval | `retrieve`, `cidoc_aware_retrieval`, `compute_coherent_subgraph`, `normalize_scores` | ~300 | Needs document_store + adjacency matrix |
| Answer generation + Wikidata | `answer_question`, `_build_graph_context`, `_build_triples_enrichment`, `_get_entity_label_from_triples`, `get_cidoc_system_prompt`, `get_wikidata_for_entity`, `fetch_wikidata_info`, `_fetch_wikidata_id_from_sparql` | ~500 | Calls retrieval + LLM provider |
| Legacy individual SPARQL | `get_all_entities`, `get_entity_literals`, `get_entity_context`, `get_entity_types_cached`, `get_entity_label` | ~350 | Only used by `_generate_documents_individual` and `create_enhanced_document` |

### 2. Cluster workflow is dead weight

The cluster workflow (export RDF → transfer → generate docs → embed on GPU cluster) consists of:

| Component | Type | Lines |
|-----------|------|------:|
| `scripts/bulk_generate_documents.py` | Script | 1,148 |
| `scripts/cluster_pipeline.py` | Script | 557 |
| `config/.env.cluster` | Config | ~15 |
| `docs/PROCESSING.md` | Doc | ~200 |
| `generate_documents_only()` | Method in `universal_rag_system.py` | ~45 |
| `_generate_documents_individual()` + `create_enhanced_document()` | Methods in `universal_rag_system.py` | ~250 |
| `embed_from_documents()` | Method in `universal_rag_system.py` | ~250 |
| `--generate-docs-only`, `--embed-from-docs`, `--no-batch`, `--process-only` | CLI args in `main.py` | ~20 |
| `_log_generation_complete()` | Method in `universal_rag_system.py` | ~15 |

`bulk_generate_documents.py` duplicates most of `universal_rag_system.py`'s document generation with a different approach (rdflib graph vs SPARQL queries). The two implementations have already diverged (different frontmatter formats, different function signatures). This is **~2,300 lines** of code serving a workflow that can be revisited later.

The main pipeline (`initialize()` → `process_rdf_data()` with `--rebuild`) handles both document generation and embedding in one pass with a live SPARQL endpoint. This is the only path that needs to exist.

### 3. `process_rdf_data()` duplicates `_generate_documents_batch()` + embedding

`process_rdf_data()` (~320 lines) is the combined generate+embed path called from `initialize()` when no cache exists. It re-implements the same chunked batch-SPARQL+FR logic as `_generate_documents_batch()`, but also interleaves embedding. This creates two parallel code paths for the same document generation logic.

With the cluster workflow removed, `_generate_documents_batch()` is only called from `generate_documents_only()` which is itself only for the cluster split. So `process_rdf_data()` becomes the sole document generation path, and the duplication disappears.

### 4. Stale planning documents

- `docs/SPARQL_OPTIMIZATION_PLAN.md` — batch SPARQL was implemented; plan is done
- `docs/IMAGE_INTEGRATION_PLAN.md` — image index was implemented; plan is done
- `docs/PROCESSING.md` — describes cluster workflow being removed

### 5. `main.py` has `load_interface_config()` which is configuration loading

This function belongs with the other config loading in `config_loader.py`.

---

## Proposed Structure

The core principle: **each file owns exactly one pipeline stage**. Functions move to the file that owns their stage. No logic changes.

### Files to create

| New File | Responsibility | Approx Lines | Source |
|----------|---------------|----------:|--------|
| `sparql_helpers.py` | All batch SPARQL query methods | ~750 | Extracted from `universal_rag_system.py` |
| `document_formatter.py` | Predicate filtering, relationship-to-NL, class name checks, relationship weights | ~150 | Extracted from `universal_rag_system.py` |

### Files to modify

| File | Change |
|------|--------|
| `universal_rag_system.py` | Remove extracted methods, import from new modules. Remove cluster-only code: `generate_documents_only()`, `_generate_documents_batch()`, `_generate_documents_individual()`, `create_enhanced_document()`, `_create_document_from_prefetched()`, `embed_from_documents()`, `_log_generation_complete()`. |
| `config_loader.py` | Absorb `load_interface_config()` from `main.py` |
| `main.py` | Remove cluster CLI args (`--generate-docs-only`, `--embed-from-docs`, `--no-batch`, `--process-only`). Import `load_interface_config` from `config_loader`. |

### Files to delete

| File | Reason |
|------|--------|
| `scripts/bulk_generate_documents.py` | Cluster workflow removed. Document generation duplicated in `process_rdf_data()`. |
| `scripts/cluster_pipeline.py` | Cluster workflow removed. |
| `config/.env.cluster` | Cluster config no longer needed. |
| `docs/PROCESSING.md` | Documents the removed cluster workflow. |
| `docs/SPARQL_OPTIMIZATION_PLAN.md` | Implemented — batch SPARQL is done. |
| `docs/IMAGE_INTEGRATION_PLAN.md` | Implemented — image index is done. |

### Files unchanged

| File | Why |
|------|-----|
| `llm_providers.py` | Clean single responsibility. `get_hardware_info()` stays — only used by one class in the same file. |
| `graph_document_store.py` | Clean single responsibility (graph storage + FAISS) |
| `fr_traversal.py` | Clean single responsibility (FR path matching) |
| `embedding_cache.py` | Clean single responsibility (embedding disk cache) |
| `dataset_manager.py` | Clean single responsibility (multi-dataset management) |
| `scripts/extract_ontology_labels.py` | Clean standalone utility |
| `scripts/evaluate_pipeline.py` | Dev tool, kept as-is |
| `scripts/test_entity_context.py` | Dev tool, kept as-is |

---

## Detailed Migration Plan

### A. Delete cluster workflow

Remove these files entirely:
- `scripts/bulk_generate_documents.py`
- `scripts/cluster_pipeline.py`
- `config/.env.cluster`
- `docs/PROCESSING.md`
- `docs/SPARQL_OPTIMIZATION_PLAN.md`
- `docs/IMAGE_INTEGRATION_PLAN.md`

### B. Remove cluster-only code from `universal_rag_system.py`

Delete these methods (only called via the cluster split workflow):
- `generate_documents_only()` — entry point for `--generate-docs-only`
- `_generate_documents_batch()` — batch doc generation without embedding (duplicated by `process_rdf_data()`)
- `_generate_documents_individual()` — legacy per-entity doc generation (`--no-batch`)
- `create_enhanced_document()` — individual SPARQL doc creation (only caller: `_generate_documents_individual`)
- `_create_document_from_prefetched()` — BFS doc from batch data (only caller: `_generate_documents_batch` BFS branch)
- `embed_from_documents()` — entry point for `--embed-from-docs`
- `_log_generation_complete()` — prints cluster transfer instructions

Keep `process_rdf_data()` — it's the remaining single path for `--rebuild`. It already handles FR+BFS document generation, satellite absorption, image index, wikidata, and embedding in one pass.

Also remove legacy individual SPARQL methods that are only called by the deleted individual generation path:
- `get_entity_literals()` — only caller: `create_enhanced_document()`
- `get_entity_context()` — only caller: `create_enhanced_document()`
- `get_entity_types_cached()` — only caller: `create_enhanced_document()`
- `get_entity_label()` — only caller: `create_enhanced_document()`

Keep `get_all_entities()` — still called by `process_rdf_data()`.

### C. Remove cluster CLI args from `main.py`

Delete these argument definitions and their handling:
- `--generate-docs-only`
- `--embed-from-docs`
- `--no-batch`
- `--process-only`

Remove the corresponding config-setting code (lines 90-97) and the process-only exit check (lines 393-396).

### D. Simplify `initialize()` in `universal_rag_system.py`

Remove the two special-mode early returns at the top:
```python
if self.config.get('generate_docs_only'):
    ...
if self.config.get('embed_from_docs'):
    ...
```

The method becomes: load from cache if available, otherwise `process_rdf_data()` + save.

### E. Create `document_formatter.py`

Extract these methods from `UniversalRagSystem` as module-level functions:

```
is_schema_predicate(predicate) -> bool
is_technical_class_name(class_name) -> bool
process_cidoc_relationship(subject_uri, predicate, object_uri, subject_label, object_label, property_labels, class_labels) -> str
get_relationship_weight(predicate_uri) -> float
```

`UniversalRagSystem` keeps thin wrapper methods that call the module functions with `self._property_labels` etc. as arguments, preserving existing internal call signatures.

### F. Create `sparql_helpers.py`

Extract a `BatchSparqlClient` class containing all batch SPARQL methods:

```
class BatchSparqlClient:
    __init__(sparql: SPARQLWrapper)
    batch_query_tsv(query) -> List[List[str]]
    escape_uri_for_values(uri) -> str
    batch_fetch_types(uris, batch_size) -> Dict[str, set]
    batch_query_outgoing(uris, batch_size) -> Dict
    batch_query_incoming(uris, batch_size) -> Dict
    batch_fetch_literals(uris, batch_size) -> Dict
    batch_fetch_type_labels(type_uris, batch_size) -> Dict
    batch_fetch_wikidata_ids(uris, batch_size) -> Dict[str, str]
    get_entities_context_batch(uris, ...) -> Dict
    build_image_index(dataset_config) -> Dict[str, List[str]]
```

`UniversalRagSystem` holds `self.batch_sparql = BatchSparqlClient(self.sparql)` and delegates calls.

### G. Move `load_interface_config()` to `config_loader.py`

Simple function move. `main.py` imports it from the new location.

---

## Post-Cleanup File Summary

| File | Lines (est.) | Single Responsibility |
|------|----------:|----------------------|
| `universal_rag_system.py` | ~2,500 | RAG orchestrator: initialization, process_rdf_data (build), retrieval, answer generation |
| `sparql_helpers.py` | ~750 | Batch SPARQL query infrastructure |
| `document_formatter.py` | ~150 | CIDOC-CRM predicate filtering, relationship formatting, weights |
| `main.py` | ~380 | Flask web app + CLI |
| `config_loader.py` | ~210 | All configuration loading (.env, datasets.yaml, interface.yaml) |
| `llm_providers.py` | 1,133 | LLM/embedding provider abstraction |
| `graph_document_store.py` | 289 | Graph storage + FAISS |
| `fr_traversal.py` | 646 | FR path matching |
| `embedding_cache.py` | 224 | Embedding disk cache |
| `dataset_manager.py` | 196 | Multi-dataset management |
| `scripts/extract_ontology_labels.py` | 436 | Ontology extraction |
| `scripts/evaluate_pipeline.py` | 122 | Dev evaluation tool |
| `scripts/test_entity_context.py` | 104 | Dev debug tool |

**Net result**: `universal_rag_system.py` drops from 5,084 to ~2,500 lines. ~2,300 lines of cluster workflow code eliminated. Each file has a clear single responsibility.

---

## What is NOT changing

- No algorithm changes (retrieval, subgraph selection, FR traversal, embedding, etc.)
- No configuration format changes (datasets.yaml, .env files, interface.yaml)
- No API endpoint changes
- `process_rdf_data()` (the `--rebuild` path) is untouched — it remains the single way to generate docs + embed
- `llm_providers.py`, `graph_document_store.py`, `fr_traversal.py`, `embedding_cache.py`, `dataset_manager.py` are untouched
- `scripts/evaluate_pipeline.py` and `scripts/test_entity_context.py` stay as dev tools

## Risks

- **Import changes**: All call sites in `universal_rag_system.py` that use extracted methods will need to delegate through thin wrappers or update to use the new modules directly. Risk: missing an import update.
- **Cluster workflow**: Removed entirely. If needed again later, it can be rebuilt from `process_rdf_data()` as the reference implementation.
- **Verification**: After restructuring, `uv run python main.py --env .env.openai --dataset mah` must start without errors. This tests the loading path (no rebuild needed, uses existing cache).
