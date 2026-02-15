# CRM_RAG Architecture Analysis

This document maps every resource the system creates and reads, explains when and why each is used, and identifies opportunities to simplify the pipeline.

---

## 1. High-Level Architecture

The system has two distinct phases: **Build** (offline, runs once per dataset) and **Query** (online, runs per user question). The build phase produces 7 on-disk artifacts. The query phase loads them into memory and runs a multi-stage retrieval pipeline.

```
BUILD PHASE                                    QUERY PHASE

SPARQL ──→ 2-hop batch queries                User question
              │                                     │
              ▼                                     ▼
     FR traversal + doc formatting             LLM query analysis
              │                                     │
              ├──→ entity_documents/*.md        FAISS + BM25 retrieval
              ├──→ edges.parquet                    │
              │         │                      RRF fusion
              │         ├──→ graph edges            │
              │         ├──→ triples index     Type-filtered channel ◄── aggregation_index.json (PageRank)
              │         │                           │
              ▼         │                      Adjacency matrix ◄── triples_index (from edges.parquet)
     document_graph.pkl │                           │
              │         │                      Coherent subgraph (greedy + MMR)
              ▼         ▼                           │
     FAISS index    aggregation_index.json     Context assembly + triples enrichment ◄── triples_index
     BM25 index         (PageRank + FR stats)       │
                                               LLM answer generation
                                                    │
                                               Response + sources
```

---

## 2. Build Phase: Resource Creation

### 2.1 Pipeline Overview

Entry point: `process_rdf_data()` in `rag_system.py` (~270 lines, lines 2018-2290).

The build processes entities in **chunks of 1000** from the SPARQL endpoint. For each chunk it runs 5 phases, then finalizes after all chunks.

```
FOR EACH CHUNK OF 1000 ENTITIES:
  Phase A: Pre-fetch (batch SPARQL)
    ├── batch_fetch_literals(chunk_uris)        → literals dict
    ├── batch_fetch_types(chunk_uris)           → types dict
    ├── batch_fetch_type_labels(type_uris)      → type labels
    └── batch_fetch_wikidata_ids(chunk_uris)    → wikidata mapping

  Phase B: Build FR subgraph (_build_fr_graph_for_chunk)
    ├── batch_query_outgoing(chunk_uris)        → 1-hop outgoing
    ├── batch_query_incoming(chunk_uris)         → 1-hop incoming
    ├── [collect intermediate URIs not in chunk]
    ├── batch_query_outgoing(intermediates)      → 2-hop outgoing
    ├── batch_query_incoming(intermediates)      → 2-hop incoming
    ├── batch_fetch_types(intermediates)         → intermediate types
    └── batch_fetch_literals(time_span_uris)    → date values
    Result: fr_outgoing, fr_incoming, entity_labels, entity_types_map, raw_triples

  Phase C: Generate documents (zero SPARQL, pure in-memory)
    ├── Identify satellites (E41_Appellation, E55_Type, E52_Time-Span)
    ├── For each non-satellite entity:
    │   ├── FR path traversal → fr_results (multi-hop relationships)
    │   ├── Direct predicate collection (1-hop non-FR predicates)
    │   ├── Target enrichments (P2_has_type, K14_has_attribute inline)
    │   ├── Format document text (plain text, [FC] header)
    │   └── Save .md file to entity_documents/
    └── Append raw_triples to edges.parquet (with dedup)

  Phase D: Embed documents
    ├── Check embedding cache (disk-based, per entity URI)
    ├── Generate embeddings for uncached docs (batch API or local model)
    ├── Add to document_store with embedding
    └── Cache new embeddings to disk

  Phase E: Save progress
    └── Pickle document_store.docs → document_graph_temp.pkl

AFTER ALL CHUNKS:
  1. _close_edges_parquet()           → finalize edges.parquet
  2. _build_edges_from_parquet()      → add weighted edges to document_store graph
  3. _chain_thin_documents()          → absorb docs < 400 chars into neighbors
  4. Rename temp → final pkl
  5. build_vector_store_batched()     → create FAISS index from pre-computed embeddings
  6. Build + save BM25 index
  7. _load_triples_index()            → entity → [triple dicts] from edges.parquet
  8. _build_aggregation_index()       → FR stats + PageRank → JSON
  9. Generate validation report
```

### 2.2 FR Traversal (`fr_traversal.py`, 692 lines)

The core document generation logic uses **Fundamental Relationships** (Tzompanaki & Doerr, 2012) to discover semantically meaningful multi-hop paths through the RDF graph.

**What it does**: Given an entity, its types, and 2-hop neighborhood data, it follows curated multi-step paths defined in `fundamental_relationships_cidoc_crm.json` to find related entities.

**Example**: For a `Thing` (e.g., a painting), FR "was produced by" follows the path:
```
Thing →[P108i_was_produced_by]→ Production Event →[P14_carried_out_by]→ Actor
```
This discovers the artist through a 2-hop chain, surfacing it directly in the painting's document.

**Key functions**:
- `match_fr_paths()` — For each FR matching the entity's FC (Fundamental Category), follows each path definition step by step through the pre-fetched outgoing/incoming indexes. Collects all reachable targets.
- `collect_direct_predicates()` — Collects 1-hop predicates that are NOT part of any FR step-0 (e.g., VIR K24_portray, custom extensions).
- `build_target_enrichments()` — For each discovered target, looks up P2_has_type and K14_has_attribute for inline annotations (e.g., "Saint Anastasia (Saint, cross of martyrdom)").
- `classify_satellite()` — Classifies thin entities (E41_Appellation, E55_Type, E52_Time-Span) for absorption into parent documents.
- `format_fr_document()` — Formats: `[FC] Label` header + literals + absorbed satellites + FR results + direct predicates.

**Why 2-hop SPARQL**: FR paths are typically 2-3 steps long (entity → event → actor). The build fetches outgoing/incoming for chunk entities (hop 1) PLUS outgoing/incoming for intermediate URIs reached at hop 1 (hop 2). This gives FR traversal the full subgraph it needs without any additional SPARQL queries during document formatting.

### 2.3 Satellite Absorption

Thin vocabulary entities (E41_Appellation, E55_Type, E52_Time-Span, E54_Dimension, E30_Right) are identified during document generation. Instead of getting their own documents (which would be 1-2 lines), their content is absorbed into the parent entity's document.

**Two stages**:
1. **At generation time** (`_identify_satellites_from_prefetched`): Satellites are detected and their info (names, dates, types) is formatted inline into the parent document via `format_absorbed_satellites()`.
2. **After all chunks** (`_chain_thin_documents`): Any remaining docs under 400 chars are absorbed into their neighbors' text and removed from the store.

**Effect on Asinou**: 606 raw entities → 286 documents (after satellite absorption).

---

## 3. Resource Catalog

### 3.1 On-Disk Artifacts

| # | Resource | Path | Created by | Read by | Purpose |
|---|----------|------|------------|---------|---------|
| 1 | **Entity documents** | `data/documents/{dataset}/entity_documents/*.md` | `process_rdf_data()` Phase C | Not read at query time (debugging/transparency only) | Human-readable entity descriptions |
| 2 | **edges.parquet** | `data/documents/{dataset}/edges.parquet` | `_append_edges_parquet()` during Phase C | `_build_edges_from_parquet()`, `_load_triples_index()` | All RDF triples with labels (6 cols: s, s_label, p, p_label, o, o_label) |
| 3 | **document_graph.pkl** | `data/cache/{dataset}/document_graph.pkl` | `save_document_graph()` after edge building | `initialize()` on startup | Serialized dict of GraphDocument objects (text + embeddings + neighbor lists) |
| 4 | **FAISS index** | `data/cache/{dataset}/vector_index/` | `build_vector_store_batched()` | `initialize()` on startup, `retrieve()` at query time | Dense vector similarity search |
| 5 | **BM25 index** | `data/cache/{dataset}/bm25_index/` | `build_bm25_index()` + `save_bm25_index()` | `initialize()` on startup, `retrieve_bm25()` at query time | Sparse keyword search |
| 6 | **Aggregation index** | `data/cache/{dataset}/aggregation_index.json` | `_build_aggregation_index()` | `_type_filtered_channel()`, `_build_aggregation_context()` | Pre-computed FR stats, entity counts, PageRank scores |
| 7 | **Embedding cache** | `data/cache/{dataset}/embeddings/` | `_process_batch_embeddings()` Phase D | Phase D (cache hit check) | Per-entity embeddings for resume/rebuild without re-embedding |

### 3.2 In-Memory Structures at Query Time

| Structure | Populated by | Used by | Content |
|-----------|-------------|---------|---------|
| `document_store.docs` | Loaded from document_graph.pkl | Everything | Dict[URI → GraphDocument] with text, embedding, metadata, neighbors |
| `document_store.vector_store` | Loaded from vector_index/ | `retrieve()`, `_type_filtered_channel()` | FAISS index for similarity search |
| `document_store.bm25_retriever` | Loaded from bm25_index/ | `retrieve()`, `_type_filtered_channel()` | BM25 sparse index |
| `document_store._fc_doc_ids` | `build_fc_type_index()` at startup | `_type_filtered_channel()` | FC category → set of doc URIs |
| `_triples_index` | `_load_triples_index()` from edges.parquet | `_build_triples_enrichment()`, `_build_sources()`, `create_adjacency_matrix()` | Dict[entity_URI → list of triple dicts] |
| `_actor_work_counts` | `_load_triples_index()` (same pass) | `_build_triples_enrichment()` | Dict[actor_URI → count of works] |
| `_aggregation_index` | Loaded from aggregation_index.json | `_type_filtered_channel()`, `_build_aggregation_context()` | FR stats, PageRank, entity counts |

### 3.3 Configuration Files

| File | Content | Used by |
|------|---------|---------|
| `config/fundamental_relationships_cidoc_crm.json` | 76 FRs, 237 paths, 92 unique properties | `FRTraversal` at build time |
| `config/fc_class_mapping.json` | 168 CRM classes → 6 FCs (Thing, Actor, Place, Event, Concept, Time) | Type-filtered retrieval, FC boosting, FR traversal |
| `config/event_classes.json` | Event class URIs | Thin-doc chaining exemption |
| `data/labels/property_labels.json` | Property URI → English label | Document formatting, triple enrichment |
| `data/labels/class_labels.json` | Class URI → English label | FC mapping expansion, document formatting |
| `data/labels/crm_taxonomy.json` | subPropertyOf + subClassOf hierarchies | FR path matching (sub-property resolution) |
| `data/labels/inverse_properties.json` | owl:inverseOf bidirectional mapping | FR path matching (inverse traversal) |

---

## 4. Query Phase: Retrieval Pipeline

### 4.1 Overview

Entry point: `answer_question()` (line 3721), a thin orchestrator calling three phases:

```
answer_question(question, chat_history)
  │
  ├── Phase 1: _analyze_query(question)        → QueryAnalysis (LLM call)
  │     Returns: query_type (SPECIFIC/ENUMERATION/AGGREGATION)
  │              categories (target FCs: ["Thing", "Actor", ...])
  │              context_categories (filter FCs)
  │
  ├── Phase 2: _prepare_retrieval(question, chat_history, k, pool_size, query_analysis)
  │     ├── Pivot detection (regex: "aside from X", "other than X")
  │     ├── If no history: retrieve(question)
  │     ├── If pivot: retrieve(clean_query) — raw only
  │     └── If follow-up: dual retrieval
  │           ├── retrieve(contextualized_query) — prev turn + current
  │           ├── Vague follow-up detection (stopword-dominated, ≤10 words)
  │           ├── retrieve(raw_question) — if not vague
  │           └── Interleaved merge + dedup
  │     Returns: List[GraphDocument] (up to k docs)
  │
  └── Phase 3: _generate_answer(question, retrieved_docs, query_analysis, ...)
        ├── Context assembly (doc text, truncated to 5000 chars each)
        ├── _build_triples_enrichment(retrieved_docs) — from _triples_index
        ├── _build_aggregation_context() — for ENUM/AGG queries
        ├── Wikidata context (top 2 entities)
        ├── System prompt tuning (ENUM → "List ALL", AGG → "Count or rank")
        ├── LLM call → answer text
        └── _build_sources(retrieved_docs) — URIs, labels, images, triples
        Returns: {answer, sources}
```

### 4.2 The `retrieve()` Method (line 3080)

The multi-stage retrieval pipeline:

```
retrieve(query, k=10, initial_pool_size=60, alpha=0.7, query_analysis)
  │
  ├── Stage 1: FAISS retrieval (pool_size candidates)
  │     document_store.retrieve(query, k=pool_size)
  │     Returns: [(GraphDocument, similarity_score), ...]
  │     Similarity = 1/(1 + L2_distance)
  │
  ├── Stage 2: BM25 retrieval (pool_size candidates)
  │     document_store.retrieve_bm25(query, k=pool_size)
  │     Returns: [(GraphDocument, bm25_score), ...]
  │
  ├── Stage 3: RRF fusion
  │     _rrf_fuse(faiss_results, bm25_results, k_rrf=60, pool_size)
  │     RRF_score(doc) = Σ 1/(60 + rank + 1) across both lists
  │     Returns: [(GraphDocument, rrf_score), ...] up to pool_size
  │
  ├── Stage 4: Type-filtered channel (_type_filtered_channel)
  │     ├── Filter FAISS to target FC categories only
  │     ├── Filter BM25 to target FC categories only
  │     ├── RRF-fuse the type-filtered results
  │     ├── For ENUM/AGG: also fuse PageRank candidates from aggregation_index
  │     ├── Reserve 50% of pool slots for type-matching docs (30% for SPECIFIC)
  │     └── Merge back into main pool
  │
  ├── Stage 5: Non-informative type capping
  │     Cap E41_Appellation, E34_Inscription, etc. to ≤25% of pool
  │
  ├── Stage 6: Adjacency matrix construction
  │     document_store.create_adjacency_matrix(candidate_ids, triples_index, weight_fn, max_hops=2)
  │     ├── 1-hop: direct edges between candidates (weighted by CIDOC-CRM semantics)
  │     ├── 2-hop: virtual edges through intermediates connecting 2+ candidates
  │     │   Virtual weight = (weight_a × weight_b) / max_hops
  │     ├── Self-loops
  │     └── Symmetric normalization: D^(-1/2) × A × D^(-1/2)
  │
  └── Stage 7: Coherent subgraph extraction
        compute_coherent_subgraph(candidates, adj_matrix, scores, k, alpha=0.7)
        Greedy selection, each round picks doc maximizing:
          score = α × relevance + (1-α) × connectivity
                  - 0.2 × max_cosine_sim_to_selected (MMR diversity)
                  + type_modifier (per-type boost/penalty)
                  - mega_entity_penalty (if >2000 triples: -0.15)
        Plus FC category boost: +0.10 for docs matching query target categories
        Returns: [GraphDocument, ...] in selection order (k docs)
```

### 4.3 Triples Enrichment at Query Time

After retrieval selects k documents, `_build_triples_enrichment()` (line 3297) enriches the LLM context with structured RDF relationships from `_triples_index`:

1. For each retrieved doc, looks up all triples from the Parquet-derived index
2. Filters out schema predicates (rdf:type, rdfs:label, owl:inverseOf, etc.)
3. Prioritizes triples by a 4-level scheme:
   - Priority 0: Inter-document triples (both endpoints are retrieved docs)
   - Priority 1: High-value predicates (temporal, creator, location, depiction)
   - Priority 2: Predicates with labels
   - Priority 3: Everything else
4. Resolves E52_Time-Span URIs to actual dates (P82a, P82b)
5. Allocates per-entity character budget, capped at 5000 chars total
6. Appended to LLM prompt as "Structured Relationships" section

### 4.4 PageRank and Aggregation Context

**PageRank** (`_compute_pagerank`, line 1257): Computed at build time on the FR discovery graph (not the document graph). Each FR match creates an edge entity → target. NetworkX PageRank with alpha=0.85 produces centrality scores. Top 500 entities per FC stored in aggregation_index.json.

**Usage at query time**:
- `_type_filtered_channel()`: For ENUMERATION/AGGREGATION queries, `_get_pagerank_candidates()` retrieves the top PageRank entities in the query's target FCs and fuses them with type-filtered FAISS+BM25 via RRF. This surfaces "important" entities that might not match the query embedding directly.
- `_build_aggregation_context()`: For ENUM/AGG queries, includes entity type counts, FC counts, top entities by PageRank, and FR relationship summaries in the LLM prompt.

---

## 5. Data Duplication and Redundancy Analysis

Several pieces of information are stored or computed in multiple places:

| Information | Stored in | Notes |
|-------------|-----------|-------|
| Entity text | document_graph.pkl (GraphDocument.text) AND entity_documents/*.md | The .md files are never read at query time — purely for debugging |
| Entity embeddings | document_graph.pkl (GraphDocument.embedding) AND FAISS index AND embedding cache | Three copies. The pkl holds them for rebuilding FAISS without re-embedding |
| Entity edges/relationships | document_graph.pkl (GraphDocument.neighbors) AND edges.parquet AND _triples_index (in memory) | neighbors = typed edge list; parquet = raw triples; _triples_index = same triples as in-memory dict |
| Entity types | document_graph.pkl (metadata["type"], metadata["all_types"]) AND _fc_doc_ids (inverted index) | FC index is derived from metadata at startup |
| FR statistics | aggregation_index.json (fr_summaries) AND PageRank scores (in same JSON) | Both derived from FR traversal during build |

### What edges.parquet actually provides

edges.parquet serves **three** consumers at different times:

1. **`_build_edges_from_parquet()`** — Build time, once. Reads (s, p, o) to add weighted edges to the document_store graph. Only entity-to-entity triples where both endpoints are documents.
2. **`_load_triples_index()`** — Startup time, once. Reads all 6 columns to build the in-memory entity → [triples] dict. Filters to triples where at least one endpoint is a document.
3. **`create_adjacency_matrix()`** — Query time, per query. Reads from `_triples_index` (not from parquet directly) to build the candidate adjacency matrix.

After startup, the parquet file itself is never touched again — all query-time access goes through the in-memory `_triples_index`.

---

## 6. Information Workflow: Life of a Triple

This section traces a single RDF triple from its origin in the SPARQL endpoint through every transformation, storage layer, and fork point until it reaches its final consumers at query time.

### 6.1 Origin — RDF triples in the SPARQL endpoint

All data starts as RDF triples in an Apache Jena Fuseki triplestore. A triple is three URIs: subject, predicate, object.

```
<http://example.org/church_X>  <http://cidoc-crm.org/P55_has_current_location>  <http://example.org/place_Y>
```

This triple states that church_X has its current location at place_Y. The predicate `P55_has_current_location` is a CIDOC-CRM property with defined semantics — it links a physical thing to the place where it resides.

### 6.2 Fetch — Batch SPARQL queries

`sparql_helpers.py` (`BatchSparqlClient`) executes batch SPARQL queries to extract triples from the endpoint.

**Two query directions per entity set**:
- `batch_query_outgoing(uris)` → `Dict[uri → [(predicate, object_uri, object_label)]]`
- `batch_query_incoming(uris)` → `Dict[uri → [(subject_uri, predicate, subject_label)]]`

Each query uses a `VALUES` clause to fetch edges for up to 1000 URIs at once. Results come back as TSV (tab-separated values) for 3× faster parsing than JSON. The TSV parser (`batch_query_tsv`, line 31) strips URI angle brackets (`<http://...>` → `http://...`), removes language tags (`"value"@en` → `value`), and drops datatype suffixes (`"value"^^<xsd:string>` → `value`).

**Output**: 3-tuples per entity — the same triple is now a Python tuple:
```python
("http://cidoc-crm.org/P55_has_current_location", "http://example.org/place_Y", "Village of Nikitari")
```

### 6.3 Conversion — Building the FR subgraph

`rag_system.py`: `_build_fr_graph_for_chunk()` (line 1409) processes the raw 3-tuples into three parallel data structures:

**a) FR indexes (2-tuple format)**:
```python
fr_outgoing[church_X] = [("P55_has_current_location", "place_Y"), ...]   # (pred, target)
fr_incoming[place_Y]  = [("P55_has_current_location", "church_X"), ...]  # (pred, source)
```
The object label is dropped — FR traversal only needs predicates and URIs to walk paths.

**b) Entity labels dict**:
```python
entity_labels["http://example.org/place_Y"] = "Village of Nikitari"
```
Labels come from `rdfs:label` in SPARQL results, or from `chunk_literals` (pre-fetched label/name properties), falling back to the URI's last path segment.

**c) Raw triples list** (6-field dicts for Parquet):
```python
{
    "subject": "http://example.org/church_X",
    "subject_label": "Panagia Phorbiottisa",
    "predicate": "http://cidoc-crm.org/P55_has_current_location",
    "predicate_label": "has current location",
    "object": "http://example.org/place_Y",
    "object_label": "Village of Nikitari"
}
```

**2-hop fetching**: Intermediate URIs reached at hop 1 (events, places, types not in the current chunk) are collected. A second round of `batch_query_outgoing` + `batch_query_incoming` fetches their edges, extending `fr_outgoing`/`fr_incoming` to cover 2-hop FR paths. All intermediate triples are also added to `raw_triples`.

### 6.4 FR path matching

`fr_traversal.py`: `match_fr_paths()` (line 232) walks pre-defined multi-step paths through the 2-tuple indexes.

**Input**: entity URI, its rdf:type URIs, the full `fr_outgoing`/`fr_incoming` indexes, labels, and type maps.

**Process**: For each FR whose domain FC matches the entity's Fundamental Category, each path's steps are walked sequentially. At each step, `_follow_property()` finds all nodes reachable via the step's property (or its inverse). Recursive steps use BFS until no new nodes appear.

**Example**: For a Thing (painting), FR "was produced by" follows:
```
church_X →[P108i_was_produced_by]→ production_event →[P14_carried_out_by]→ artist_Z
```

**Output**: List of FR result dicts:
```python
[{"fr_id": "FR3", "fr_label": "was produced by", "targets": [("artist_Z", "John the Painter")], "total_count": 1}]
```

### 6.5 Document generation

`rag_system.py`: `_create_fr_document_from_prefetched()` (line 1630) assembles the final document text.

**Combines four sources**:
1. **FR results** — multi-hop relationships discovered by path matching
2. **Direct predicates** — 1-hop non-FR predicates (VIR K24_portray, extensions) via `collect_direct_predicates()`
3. **Target enrichments** — P2_has_type and K14_has_attribute annotations for discovered targets, via `build_target_enrichments()`
4. **Absorbed satellites** — content from E41_Appellation, E55_Type, E52_Time-Span entities absorbed into the parent

**Output**: Natural language markdown text formatted by `format_fr_document()`:
```
[Thing] Panagia Phorbiottisa
Types: Church, E18_Physical_Thing

was produced by: Production of Panagia Phorbiottisa
  carried out by: Nikephoros Magistros (Donor, historical figure)

has current location: Village of Nikitari (Village, Cyprus)
falls within: Nicosia District → Cyprus
```

This text is saved to `entity_documents/*.md` (for debugging) and stored in `GraphDocument.text` (for retrieval).

### 6.6 Parquet write — structured triple archive

`rag_system.py`: `_append_edges_parquet()` (line 954) writes the `raw_triples` list to disk incrementally.

**Format**: 6 columns — `s`, `s_label`, `p`, `p_label`, `o`, `o_label`

**Deduplication**: Each triple is hashed by `hash((subject, predicate, object))`. The hash set (`_seen_triple_hashes`) persists across chunks, so intermediate entities fetched by multiple chunks are not duplicated.

**Writer**: A streaming `pyarrow.ParquetWriter` stays open across all chunks, writing batch tables without loading the full file into memory. `_close_edges_parquet()` (line 997) finalizes the file after all chunks are processed.

**Scale**: For the MAH dataset (866K entities), the deduped file contains ~14.6M unique triples.

### 6.7 First Parquet read — graph edges (build time only)

`rag_system.py`: `_build_edges_from_parquet()` (line 1364) reads the Parquet file in 500K-row batches, using only 3 columns (`s`, `p`, `o`).

**What it does**: For each triple where both subject and object are document entities (exist in `document_store.docs`), adds a weighted bidirectional edge via `document_store.add_edge()`.

**Weights**: CIDOC-CRM semantic weights from `document_formatter.py` (e.g., P89_falls_within: 0.9, P55_has_current_location: 0.9, P46_is_composed_of: 0.8, default: 0.5).

**Single consumer**: These edges are used exclusively by `_chain_thin_documents()` — the post-build step that absorbs documents under 400 characters into their richest neighbor. No query-time code reads `GraphDocument.neighbors`.

### 6.8 Second Parquet read — triples index

`rag_system.py`: `_load_triples_index()` (line 1010) reads all 6 columns in 500K-row batches.

**What it builds**: `_triples_index` — a dict mapping each entity URI to its list of triple dicts (with all 6 fields). Only indexes triples where at least one endpoint is a document entity.

**Same-pass side index**: Also builds `_actor_work_counts` by tracking P14_carried_out_by and P108i_was_produced_by chains (event → actor, work → event → actor).

**Deduplication**: Hashes `(s, p, o)` to handle legacy pre-dedup files.

**Lifetime**: Loaded once at startup (or after build), persists in memory for all query-time operations. The Parquet file is never touched again after this read.

### 6.9 Embedding — from text to vectors

Document text is converted to vector embeddings (BAAI/bge-m3 1024-dim or OpenAI text-embedding-3-small). Embeddings are stored in three places:

1. `GraphDocument.embedding` — in the pickle file, enables FAISS rebuild without re-embedding
2. **FAISS index** (`vector_index/`) — dense vector similarity search at query time
3. **BM25 index** (`bm25_index/`) — sparse keyword search, tokenized from document text

### 6.10 Query-time consumers (3 paths from triples index)

After build, the triple's data is consumed through three query-time paths:

**a) Adjacency matrix** (`document_store.create_adjacency_matrix()`, line 191):
- Reads `_triples_index` for each candidate document
- Builds N×N matrix: 1-hop direct edges (both endpoints are candidates) + virtual 2-hop edges (through intermediates connecting 2+ candidates)
- Virtual weight = `(weight_a × weight_b) / max_hops`
- Symmetric normalization: D^(-1/2) × A × D^(-1/2)
- Used by coherent subgraph selection for the connectivity signal

**b) Triples enrichment** (`_build_triples_enrichment()`):
- For each retrieved document, looks up all triples from `_triples_index`
- Filters out schema predicates, prioritizes by 4-level scheme (inter-doc=0, high-value=1, labeled=2, other=3)
- Resolves E52_Time-Span URIs to dates via `_resolve_time_span()`
- Appended to the LLM prompt as "Structured Relationships" — provides factual grounding beyond document text
- Capped at 5000 chars total

**c) Source entries** (`_build_sources()`):
- Attaches raw triples to the API response for each source document
- Enables the chat UI to display structured relationship data alongside answers

### 6.11 Flow diagram

```
SPARQL Endpoint (RDF triples)
         │
         ▼
  batch_query_outgoing / incoming          ← sparql_helpers.py
  (TSV parsing: strip <>, @lang, ^^type)
         │
         │ 3-tuples: (pred, target, label)
         ▼
  _build_fr_graph_for_chunk                ← rag_system.py
         │
    ┌────┼────────────────┐
    │    │                │
    ▼    ▼                ▼
  fr_outgoing/      entity_labels     raw_triples
  fr_incoming                         (6-field dicts)
  (2-tuples)                               │
    │                                      │
    ▼                                      ▼
  match_fr_paths     ──────────→    _append_edges_parquet    ← FORK POINT
  (fr_traversal.py)                        │
    │                                      │ edges.parquet (6 cols)
    ▼                                      │
  _create_fr_document   ◄─── direct_preds  ├──────────────────────────────┐
  (doc text + .md file)       target_enrichments                          │
    │                                      │                              │
    ▼                                      ▼                              ▼
  GraphDocument.text              _build_edges_from_parquet    _load_triples_index
    │                             (s, p, o → weighted edges)   (all 6 cols → dict)
    ▼                                      │                              │
  Embedding                                ▼                              │
    │                             GraphDocument.neighbors                 │
    ├──→ FAISS index              (consumed only by                       │
    ├──→ BM25 index                _chain_thin_documents)                 │
    └──→ document_graph.pkl                                               │
                                                           ┌──────────────┤
                                          QUERY TIME       │              │
                                                           ▼              │
                                                   Adjacency matrix       │
                                                   (1-hop + virtual       │
                                                    2-hop edges)          │
                                                           │              ▼
                                                           │    Triples enrichment
                                                           │    (→ LLM prompt)
                                                           │              │
                                                           ▼              ▼
                                                   Coherent subgraph   Source entries
                                                   selection           (→ API response)
```

---

## 7. Complexity Assessment

### What each retrieval stage contributes

| Stage | Purpose | Complexity cost | Empirical value |
|-------|---------|----------------|-----------------|
| FAISS | Dense semantic similarity | Low (fast, core) | Essential — baseline retrieval |
| BM25 | Sparse keyword matching | Low (fast, complementary) | Catches exact-name matches FAISS misses |
| RRF fusion | Merge FAISS + BM25 | Trivial | Standard practice, well-justified |
| Type-filtered channel | Boost docs of the right ontological type | Medium (parallel retrieval + FC index) | Helps ENUM/AGG queries ("list all actors") |
| PageRank fusion | Surface central entities | Medium (aggregation index + NetworkX at build time) | Marginal benefit — only for ENUM/AGG, effect unclear |
| Non-informative capping | Prevent metadata docs flooding results | Low (simple filter) | Useful quality guard |
| Adjacency matrix | Capture graph connectivity between candidates | High (N^2 matrix, 2-hop virtual edges, normalization) | Moderate — helps when related docs aren't embedding-similar |
| Coherent subgraph | Balance relevance + connectivity + diversity | Medium (greedy O(k*N)) | Valuable for multi-entity context coherence |
| LLM query analysis | Classify query type + target FCs | 1 extra LLM call per query | Enables dynamic k and type filtering |
| Triples enrichment | Add structured relationships to LLM context | Low (dict lookup + formatting) | Valuable — provides factual grounding beyond doc text |
| Aggregation context | Dataset-level statistics for counting queries | Low (JSON lookup + formatting) | Useful for "how many X?" questions |

### The graph-related cost chain

The most expensive chain is: **edges.parquet → graph edges → adjacency matrix → coherent subgraph**. Let's trace what each step buys:

1. **edges.parquet** (14.7 GB for MAH before dedup): Stores all 2-hop RDF triples. Needed because FR traversal fetches intermediates that may connect document entities.
2. **Graph edges** (in document_store.neighbors): Built from parquet, only entity-to-entity. Used by adjacency matrix at query time. Also used by `_chain_thin_documents()` to find neighbors for absorption.
3. **Adjacency matrix** (per query, N candidates): Reads `_triples_index` to find direct and 2-hop connections between candidates. The 2-hop virtual edges here are different from FR 2-hop — they find intermediate nodes that connect two candidate documents.
4. **Coherent subgraph**: Uses the adjacency matrix to prefer selecting connected documents over isolated ones (α=0.3 connectivity weight).

---

## 8. Proposed Simplifications (Re-evaluated)

The following suggestions have been critically re-evaluated for a **universal, dataset-agnostic CIDOC-CRM RAG system** that must work for both small datasets (Asinou: 286 docs) and large datasets (MAH: 866K docs). Each suggestion includes pros and cons. Suggestions that don't hold up under scrutiny are explicitly marked as withdrawn with an explanation.

---

### 8.1 Eliminate graph edges from document_store — RECOMMENDED

**What**: `_build_edges_from_parquet()` reads the entire parquet file to add typed, weighted edges to `GraphDocument.neighbors`. Code analysis shows these edges are consumed by exactly **one** caller: `_chain_thin_documents()` at build time (line 862: `if ... doc.neighbors`). At query time, the adjacency matrix is built entirely from `_triples_index` (via `create_adjacency_matrix(doc_ids, triples_index, ...)`), never from `neighbors`.

**Proposal**: Replace `doc.neighbors` lookups in `_chain_thin_documents()` with `_triples_index` lookups. Then remove `_build_edges_from_parquet()`, `add_edge()`, and the neighbor list from `GraphDocument`.

**Pros**:
- Eliminates one full parquet read at build time (currently the second of two reads)
- Smaller `document_graph.pkl` (no neighbor lists serialized)
- Removes ~100 lines of dead-at-query-time code (`add_edge`, neighbor management, CIDOC-CRM weight calculation during edge building)
- Zero information loss — the adjacency matrix already builds its own connectivity from `_triples_index`

**Cons**:
- `_chain_thin_documents()` must be rewritten to discover neighbors from `_triples_index` instead of pre-built neighbor lists. This is straightforward (iterate `triples_index[doc_id]`, collect endpoints that are in `document_store.docs`) but requires careful testing of thin-doc absorption counts.
- Requires `_triples_index` to be loaded before thin-doc chaining runs. Current build order must be verified/adjusted.

**Verdict**: Clear win. The graph edges are a vestige of an earlier design where `neighbors` drove query-time connectivity. Now that the adjacency matrix uses `_triples_index`, the edge machinery is genuinely dead weight.

---

### 8.2 Serialize _triples_index in document_graph.pkl — RECOMMENDED

**What**: On every startup, `_load_triples_index()` re-reads the entire edges.parquet file to rebuild `_triples_index` (entity_uri → [triple dicts]). For MAH (~14.6M unique triples), this takes significant time. The same data is always read identically.

**Proposal**: After building `_triples_index` during the build phase, serialize it alongside the document store in `document_graph.pkl`. On startup, load from pickle (fast deserialization) instead of re-parsing parquet.

**Pros**:
- Eliminates parquet read on every startup — particularly impactful for large datasets
- Startup becomes a single pickle load instead of pickle + parquet parse
- Parquet is still written at build time (streaming writes are essential), but only consumed once during the same build session

**Cons**:
- Larger pickle file. For MAH with ~14.6M triples filtered to doc entities, the triples_index dict could add ~500 MB to the pickle. For small datasets (Asinou: ~920 triples) the overhead is negligible.
- Debugging becomes harder — can't inspect triples via parquet tools. Mitigated by keeping the parquet file on disk as a build artifact (just not loading it at startup).
- If combined with 8.1 (removing graph edges), the parquet file becomes a build-only intermediate. Could be deleted after build, but keeping it costs nothing on disk and aids debugging.

**Verdict**: Good trade-off for production. The startup speed improvement matters for large datasets, and the file size increase is acceptable since the triples_index is already filtered to doc entities only.

---

### ~~8.3 Remove PageRank fusion~~ — WITHDRAWN

**Why withdrawn**: PageRank provides a **structurally distinct signal** that neither FAISS nor BM25 can replicate. For ENUMERATION queries ("list all painters in the collection"), FAISS finds entities whose document text is embedding-similar to the query, and BM25 finds keyword matches. But neither surfaces entities that are **graph-central** — those with the most FR connections to other entities in the dataset.

Concretely:
- PageRank is computed **once at build time** (~80 lines, NetworkX). The query-time cost is a dict lookup + one RRF fusion — negligible.
- For ENUM/AGG queries, PageRank ensures that the most interconnected entities (e.g., a prolific artist connected to many works via multiple FRs) appear in the candidate pool, even if their document embedding isn't close to the query.
- Without PageRank, "list all artists who worked in Geneva" could miss important artists whose documents focus on biography rather than location — PageRank would still surface them by graph centrality.

**Conclusion**: Keep PageRank. The build-time cost is trivial, the query-time cost is a dict lookup, and it provides a genuinely orthogonal signal for ENUM/AGG queries.

---

### ~~8.4 Skip adjacency matrix for SPECIFIC queries~~ — WITHDRAWN

**Why withdrawn**: Two reasons:

1. **Computational cost is negligible regardless of dataset size.** The adjacency matrix operates on the candidate **pool** (N = k × POOL_MULTIPLIER), not the full dataset. For SPECIFIC queries: N = 10 × 6 = 60 → a 60×60 matrix. For AGGREGATION: N = 25 × 6 = 150 → a 150×150 matrix. Both are trivial to compute even on modest hardware. The pool size is bounded by `RetrievalConfig`, not by dataset size — MAH (866K docs) and Asinou (286 docs) produce identically sized matrices.

2. **CIDOC-CRM data is inherently relational.** Even SPECIFIC queries benefit from the connectivity signal. "Tell me about Panagia Phorbiottisa" should surface not just the church entity, but its location (P55_has_current_location), its frescoes (P46_is_composed_of), its donors (P11_had_participant via events). The adjacency matrix nudges the coherent subgraph selector toward these connected entities (at α=0.3, a mild but useful preference). Without it, the top-10 are pure embedding similarity, which can miss structurally important neighbors whose text isn't embedding-close.

**Conclusion**: Keep the adjacency matrix for all query types. The cost is O(N²) where N ≤ 150 — always fast — and the connectivity signal genuinely helps CIDOC-CRM retrieval where entity relationships carry core information.

---

### ~~8.5 Pre-compute doc-to-doc adjacency at build time~~ — WITHDRAWN

**Why withdrawn**: This is a premature optimization that trades massive memory for minimal speed gains.

1. **Memory explosion for large datasets.** A single E12_Production event can connect to hundreds of E22_Man-Made_Object entities. Pre-computing all 2-hop pairs: if one event connects 500 works, that's 500×499/2 = 124,750 pairs from one intermediate. Across all events, types, and places in MAH (866K docs), the sparse dict could easily exceed available memory.

2. **The problem it solves doesn't exist.** The current per-query adjacency matrix operates on the small candidate pool (60-150 docs), not the full graph. Computing a 150×150 matrix from `_triples_index` lookups takes milliseconds. There's no performance bottleneck to fix.

3. **Different purposes shouldn't be unified.** Build-time 2-hop (FR traversal) serves document content generation — it fetches intermediate entities so FR paths can discover relationships like Thing→Event→Actor. Query-time 2-hop (adjacency matrix) serves retrieval connectivity — it finds shared intermediaries between candidate documents. These are conceptually different operations on different subsets of data.

**Conclusion**: Keep the current per-query adjacency matrix computation. It's fast (bounded by pool size, not dataset size), memory-efficient, and correctly scoped to the candidate pool.

---

### 8.6 Make entity_documents/*.md files optional — RECOMMENDED

**What**: Individual .md files are written during build but never read at query time. The same text is stored in `document_graph.pkl` (via `GraphDocument.text`). For MAH (866K entities), writing these files adds significant I/O.

**Original proposal was to remove them entirely. Revised proposal**: Make them **opt-out** via a `--no-md-files` flag or config option. Default behavior: write them (useful for development and debugging). Production/cluster runs: skip them.

**Pros**:
- Saves significant I/O for large datasets (866K file writes on MAH)
- No impact on retrieval quality (files are never read at query time)
- The .md files remain available for development, where browsing individual entity documents is valuable for debugging document generation quality

**Cons**:
- Minor code change to add the flag/config option
- Loss of .md files in production means debugging requires loading the pickle or re-running the build with the flag enabled. But production debugging is rare, and the pickle can be queried via the `--question` CLI mode.

**Verdict**: Simple, risk-free improvement. Keep the default behavior (write files) for backward compatibility and debugging, but allow skipping for production builds on large datasets.

---

### 8.7 Split rag_system.py into focused modules — RECOMMENDED (low priority)

**What**: `rag_system.py` is ~3650 lines combining build-time logic (SPARQL queries, FR orchestration, edge building, embedding, aggregation index, validation) and query-time logic (retrieval pipeline, context assembly, answer generation, triples enrichment).

**Proposal**: Split along the build/query boundary into 3-4 modules (builder, retriever, answerer, thin orchestrator).

**Pros**:
- Clear separation of build-time vs query-time code — currently it's hard to tell which methods run when
- Each module becomes independently testable
- Easier to modify the retrieval pipeline without risk of breaking build logic (and vice versa)
- New contributors can understand the system module by module

**Cons**:
- Significant refactor effort (~3650 lines to reorganize)
- Risk of introducing bugs during the split — cross-references between build and query code (e.g., `_triples_index` is built in the build phase but consumed in the query phase) require careful interface design
- The split only improves maintainability, not retrieval quality. It's a developer experience improvement, not a pipeline improvement.
- Until the other changes above stabilize (especially 8.1 removing graph edges), splitting may need to be redone

**Verdict**: Worth doing eventually, but should wait until changes 8.1, 8.2, and 8.6 are implemented and stable. Splitting a moving target creates unnecessary rework.

---

### 8.8 Summary

| # | Change | Effort | Risk | Status |
|---|--------|--------|------|--------|
| 1 | Eliminate graph edges, use _triples_index for thin-doc chaining (8.1) | Medium | Low | **Recommended** — removes dead machinery |
| 2 | Serialize _triples_index in pickle (8.2) | Medium | Low | **Recommended** — faster startup |
| 3 | Make entity_documents/*.md optional (8.6) | Small | Very low | **Recommended** — saves I/O on large datasets |
| 4 | Split rag_system.py into modules (8.7) | Large | Medium | **Recommended** (after 1-3 are stable) |
| 5 | ~~Remove PageRank fusion (8.3)~~ | — | — | **Withdrawn** — provides unique graph-centrality signal for ENUM/AGG |
| 6 | ~~Skip adjacency for SPECIFIC (8.4)~~ | — | — | **Withdrawn** — cost is negligible, benefit is real |
| 7 | ~~Pre-compute doc-to-doc adjacency (8.5)~~ | — | — | **Withdrawn** — memory-prohibitive for large datasets, solves non-problem |
