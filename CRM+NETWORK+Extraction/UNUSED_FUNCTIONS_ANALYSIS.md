# Unused Functions Analysis

**Generated**: 2025-11-05
**Purpose**: Identify obsolete and unused functions in universal_rag_system.py that are artifacts from previous coding attempts

---

## Summary

**Total Functions Analyzed**: 39
**Actively Used**: 25 (including `normalize_scores()`)
**UNUSED/OBSOLETE**: 10 (in universal_rag_system.py + graph_document_store.py)
**Potentially Unused**: 4 (nested functions that could be refactored)

**Estimated Lines of Code to Remove**: ~425 lines from main files

⚠️ **CORRECTION**: Initial analysis incorrectly marked `normalize_scores()` as unused. It IS actively used by `compute_coherent_subgraph()`.

---

## 🔴 UNUSED FUNCTIONS (Safe to Delete)

These functions are defined but **NEVER called** anywhere in the codebase. They are artifacts from previous implementations.

### 1. **`generate_sparql_query(question)`** (Line 1056)
- **Purpose**: Generate SPARQL queries from natural language questions
- **Why Unused**: Was part of a hybrid approach that tried direct SPARQL querying before RAG
- **Called By**: Only by `answer_with_direct_query()` which itself is unused
- **Impact**: ~40 lines

### 2. **`answer_with_direct_query(question)`** (Line 1097)
- **Purpose**: Answer questions directly with SPARQL queries
- **Why Unused**: The hybrid approach was abandoned in favor of pure RAG retrieval
- **Called By**: Only by `hybrid_answer_question()` which itself is unused
- **Impact**: ~20 lines

### 3. **`hybrid_answer_question(question)`** (Line 1236)
- **Purpose**: Hybrid approach combining direct SPARQL + RAG fallback
- **Why Unused**: System now uses pure RAG approach with `answer_question()` instead
- **Called By**: NEVER called from anywhere
- **Impact**: ~50 lines
- **Note**: This was likely replaced by the current `answer_question()` method

### 4. **`relationship_aware_retrieval(query, k=20)`** (Line 1636)
- **Purpose**: Enhanced retrieval using vector similarity + PersonalizedPageRank
- **Why Unused**: Replaced by `cidoc_aware_retrieval()` which is simpler and actually used
- **Called By**: NEVER called
- **Impact**: ~80 lines
- **Dependencies**: Uses `extract_entities_from_query()`, `calculate_relationship_scores()`, `normalize_scores()`

### 5. **`extract_entities_from_query(query)`** (Line 1495)
- **Purpose**: Extract entity URIs from natural language queries
- **Why Unused**: Was only used by `relationship_aware_retrieval()` which is unused
- **Called By**: Only by unused `relationship_aware_retrieval()`
- **Impact**: ~35 lines

### 6. **`calculate_relationship_scores(entity_uris, damping, iterations)`** (Line 1573)
- **Purpose**: Calculate PersonalizedPageRank scores for graph ranking
- **Why Unused**: Was only used by `relationship_aware_retrieval()` which is unused
- **Called By**: Only by unused `relationship_aware_retrieval()`
- **Impact**: ~60 lines
- **Note**: Uses complex networkx PageRank calculations that are never executed

### 7. ~~**`normalize_scores(scores)`** (Line 1535)~~ ⚠️ **KEEP THIS - ACTIVELY USED**
- **Purpose**: Min-max normalization of score dictionaries
- **Status**: ❌ **INCORRECT ANALYSIS** - This function IS actively used!
- **Called By**: `compute_coherent_subgraph()` at lines 2089 and 2152
- **Used In Active Path**: `answer_question()` → `retrieve()` → `compute_coherent_subgraph()` → `normalize_scores()`
- **Impact**: ~35 lines - **DO NOT REMOVE**

### 8. **`get_entity_details(entity_uri)`** (Line 1388)
- **Purpose**: Get detailed information about a specific entity via SPARQL
- **Why Unused**: Not used in current retrieval or display pipeline
- **Called By**: NEVER called
- **Impact**: ~30 lines

### 9. **`get_wikidata_entities()`** (Line 1898)
- **Purpose**: Get all Wikidata entity mappings from the RDF graph
- **Why Unused**: System now fetches Wikidata on-demand per entity, not bulk upfront
- **Called By**: NEVER called
- **Impact**: ~35 lines

### 10. **`batch_process_documents(entities, batch_size, sleep_time)`** (Line 1936)
- **Purpose**: Process entities in batches with rate limiting
- **Why Unused**: The batching logic was moved directly into `process_rdf_data()`
- **Called By**: NEVER called
- **Impact**: ~45 lines
- **Note**: `process_rdf_data()` now has inline batching code (lines 828-896)

### 11. **`get_subgraph(doc_ids)`** in graph_document_store.py (Line 216)
- **Purpose**: Extract a subgraph of documents given a list of IDs
- **Why Unused**: The subgraph extraction logic now uses `compute_coherent_subgraph()` instead
- **Called By**: NEVER called
- **Impact**: ~30 lines
- **Note**: This was likely an earlier version of subgraph extraction

---

## 🟡 POTENTIALLY UNUSED HELPER FUNCTIONS

These helper functions may or may not be useful depending on usage patterns. Need deeper investigation.

### 11. **Internal Nested Functions** (Lines 238-286)
- `load_document_graph()` and `save_document_graph()` inside `initialize()`
- **Status**: These are used, but only within the nested scope
- **Note**: Could potentially be refactored to class-level methods for clarity

---

## ✅ ACTIVELY USED FUNCTIONS

These functions are essential to the current system operation:

### Core Initialization & Setup
- `__init__()` - Constructor
- `_load_property_labels()` - Load ontology labels
- `embeddings` - Property for embedding provider
- `test_connection()` - SPARQL connection test
- `initialize()` - System initialization

### Document Processing & Graph Building
- `process_rdf_data()` - Main RDF processing pipeline
- `create_enhanced_document()` - Create rich entity documents
- `save_entity_document()` - Save documents to disk
- `build_vector_store_batched()` - Build FAISS index
- `get_all_entities()` - Extract entities from SPARQL
- `get_outgoing_relationships()` - Get entity outgoing edges
- `get_incoming_relationships()` - Get entity incoming edges
- `get_entity_literals()` - Get literal properties

### Filtering & Utility
- `is_schema_predicate()` - Filter schema-level predicates (✅ USED in `get_entity_context()`)
- `is_technical_class_name()` - Filter technical CIDOC class names (✅ USED in `process_rdf_data()`)
- `get_entity_label()` - Get human-readable entity labels
- `process_cidoc_relationship()` - Convert RDF triples to natural language

### Retrieval & Question Answering
- `retrieve()` - Main retrieval entry point
- `cidoc_aware_retrieval()` - CIDOC-aware document retrieval (used by `retrieve()`)
- `compute_coherent_subgraph()` - Subgraph extraction algorithm
- `answer_question()` - Main QA entry point (called from main.py)
- `get_cidoc_system_prompt()` - Generate system prompt

### Context & Traversal
- `get_entity_context()` - Bidirectional graph traversal for context

### Wikidata Integration
- `get_wikidata_for_entity()` - Get Wikidata ID for entity
- `fetch_wikidata_info()` - Fetch Wikidata details via API

---

## 📊 Call Chain Diagram

### Current Active Path (QA)
```
main.py: /api/chat
  └─> answer_question()
       └─> retrieve()
            ├─> cidoc_aware_retrieval()  [First stage retrieval]
            └─> compute_coherent_subgraph()  [Subgraph extraction]
       └─> get_wikidata_for_entity()
       └─> fetch_wikidata_info()
```

### Current Active Path (Graph Building)
```
initialize()
  └─> process_rdf_data()
       ├─> get_all_entities()
       ├─> create_enhanced_document()
       │    ├─> get_entity_context()
       │    │    ├─> is_schema_predicate()  ✅ USED
       │    │    └─> process_cidoc_relationship()
       │    │         └─> get_entity_label()
       │    └─> is_technical_class_name()  ✅ USED
       ├─> save_entity_document()
       ├─> get_outgoing_relationships()
       ├─> get_incoming_relationships()
       └─> build_vector_store_batched()
```

### DEAD Code Paths (Never Executed)
```
❌ hybrid_answer_question()  [NEVER CALLED]
    └─> answer_with_direct_query()
         └─> generate_sparql_query()

❌ relationship_aware_retrieval()  [NEVER CALLED]
    ├─> extract_entities_from_query()
    ├─> calculate_relationship_scores()
    └─> normalize_scores()

❌ get_entity_details()  [NEVER CALLED]
❌ get_wikidata_entities()  [NEVER CALLED]
❌ batch_process_documents()  [NEVER CALLED]
```

---

## 🎯 Recommendations

### Priority 1: Safe Deletions (High Confidence)
Remove these functions immediately - they are confirmed unused:

**In universal_rag_system.py:**
1. `generate_sparql_query()`
2. `answer_with_direct_query()`
3. `hybrid_answer_question()`
4. `relationship_aware_retrieval()`
5. `extract_entities_from_query()`
6. `calculate_relationship_scores()`
7. ~~`normalize_scores()`~~ **⚠️ DO NOT REMOVE - ACTIVELY USED**
8. `get_entity_details()`
9. `get_wikidata_entities()`
10. `batch_process_documents()`

**In graph_document_store.py:**
11. `get_subgraph()`

**Estimated cleanup**: ~425 lines removed (excluding `normalize_scores()`)

### Priority 2: Archive for Reference
If you want to preserve these for future experimentation:
- Move them to a `legacy_functions.py` file in the `legacy_approach/` directory
- Document why they were replaced
- This maintains code history without cluttering the main system

### Priority 3: Code Quality Improvements
- The nested `load_document_graph()` and `save_document_graph()` functions could be promoted to class-level methods
- Consider extracting the inline batching logic from `process_rdf_data()` if you want to make it reusable

---

## 🔍 Answers to Your Question

> **"what is the use of `is_technical_class_name()`?"**

**Answer**: `is_technical_class_name()` IS ACTUALLY USED! It's called in:
- `process_rdf_data()` line 869: Filters out technical CIDOC-CRM class names (like "E22_Man-Made_Object") when determining an entity's primary type
- `create_enhanced_document()` line 687: Filters technical class names from entity types before displaying them

**Purpose**: Prevents ugly technical identifiers like "E22_Man-Made_Object", "E53_Place" from appearing in the generated documents. Instead, it allows human-readable types like "Church", "Painting", "Location" to be used.

**Verdict**: ✅ **KEEP THIS FUNCTION** - It's actively used and important for clean output.

---

## 📝 Action Plan

1. **Backup first**: Commit current code before deletions
2. **Delete unused functions** listed in Priority 1
3. **Run tests** to ensure nothing breaks
4. **Rebuild with** `--rebuild` flag to verify graph building still works
5. **Test QA** to verify retrieval still works
6. **Consider archiving** deleted code to `legacy_approach/` for reference

After cleanup, your code will be ~460 lines shorter and much easier to maintain!

---

## 📋 Quick Summary for Deletion

### Files to modify:
1. **universal_rag_system.py** - Remove 10 unused functions (~430 lines)
2. **graph_document_store.py** - Remove 1 unused function (~30 lines)

### Total impact:
- **Lines removed**: ~460
- **Functions removed**: 11
- **Risk level**: Low (all functions are confirmed unused via grep search)
- **Testing needed**: Run system with `--rebuild` and verify QA works

### Git commit message suggestion:
```
chore: Remove unused legacy functions

- Remove 10 unused functions from universal_rag_system.py
  - Legacy SPARQL query generation (generate_sparql_query, answer_with_direct_query, hybrid_answer_question)
  - Legacy PersonalizedPageRank retrieval (relationship_aware_retrieval, calculate_relationship_scores, normalize_scores, extract_entities_from_query)
  - Unused utility functions (get_entity_details, get_wikidata_entities, batch_process_documents)
- Remove unused get_subgraph from graph_document_store.py
- All removed functions are confirmed unused via codebase analysis
- Reduces codebase by ~460 lines
- No functional changes to active code paths
```
