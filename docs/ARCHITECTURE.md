# RAG System Architecture

**Universal Graph-Based RAG System for CIDOC-CRM and Semantic RDF Data**

Version: 2.0
Last Updated: 2026-01-24

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Key Algorithms](#key-algorithms)
6. [File Structure](#file-structure)
7. [Component Interactions](#component-interactions)
8. [Configuration](#configuration)
9. [Extension Points](#extension-points)
10. [Performance Considerations](#performance-considerations)
11. [Troubleshooting](#troubleshooting)

---

## System Overview

### Purpose

This system implements a **Retrieval-Augmented Generation (RAG)** pipeline specialized for **CIDOC-CRM** (Conceptual Reference Model) and semantic RDF data. It transforms complex ontology-based knowledge graphs into natural language documents and enables intelligent question-answering over cultural heritage datasets.

### Key Features

- **Multi-Dataset Support**: Lazy-loaded datasets with per-dataset caching and configuration
- **CIDOC-CRM Aware**: Deep understanding of CIDOC-CRM ontologies and extensions (VIR, CRMdig, FRBRoo)
- **Graph-Based Retrieval**: Uses graph structure and relationships for coherent document selection
- **Local Embeddings**: Fast sentence-transformers embeddings (no API rate limits)
- **Cluster Pipeline**: Unified workflow for GPU cluster processing of large datasets
- **Multi-LLM Support**: Abstraction layer supporting OpenAI, Anthropic, R1, Ollama, and local models
- **Embedding Cache**: Resumable processing with disk-based embedding cache
- **Coherent Subgraph Extraction**: Balances relevance and connectivity for better context
- **Natural Language Generation**: Converts technical RDF triples into readable descriptions

### Technology Stack

- **Language**: Python 3.x
- **Graph Processing**: NetworkX, NumPy
- **Vector Storage**: FAISS (Facebook AI Similarity Search)
- **Semantic Queries**: SPARQL (via SPARQLWrapper)
- **Ontology Processing**: RDFLib
- **LLM Integration**: LangChain (OpenAI, Anthropic, Cohere, Ollama)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INITIALIZATION PHASE                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  1. ONTOLOGY EXTRACTION LAYER                 │
        │  (extract_ontology_labels.py)                 │
        │                                               │
        │  Input:  ontology/*.ttl, *.rdf, *.owl        │
        │  Output: property_labels.json                 │
        │          ontology_classes.json                │
        │          class_labels.json                    │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  2. RDF DATA PROCESSING LAYER                 │
        │  (universal_rag_system.py)                    │
        │                                               │
        │  • Query SPARQL endpoint for entities         │
        │  • Get entity types, properties, relations    │
        │  • Convert to natural language documents      │
        │  • Track missing ontology elements            │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  3. GRAPH CONSTRUCTION LAYER                  │
        │  (graph_document_store.py)                    │
        │                                               │
        │  • Create GraphDocument nodes                 │
        │  • Build edges with CIDOC-CRM weights         │
        │  • Generate embeddings                        │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  4. VECTOR STORAGE LAYER                      │
        │  (FAISS + Pickle)                             │
        │                                               │
        │  • Store document_graph.pkl                   │
        │  • Build vector_index/ (FAISS)                │
        │  • Save entity_documents/*.md                 │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  5. VALIDATION & REPORTING                    │
        │                                               │
        │  • Generate ontology_validation_report.txt    │
        │  • Log missing classes/properties             │
        │  • Provide actionable fix instructions        │
        └───────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           QUERY PHASE                                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  1. VECTOR RETRIEVAL                          │
        │  (graph_document_store.retrieve())            │
        │                                               │
        │  • Embed query using LLM provider             │
        │  • FAISS similarity search (k × 2)            │
        │  • Return candidate documents                 │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  2. CIDOC-CRM AWARE RETRIEVAL                 │
        │  (cidoc_aware_retrieval())                    │
        │                                               │
        │  • Build subgraph of candidates               │
        │  • Query SPARQL for relationships             │
        │  • Apply relationship weights                 │
        │  • Combine vector + graph scores              │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  3. COHERENT SUBGRAPH EXTRACTION              │
        │  (compute_coherent_subgraph())                │
        │                                               │
        │  • Build adjacency matrix (multi-hop)         │
        │  • Greedy selection algorithm                 │
        │  • Balance relevance + connectivity           │
        │  • Return top k documents                     │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  4. CONTEXT ASSEMBLY                          │
        │                                               │
        │  • Combine selected documents                 │
        │  • Add Wikidata context (optional)            │
        │  • Format for LLM consumption                 │
        └───────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │  5. ANSWER GENERATION                         │
        │  (llm_provider.generate())                    │
        │                                               │
        │  • Apply CIDOC-CRM system prompt              │
        │  • Generate natural language answer           │
        │  • Return with source attribution             │
        └───────────────────────────────────────────────┘
```

---

## Core Components

### 1. Ontology Extraction Layer

**File**: `extract_ontology_labels.py`

**Purpose**: Extract English labels and class definitions from ontology files to enable proper natural language generation.

**Key Functions**:
- `extract_property_labels()`: Extracts property URI → English label mappings
- `extract_ontology_classes()`: Extracts class URIs and their English labels
- `run_extraction()`: Main orchestrator that generates all JSON label files

**Inputs**:
- `ontology/*.ttl`, `*.rdf`, `*.owl`, `*.n3` files
- CIDOC-CRM core and extensions (VIR, CRMdig, FRBRoo, etc.)

**Outputs**:
```
property_labels.json       # URI → English label (properties)
ontology_classes.json      # Set of technical class identifiers
class_labels.json          # URI → English label (classes)
```

**Process**:
1. Parse RDF ontology files using RDFLib
2. Query for properties (rdf:Property, owl:ObjectProperty, owl:DatatypeProperty)
3. Query for classes (owl:Class, rdfs:Class, rdf:Class)
4. Extract English labels (lang="en") preferentially
5. Fallback to deriving labels from URI local names
6. Save to JSON for fast lookup during processing

---

### 2. RDF Data Processing Layer

**File**: `universal_rag_system.py`

**Purpose**: Core RAG system that processes RDF data, builds graph documents, and handles retrieval.

**Key Classes**:
- `UniversalRagSystem`: Main orchestrator
- `RetrievalConfig`: Configuration constants

**Key Methods**:

#### Initialization & Setup
- `__init__()`: Initialize SPARQL endpoint, LLM provider, document store
- `_load_property_labels()`: Load property labels (auto-extract if missing)
- `_load_ontology_classes()`: Load ontology classes (auto-extract if missing)
- `_load_class_labels()`: Load class labels (auto-extract if missing)
- `initialize()`: Test connection, load or build document graph

#### RDF Processing
- `get_all_entities()`: Query SPARQL for all entities with literals
- `create_enhanced_document()`: Convert entity to natural language document
  - Get entity types (with fallback strategy)
  - Get entity literals (labels, descriptions, WKT, dates)
  - Get entity context via `get_entity_context()`
  - Filter out technical CIDOC-CRM class names
  - Format as markdown document
- `process_cidoc_relationship()`: Convert RDF triple to natural language
- `save_entity_document()`: Save markdown file to `entity_documents/`

#### Graph Construction
- `process_rdf_data()`: Main processing pipeline
  1. Get all entities from SPARQL
  2. Create enhanced documents for each entity
  3. Add documents to graph store
  4. Build edges with CIDOC-CRM relationship weights
  5. Save document graph and vector index
  6. Generate validation report

#### Retrieval & Search
- `cidoc_aware_retrieval()`: First-stage retrieval with graph scoring
  - Vector similarity search
  - Query SPARQL for relationships between candidates
  - Apply relationship weights (P89_falls_within, P55_has_current_location, etc.)
  - Combine vector + graph scores
- `compute_coherent_subgraph()`: Second-stage coherent extraction
  - Build multi-hop adjacency matrix
  - Greedy selection balancing relevance + connectivity
  - Normalize scores for fair combination
- `retrieve()`: Orchestrate retrieval pipeline
  1. CIDOC-aware retrieval (initial pool)
  2. Create adjacency matrix
  3. Extract coherent subgraph
  4. Return top k documents

#### Answer Generation
- `answer_question()`: End-to-end question answering
  1. Retrieve relevant documents
  2. Fetch Wikidata context (optional)
  3. Assemble context
  4. Generate answer using LLM
  5. Return answer with sources

#### Validation
- `generate_validation_report()`: Create comprehensive report
  - List missing classes and properties
  - Provide step-by-step fix instructions
  - Save to `ontology_validation_report.txt`

---

### 3. Graph Document Store

**File**: `graph_document_store.py`

**Purpose**: Manage graph-structured documents with vector retrieval capabilities.

**Key Classes**:
- `GraphDocument`: Document node with neighbors
  - `id`: Entity URI
  - `text`: Natural language document
  - `metadata`: Entity type, label, raw triples, etc.
  - `embedding`: Vector representation
  - `neighbors`: List of edges with weights
- `GraphDocumentStore`: Store for all documents
  - `docs`: Dict of entity URI → GraphDocument
  - `vector_store`: FAISS index for vector search
  - `embeddings_model`: LLM provider embeddings

**Key Methods**:
- `add_document()`: Add entity document to graph
- `add_edge()`: Create bidirectional edge with weight
- `rebuild_vector_store()`: Build FAISS index from documents
- `retrieve()`: Vector similarity search
- `create_adjacency_matrix()`: Build weighted adjacency matrix
  - Multi-hop connections (up to max_hops)
  - Symmetric normalization
  - Numerical stability handling
- `save_document_graph()`: Serialize to pickle
- `load_document_graph()`: Load from pickle

---

### 4. LLM Provider Abstraction

**File**: `llm_providers.py`

**Purpose**: Unified interface for multiple LLM and embedding providers.

**Key Classes**:
- `BaseLLMProvider`: Abstract base class
  - `generate()`: Generate text completion
  - `get_embeddings()`: Get vector embeddings
  - `supports_batch_embedding()`: Check batch support
- `OpenAIProvider`: OpenAI API (GPT-4o, text-embedding-3-small)
- `AnthropicProvider`: Anthropic Claude API (uses OpenAI for embeddings)
- `SentenceTransformersProvider`: Local embeddings (BAAI/bge-m3, etc.)
- `R1Provider`: DeepSeek R1 API
- `OllamaProvider`: Local Ollama models

**Factory Functions**:
- `get_llm_provider()`: Create LLM provider from config
- `get_embedding_provider()`: Create embedding provider (can differ from LLM)

**Configuration Example**:
```python
# OpenAI for both LLM and embeddings
config = {
    "llm_provider": "openai",
    "api_key": "sk-...",
    "model": "gpt-4o",
    "embedding_model": "text-embedding-3-small",
    "temperature": 0.7
}

# OpenAI for LLM, local for embeddings
config = {
    "llm_provider": "openai",
    "embedding_provider": "local",
    "embedding_model": "BAAI/bge-m3",
    "embedding_device": "cuda",  # or "mps", "cpu"
    "embedding_batch_size": 64
}
```

### 5. Dataset Manager

**File**: `dataset_manager.py`

**Purpose**: Manage multiple RAG system instances with lazy loading.

**Key Class**: `DatasetManager`
- `list_datasets()`: Return available datasets with status
- `get_dataset()`: Get or lazy-initialize RAG system for dataset
- `is_initialized()`: Check if dataset is loaded in memory
- `get_cache_paths()`: Return dataset-specific cache paths
- `get_interface_config()`: Merge default interface with dataset overrides

**Configuration** (`config/datasets.yaml`):
```yaml
default_dataset: asinou

datasets:
  asinou:
    name: asinou
    display_name: "Asinou Church"
    endpoint: "http://localhost:3030/asinou/sparql"
    embedding:
      provider: local
      model: BAAI/bge-m3
    interface:
      page_title: "Asinou Dataset Chat"
      example_questions:
        - "Where is Panagia Phorbiottisa located?"
```

### 6. Embedding Cache

**File**: `embedding_cache.py`

**Purpose**: Disk-based embedding cache for resumable processing.

**Key Class**: `EmbeddingCache`
- `get()`: Retrieve cached embedding
- `set()`: Cache an embedding
- `get_batch()`: Batch retrieval with cache hits/misses
- `count()`: Number of cached embeddings
- `get_stats()`: Cache statistics

**Features**:
- Stores embeddings as NumPy arrays in subdirectories
- Uses MD5 hash of document ID for file names
- Enables stopping and resuming large dataset processing

---

## Data Flow

### Initialization Flow

```
1. System Start
   └─> Load ontology labels (property_labels.json, class_labels.json, ontology_classes.json)
       └─> If missing: Run extract_ontology_labels.py

2. Initialize LLM Provider
   └─> Create embeddings model
   └─> Create chat completion model

3. Initialize Document Store
   └─> Create GraphDocumentStore with embeddings

4. Check for Cached Data
   ├─> If document_graph.pkl exists:
   │   └─> Load from disk
   └─> Else:
       └─> Process RDF Data (see below)

5. Process RDF Data (if no cache)
   ├─> Query SPARQL for all entities
   ├─> For each entity:
   │   ├─> Get types (check class_labels.json)
   │   ├─> Get literals (labels, descriptions, etc.)
   │   ├─> Get relationships (convert using property_labels.json)
   │   ├─> Create natural language document
   │   ├─> Save to entity_documents/*.md
   │   ├─> Add to graph store
   │   └─> Generate embedding
   │
   ├─> Build graph edges with CIDOC-CRM weights
   ├─> Save document_graph.pkl
   ├─> Build FAISS vector index
   ├─> Save to vector_index/
   └─> Generate validation report
```

### Query Flow

```
1. User Question
   └─> "What churches are depicted in Byzantine icons?"

2. Vector Retrieval (Initial)
   ├─> Embed query using LLM provider
   ├─> FAISS similarity search (k × 2 candidates)
   └─> Get candidate GraphDocuments

3. CIDOC-Aware Retrieval
   ├─> Build subgraph of candidates
   ├─> Query SPARQL for relationships
   │   └─> Example: P89_falls_within, P62_depicts, etc.
   ├─> Apply relationship weights
   │   └─> Higher weights for spatial/visual relationships
   ├─> Combine vector score + graph score
   └─> Rank candidates

4. Coherent Subgraph Extraction
   ├─> Build multi-hop adjacency matrix
   ├─> Normalize with symmetric normalization
   ├─> Greedy selection algorithm:
   │   ├─> Start with highest relevance document
   │   ├─> Iteratively select documents that:
   │   │   └─> Balance relevance (vector score) + connectivity (graph score)
   │   └─> Stop at k documents
   └─> Return coherent subgraph

5. Context Assembly
   ├─> Combine selected documents
   ├─> Optional: Fetch Wikidata context
   ├─> Format for LLM consumption
   └─> Prepare source attribution

6. Answer Generation
   ├─> Apply CIDOC-CRM system prompt
   │   └─> "You are a cultural heritage expert..."
   │   └─> "Never use technical ontology identifiers..."
   ├─> Generate answer using LLM
   └─> Return answer + sources
```

---

## Key Algorithms

### 1. CIDOC-CRM Aware Retrieval

**Purpose**: Enhance vector retrieval with ontology-specific relationship weights.

**Algorithm**:
```python
def cidoc_aware_retrieval(query, k):
    # Step 1: Vector search
    candidates = vector_store.similarity_search(query, k × 2)

    # Step 2: Build relationship graph
    G = NetworkX.DiGraph()
    for doc in candidates:
        G.add_node(doc.id, score=0.0)

    # Step 3: Query SPARQL for relationships
    relationships = sparql.query("""
        SELECT ?subject ?predicate ?object
        WHERE {
            ?subject ?predicate ?object .
            FILTER(?subject IN [...candidates...])
            FILTER(?object IN [...candidates...])
        }
    """)

    # Step 4: Add weighted edges
    relationship_weights = {
        "P89_falls_within": 0.9,        # Spatial containment
        "P55_has_current_location": 0.9, # Location
        "P56_bears_feature": 0.8,        # Physical features
        "P46_is_composed_of": 0.8,       # Part-whole
        "P108i_was_produced_by": 0.7,    # Creation
        "K24_portray": 0.7,              # Visual representation
        "P2_has_type": 0.6               # Type classification
    }

    for rel in relationships:
        weight = relationship_weights.get(rel.predicate, 0.5)
        G.add_edge(rel.subject, rel.object, weight=weight)

    # Step 5: Combine scores
    for i, doc in enumerate(candidates):
        vector_score = (len(candidates) - i) / len(candidates)
        graph_score = G.nodes[doc.id].get("score", 0.0)
        combined_score = α × vector_score + (1 - α) × graph_score
        doc.combined_score = combined_score

    # Step 6: Re-rank and return top k
    candidates.sort(key=lambda d: d.combined_score, reverse=True)
    return candidates[:k]
```

**Parameters**:
- `α` (VECTOR_PAGERANK_ALPHA): 0.6 (60% vector, 40% graph)

---

### 2. Coherent Subgraph Extraction

**Purpose**: Select documents that are both relevant and well-connected.

**Algorithm**:
```python
def compute_coherent_subgraph(candidates, adjacency_matrix, initial_scores, k, α):
    n = len(candidates)
    selected = []
    selected_mask = [False] × n

    # Normalize initial scores to [0, 1]
    scores = normalize(initial_scores)

    # Step 1: Select highest relevance document
    first_idx = argmax(scores)
    selected.append(first_idx)
    selected_mask[first_idx] = True

    # Step 2: Iteratively select k-1 more documents
    for iteration in range(1, k):
        best_idx = -1
        best_score = -∞

        # For each unselected candidate
        for idx in range(n):
            if selected_mask[idx]:
                continue

            # Compute connectivity to selected documents
            connectivity = 0.0
            for selected_idx in selected:
                edge_weight = max(
                    adjacency_matrix[idx, selected_idx],
                    adjacency_matrix[selected_idx, idx]
                )
                connectivity += edge_weight

            # Average by number of selected
            connectivity /= len(selected)

            # Normalize connectivity to [0, 1]
            connectivity_norm = normalize([connectivity])[0]

            # Combine relevance + connectivity
            combined = α × scores[idx] + (1 - α) × connectivity_norm

            if combined > best_score:
                best_score = combined
                best_idx = idx

        # Add best document to selection
        selected.append(best_idx)
        selected_mask[best_idx] = True

    return [candidates[idx] for idx in selected]
```

**Parameters**:
- `α` (RELEVANCE_CONNECTIVITY_ALPHA): 0.7 (70% relevance, 30% connectivity)
- `k`: Number of documents to select (default: 10)

---

### 3. Multi-Hop Adjacency Matrix

**Purpose**: Capture both direct and indirect connections between documents.

**Algorithm**:
```python
def create_adjacency_matrix(doc_ids, max_hops=2):
    n = len(doc_ids)
    adj_matrix = zeros(n, n)

    # Step 1: Fill with direct edges (1-hop)
    for i, doc_id in enumerate(doc_ids):
        doc = docs[doc_id]
        for neighbor in doc.neighbors:
            if neighbor.doc_id in doc_ids:
                j = doc_ids.index(neighbor.doc_id)
                adj_matrix[i, j] = neighbor.weight

    original_adj = adj_matrix.copy()
    current_power = original_adj.copy()

    # Step 2: Add multi-hop connections
    for hop in range(2, max_hops + 1):
        # Matrix multiplication: A^hop = A^(hop-1) × A
        current_power = current_power @ original_adj

        # Add with reduced weight (1/hop)
        adj_matrix += current_power × (1.0 / hop)

    # Step 3: Add self-loops
    adj_matrix += identity(n)

    # Step 4: Symmetric normalization
    # D^(-1/2) × A × D^(-1/2)
    rowsum = adj_matrix.sum(axis=1)
    d_inv_sqrt = power(rowsum, -0.5)
    d_mat_inv_sqrt = diag(d_inv_sqrt)
    adj_normalized = d_mat_inv_sqrt @ adj_matrix @ d_mat_inv_sqrt

    return adj_normalized
```

**Example**:
```
Document A → B (weight=1.5)
Document B → C (weight=1.0)

1-hop: A→B (1.5), B→C (1.0)
2-hop: A→C (1.5 × 1.0 × 1/2 = 0.75)

Final adjacency:
  A    B    C
A [1.0  1.5  0.75]
B [1.5  1.0  1.0 ]
C [0.75 1.0  1.0 ]
```

---

## File Structure

### Project Organization

```
CRM_RAG/
├── main.py                          # Flask web application
├── universal_rag_system.py          # Core RAG orchestrator
├── graph_document_store.py          # Graph-based document storage
├── llm_providers.py                 # LLM abstraction layer
├── dataset_manager.py               # Multi-dataset management
├── embedding_cache.py               # Disk-based embedding cache
├── config_loader.py                 # Configuration loading
│
├── config/                          # Configuration files
│   ├── .env.openai                  # OpenAI provider config
│   ├── .env.local                   # Local embeddings config
│   ├── .env.cluster                 # GPU cluster config
│   ├── .env.secrets                 # API keys (git-ignored)
│   ├── datasets.yaml                # Multi-dataset configuration
│   ├── interface.yaml               # Chat UI customization
│   └── event_classes.json           # CIDOC-CRM event classes
│
├── data/                            # All data files
│   ├── ontologies/                  # Ontology files
│   │   ├── CIDOC_CRM_v7.1.3.rdf
│   │   ├── vir.ttl
│   │   └── CRMdig_v3.2.1.rdf
│   │
│   ├── labels/                      # Extracted ontology labels (shared)
│   │   ├── property_labels.json
│   │   ├── class_labels.json
│   │   └── ontology_classes.json
│   │
│   ├── exports/                     # RDF bulk exports
│   │   └── <dataset>_dump.ttl
│   │
│   ├── cache/                       # Per-dataset caches
│   │   └── <dataset>/
│   │       ├── document_graph.pkl
│   │       ├── vector_index/
│   │       └── embeddings/
│   │
│   └── documents/                   # Per-dataset entity documents
│       └── <dataset>/
│           ├── entity_documents/*.md
│           └── documents_metadata.json
│
├── scripts/                         # Utility scripts
│   ├── extract_ontology_labels.py   # Ontology label extraction
│   ├── bulk_generate_documents.py   # Fast bulk RDF export
│   ├── cluster_pipeline.py          # Unified cluster workflow
│   └── test_entity_context.py       # Debug utility
│
├── static/                          # Web interface assets
├── templates/                       # Flask HTML templates
├── logs/                            # Application logs
└── docs/                            # Documentation
```

### Data Files

| File | Purpose | Format | Generated By |
|------|---------|--------|--------------|
| `data/labels/property_labels.json` | Property URI → English label | JSON dict | `extract_ontology_labels.py` |
| `data/labels/ontology_classes.json` | Set of ontology class identifiers | JSON array | `extract_ontology_labels.py` |
| `data/labels/class_labels.json` | Class URI → English label | JSON dict | `extract_ontology_labels.py` |
| `data/exports/<dataset>_dump.ttl` | Bulk RDF export from SPARQL | Turtle | `bulk_generate_documents.py` |
| `data/cache/<dataset>/document_graph.pkl` | Serialized GraphDocument objects | Pickle | `process_rdf_data()` |
| `data/cache/<dataset>/vector_index/` | FAISS vector index | FAISS binary | `rebuild_vector_store()` |
| `data/cache/<dataset>/embeddings/` | Cached embeddings for resumability | NumPy | `EmbeddingCache` |
| `data/documents/<dataset>/entity_documents/*.md` | Human-readable entity documents | Markdown | `save_entity_document()` |
| `data/documents/<dataset>/documents_metadata.json` | Entity metadata for embedding | JSON | `generate_documents_only()` |
| `logs/ontology_validation_report.txt` | Validation report | Text | `generate_validation_report()` |

---

## Component Interactions

### 1. Initialization Phase

```
User Script
    │
    ├─> UniversalRagSystem.__init__()
    │       │
    │       ├─> _load_property_labels()
    │       │       │
    │       │       ├─> Check property_labels.json exists
    │       │       └─> If not: run_extraction()
    │       │                   │
    │       │                   └─> extract_ontology_labels.py
    │       │
    │       ├─> _load_ontology_classes()
    │       │       └─> Similar to above
    │       │
    │       ├─> _load_class_labels()
    │       │       └─> Similar to above
    │       │
    │       └─> get_llm_provider()
    │               │
    │               └─> llm_providers.py
    │
    └─> initialize()
            │
            ├─> test_connection() → SPARQL endpoint
            │
            ├─> Check document_graph.pkl exists
            │   ├─> Yes: load_document_graph()
            │   └─> No:  process_rdf_data()
            │               │
            │               ├─> get_all_entities() → SPARQL
            │               │
            │               ├─> For each entity:
            │               │   ├─> create_enhanced_document()
            │               │   │       │
            │               │   │       ├─> Query types → SPARQL
            │               │   │       │   └─> Lookup in class_labels.json
            │               │   │       │
            │               │   │       ├─> get_entity_context()
            │               │   │       │   └─> Query relationships → SPARQL
            │               │   │       │       └─> Lookup in property_labels.json
            │               │   │       │
            │               │   │       └─> Convert to natural language
            │               │   │
            │               │   ├─> save_entity_document()
            │               │   │
            │               │   └─> document_store.add_document()
            │               │           └─> Generate embedding → LLM Provider
            │               │
            │               ├─> Build graph edges
            │               │   └─> document_store.add_edge()
            │               │
            │               ├─> build_vector_store_batched()
            │               │   └─> FAISS.from_documents()
            │               │
            │               └─> generate_validation_report()
            │
            └─> Save caches
```

### 2. Query Phase

```
User Question
    │
    └─> answer_question()
            │
            ├─> retrieve()
            │       │
            │       ├─> cidoc_aware_retrieval()
            │       │       │
            │       │       ├─> document_store.retrieve()
            │       │       │       └─> FAISS similarity search
            │       │       │
            │       │       ├─> Query SPARQL for relationships
            │       │       │
            │       │       ├─> Build graph with weights
            │       │       │
            │       │       └─> Combine vector + graph scores
            │       │
            │       ├─> create_adjacency_matrix()
            │       │       └─> Multi-hop graph matrix
            │       │
            │       └─> compute_coherent_subgraph()
            │               └─> Greedy selection
            │
            ├─> get_wikidata_for_entity() (optional)
            │       └─> fetch_wikidata_info()
            │
            ├─> Assemble context from documents
            │
            ├─> llm_provider.generate()
            │       └─> OpenAI/Anthropic/R1/Ollama API
            │
            └─> Return answer + sources
```

---

## Configuration

### Configuration Files

Configuration is split across multiple files in `config/`:

**Provider Configuration** (`config/.env.openai`, `.env.local`, `.env.cluster`):
```bash
# LLM Provider
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o

# Embedding Provider (can differ from LLM)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=auto
EMBEDDING_BATCH_SIZE=64

# Cache settings
USE_EMBEDDING_CACHE=true

# Server settings
TEMPERATURE=0.7
PORT=5001
```

**API Keys** (`config/.env.secrets`):
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**Dataset Configuration** (`config/datasets.yaml`):
```yaml
default_dataset: asinou

datasets:
  asinou:
    name: asinou
    display_name: "Asinou Church"
    endpoint: "http://localhost:3030/asinou/sparql"
    embedding:
      provider: local
      model: BAAI/bge-m3
    interface:
      page_title: "Asinou Dataset Chat"
      example_questions:
        - "Where is Panagia Phorbiottisa located?"

  museum:
    name: museum
    display_name: "Museum Collection"
    endpoint: "http://localhost:3030/museum/sparql"
```

**Interface Customization** (`config/interface.yaml`):
```yaml
page_title: "Cultural Heritage Chat"
header_title: "Heritage Assistant"
welcome_message: "Ask me about cultural heritage..."
example_questions:
  - "What churches have Byzantine frescoes?"
```

### System Configuration

In `universal_rag_system.py`:

```python
class RetrievalConfig:
    # Score combination weights
    VECTOR_PAGERANK_ALPHA = 0.6          # Vector vs Graph in first stage
    RELEVANCE_CONNECTIVITY_ALPHA = 0.7   # Relevance vs Connectivity in subgraph

    # PageRank parameters
    PAGERANK_DAMPING = 0.85
    PAGERANK_ITERATIONS = 20

    # Rate limiting
    TOKENS_PER_MINUTE_LIMIT = 950_000

    # Retrieval parameters
    DEFAULT_RETRIEVAL_K = 10             # Final documents to return
    INITIAL_POOL_MULTIPLIER = 2          # Candidate pool size

    # Processing parameters
    DEFAULT_BATCH_SIZE = 50
    ENTITY_CONTEXT_DEPTH = 2             # Relationship traversal depth
    MAX_ADJACENCY_HOPS = 2               # Multi-hop connections
```

---

## Extension Points

### 1. Adding New Ontologies

**Steps**:
1. Obtain ontology file (.ttl, .rdf, .owl)
2. Copy to `ontology/` directory
3. Run `python extract_ontology_labels.py`
4. Delete caches and rebuild:
   ```bash
   rm -rf document_graph.pkl vector_index/ entity_documents/
   ```
5. Re-initialize RAG system

### 2. Adding New LLM Providers

**Steps**:
1. Create new provider class in `llm_providers.py`:
   ```python
   class MyProvider(BaseLLMProvider):
       def __init__(self, api_key, model, **kwargs):
           self.api_key = api_key
           self.model = model

       def generate(self, system_prompt, user_prompt):
           # Implement API call
           pass

       def get_embeddings(self, text):
           # Implement embedding generation
           pass
   ```

2. Add to factory function:
   ```python
   def get_llm_provider(provider_name, config):
       if provider_name == "my_provider":
           return MyProvider(
               api_key=config.get("api_key"),
               model=config.get("model")
           )
   ```

3. Update `.env`:
   ```bash
   LLM_PROVIDER=my_provider
   MY_PROVIDER_API_KEY=...
   ```

### 3. Custom Relationship Weights

In `cidoc_aware_retrieval()`, modify:

```python
relationship_weights = {
    "http://www.cidoc-crm.org/cidoc-crm/P89_falls_within": 0.9,
    "http://www.cidoc-crm.org/cidoc-crm/P55_has_current_location": 0.9,
    "http://w3id.org/vir#K24_portray": 0.7,
    "http://my-ontology.org/myProperty": 0.8,  # Add custom
}
```

### 4. Custom Document Formatting

Override `create_enhanced_document()`:

```python
def create_enhanced_document(self, entity_uri):
    # Custom logic here
    text = f"# {entity_label}\n\n"
    text += "## Custom Section\n"
    # ... add your custom sections
    return text, entity_label, entity_types, raw_triples
```

### 5. Alternative Vector Stores

Replace FAISS with another vector store:

```python
from langchain_community.vectorstores import Chroma, Pinecone, Qdrant

class GraphDocumentStore:
    def rebuild_vector_store(self):
        # Replace FAISS with Chroma
        self.vector_store = Chroma.from_documents(
            docs_for_faiss,
            self.embeddings_model,
            persist_directory="./chroma_db"
        )
```

---

## Performance Considerations

### Memory Usage

- **Document Graph**: ~1MB per 1000 entities
- **FAISS Index**: ~4KB per document (for 1536-dim embeddings)
- **Entity Documents**: ~5KB per markdown file

**Optimization**:
- Use batched processing (`DEFAULT_BATCH_SIZE = 50`)
- Implement rate limiting (`TOKENS_PER_MINUTE_LIMIT`)
- Cache embeddings and document graph

### Query Latency

Typical query breakdown (for 10k entity dataset):
- Vector search: ~50ms
- SPARQL relationship query: ~100ms
- Coherent subgraph extraction: ~200ms
- LLM generation: ~2000ms (depends on provider)

**Total**: ~2.5 seconds per query

### Scalability

- **Small datasets** (<1k entities): All processing in memory
- **Medium datasets** (1k-10k entities): Use caching and batching
- **Large datasets** (>10k entities): Consider distributed SPARQL endpoint

---

## Troubleshooting

### Common Issues

**1. Missing Ontology Classes/Properties**

**Symptom**: Warning messages about missing classes
**Solution**: See `ontology_validation_report.txt` for detailed instructions

**2. SPARQL Timeout**

**Symptom**: `SPARQLWrapper.QueryBadFormed` or timeout errors
**Solution**: Reduce `ENTITY_CONTEXT_DEPTH` or `MAX_ADJACENCY_HOPS`

**3. Rate Limit Exceeded**

**Symptom**: `rate_limit_exceeded` errors during processing
**Solution**: Adjust `TOKENS_PER_MINUTE_LIMIT` or increase batch sleep time

**4. Out of Memory**

**Symptom**: Process killed during vector store building
**Solution**: Reduce `DEFAULT_BATCH_SIZE` or process in chunks

**5. Poor Retrieval Quality**

**Symptom**: Irrelevant documents retrieved
**Solution**: Adjust `VECTOR_PAGERANK_ALPHA` and `RELEVANCE_CONNECTIVITY_ALPHA`
