# RAG Architecture for CIDOC-CRM

Graph-based RAG (Retrieval-Augmented Generation) system for querying CIDOC-CRM RDF data. Combines FR-guided document generation, an igraph knowledge graph, and a multi-stage retrieval pipeline (FAISS + BM25 + PPR). Supports multiple datasets with lazy loading and per-dataset caching.

## Demo

https://github.com/user-attachments/assets/692a69ff-c25f-40b1-8a36-1a660e810060



## Repository Structure

```
CRM_RAG/
├── main.py                      # Thin entry point → crm_rag.app.main()
├── pyproject.toml               # Dependencies
├── src/crm_rag/
│   ├── __init__.py              # PROJECT_ROOT constant, pickle compat alias
│   ├── app.py                   # Flask routes, security, dataset init
│   ├── rag_system.py            # Core orchestrator (~3630 lines)
│   ├── document_store.py        # GraphDocument + FAISS/BM25 vector search
│   ├── knowledge_graph.py       # igraph wrapper (triples, PPR, PageRank, stats)
│   ├── llm_providers.py         # OpenAI / Anthropic / R1 / Ollama abstraction
│   ├── config_loader.py         # .env + .env.secrets + datasets.yaml loader
│   ├── dataset_manager.py       # Multi-dataset lazy loading
│   ├── fr_traversal.py          # FR formatting + FC classification
│   ├── fr_materializer.py       # igraph-native FR walker
│   ├── fundamental_relationships.py # 98 FR definitions, 2325 expanded paths
│   ├── document_formatter.py    # Predicate/class formatting, relationship weights
│   ├── sparql_helpers.py        # BatchSparqlClient
│   └── embedding_cache.py       # Disk-based embedding cache
├── config/
│   ├── .env.openai.example      # LLM config templates
│   ├── .env.secrets.example     # API keys template
│   ├── datasets.yaml            # SPARQL endpoints + per-dataset config
│   ├── interface.yaml           # Chat UI customization
│   ├── prompts.yaml             # System + query-analysis prompts
│   ├── event_classes.json       # CRM event class URIs
│   ├── fc_class_mapping.json    # 168 CRM classes → 6 FCs
│   └── relationship_weights.json # Predicate → weight (0.0-1.0)
├── data/
│   ├── ontologies/              # CRM + VIR + CRMdig RDF files
│   ├── labels/                  # Auto-generated label JSON files
│   ├── cache/<dataset>/         # document_graph.pkl, knowledge_graph.pkl, indices
│   └── documents/<dataset>/     # Entity documents (markdown)
├── scripts/
│   ├── extract_ontology_labels.py      # Ontology → JSON labels
│   ├── evaluate_pipeline.py            # Pipeline evaluation → reports/
│   └── build_mah_reference_answers.py  # MAH evaluation ground truth builder
├── templates/                   # Flask HTML (base, chat, graph)
├── static/                      # CSS + JS (base, chat, graph)
├── docs/                        # Architecture paper, technical report
└── reports/                     # Eval outputs: <dataset>_<timestamp>.json
```

## Setup

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

### 2. Configure API Keys

```bash
# Copy the secrets template (for API keys)
cp config/.env.secrets.example config/.env.secrets

# Edit config/.env.secrets and add your actual API keys
OPENAI_API_KEY=your_actual_openai_key_here
ANTHROPIC_API_KEY=your_actual_anthropic_key_here

# Copy the provider configuration you want to use
cp config/.env.openai.example config/.env.openai
# OR
cp config/.env.claude.example config/.env.claude
# OR
cp config/.env.ollama.example config/.env.ollama
```

### 3. Configure Datasets

Create `config/datasets.yaml` to define your SPARQL datasets:

```yaml
default_dataset: asinou

datasets:
  asinou:
    name: asinou
    display_name: "Asinou Church"
    description: "Asinou church dataset with frescoes and iconography"
    endpoint: "http://localhost:3030/asinou/sparql"
    embedding:
      provider: local
      model: BAAI/bge-m3
    interface:
      page_title: "Asinou Dataset Chat"
      welcome_message: "Ask me about Asinou church..."
      example_questions:
        - "Where is Panagia Phorbiottisa located?"
        - "What frescoes are in the church?"

  mah:
    name: mah
    display_name: "Museum Collection"
    description: "Museum artworks, artists, and exhibitions"
    endpoint: "http://localhost:3030/mah/sparql"
    embedding:
      provider: openai
    interface:
      page_title: "Museum Collection Chat"
      example_questions:
        - "Which pieces from Swiss Artists are in the museum?"
```

Each dataset gets its own cache directory under `data/cache/<dataset_id>/`.

### 4. Extract Ontology Labels

```bash
uv run python scripts/extract_ontology_labels.py
```

This creates label files in `data/labels/` used by the RAG system.

### 5. Start Your SPARQL Endpoint

Ensure your SPARQL server is running with your CIDOC-CRM dataset loaded at the configured endpoint.

## Usage

### Basic Usage

```bash
# Run with OpenAI
uv run python main.py --env .env.openai

# Run with local embeddings (recommended for large datasets)
uv run python main.py --env .env.local --dataset asinou

# Force rebuild of document graph and vector store
uv run python main.py --env .env.openai --rebuild

# CLI mode: single question
uv run python main.py --env .env.local --dataset asinou --question "What frescoes are in Asinou?"

# CLI mode: interactive
uv run python main.py --env .env.local --dataset asinou --question
```

Access the chat interface at `http://localhost:5001`

### Local Embeddings (Recommended for Large Datasets)

For datasets with 5,000+ entities, use local embeddings to avoid API rate limits:

```bash
cp config/.env.local.example config/.env.local
uv run python main.py --env .env.local --dataset asinou --rebuild
```

| Method | 50,000 entities | Cost |
|--------|-----------------|------|
| OpenAI API | 2-4 days | ~$10-20 |
| Local (CPU) | 1-2 hours | Free |
| Local (GPU) | 10-20 minutes | Free |

### Multi-Dataset Mode

When `config/datasets.yaml` is configured, the chat interface displays a dataset selector dropdown. Datasets are lazily loaded on first selection.

### Clearing Cache

```bash
# Clear cache for a specific dataset and rebuild
rm -rf data/cache/asinou/ data/documents/asinou/
uv run python main.py --env .env.local --dataset asinou --rebuild
```

### CLI Reference

| Flag | Description |
|------|-------------|
| `--env <file>` | Path to environment config file (e.g., `.env.openai`) |
| `--dataset <id>` | Dataset ID to process (from datasets.yaml) |
| `--rebuild` | Force rebuild of document graph and vector store |
| `--embedding-provider <name>` | Embedding provider: `openai`, `local`, `sentence-transformers`, `ollama` |
| `--embedding-model <model>` | Embedding model name (e.g., `BAAI/bge-m3`) |
| `--no-embedding-cache` | Disable embedding cache (force re-embedding) |
| `--question [QUESTION]` | CLI mode: pass a question or omit for interactive |
| `--debug` | Enable debug logging |

### Evaluation

```bash
uv run python scripts/evaluate_pipeline.py --dataset asinou
```

Results are written to `reports/<dataset>_<timestamp>.json`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/`, `/chat` | Chat interface |
| GET | `/graph` | Cosmograph graph visualization |
| GET | `/api/datasets` | List all available datasets with status |
| POST | `/api/datasets/<id>/select` | Initialize and select a dataset, returns interface config |
| POST | `/api/chat` | Send a question. Body: `{"question": "...", "dataset_id": "..."}` |
| GET | `/api/info` | System info (LLM provider, model) |
| GET | `/api/entity/<uri>/wikidata` | Wikidata entity info |
| GET | `/api/graph/data` | Knowledge graph data for visualization |
| GET | `/api/datasets/<id>/top-entities` | Top PageRank entities for dataset |

## Architecture

### Key Components

- **DatasetManager** (`src/crm_rag/dataset_manager.py`): Manages multiple RAG system instances with lazy loading
- **UniversalRagSystem** (`src/crm_rag/rag_system.py`): Core RAG orchestrator — document generation, retrieval, answer generation
- **GraphDocumentStore** (`src/crm_rag/document_store.py`): FAISS + BM25 vector search with FC type index
- **KnowledgeGraph** (`src/crm_rag/knowledge_graph.py`): igraph wrapper for RDF + FR edges, PPR, PageRank
- **LLM Providers** (`src/crm_rag/llm_providers.py`): Abstraction layer for OpenAI, Anthropic, R1, Ollama, and local embeddings
- **FR Materializer** (`src/crm_rag/fr_materializer.py`): igraph-native Fundamental Relationship walker
- **EmbeddingCache** (`src/crm_rag/embedding_cache.py`): Disk-based embedding cache for resumable processing

### Build Pipeline

1. **Phase 1**: Load RDF triples into igraph (chunked SPARQL)
2. **Phase 2**: Materialize FR edges, identity-based event contraction, satellite identification
3. **Phase 2.5**: Pre-compute enrichments and time-span date caches
4. **Phase 3**: Generate entity documents from FR edges + direct predicates, embed, save

### Retrieval Pipeline

1. **Query analysis**: LLM classifies query → SPECIFIC/ENUMERATION/AGGREGATION → dynamic k
2. **Multi-channel retrieval**: FAISS + BM25 + PPR (Personalized PageRank) → RRF fusion
3. **Type-filtered channel**: FC-aware retrieval for type-specific queries
4. **Coherent subgraph extraction**: Greedy selection balancing relevance (α=0.7) and PPR connectivity (0.3), with MMR diversity penalty
5. **Answer generation**: Context assembly with triples enrichment, prompt tuning by query type, LLM call
