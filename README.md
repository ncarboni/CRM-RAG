# RAG Architecture for CIDOC-CRM

Graph-based RAG (Retrieval-Augmented Generation) system for querying CIDOC-CRM RDF data. Supports multiple datasets with lazy loading and per-dataset caching.

## Demo

<video src="docs/demo.mp4" width="320" height="240" controls></video>

## Repository Structure

```
CRM_RAG/
├── config/              Configuration files
│   ├── .env.openai.example
│   ├── .env.claude.example
│   ├── .env.r1.example
│   ├── .env.ollama.example
│   ├── .env.local.example    # Local embeddings (fast, no API)
│   ├── .env.secrets.example
│   ├── datasets.yaml         # Multi-dataset configuration
│   ├── event_classes.json    # CIDOC-CRM event classes for graph traversal
│   ├── interface.yaml        # Chat interface customization
│   └── README.md             # Configuration guide
├── data/                All data files
│   ├── ontologies/      CIDOC-CRM, VIR, CRMdig ontology files
│   ├── labels/          Extracted labels (shared across datasets)
│   ├── cache/           Per-dataset caches (auto-generated)
│   │   ├── asinou/          # Dataset-specific cache
│   │   │   ├── document_graph.pkl
│   │   │   ├── vector_index/
│   │   │   └── embeddings/  # Embedding cache for resumability
│   │   └── museum/          # Another dataset cache
│   │       ├── document_graph.pkl
│   │       ├── vector_index/
│   │       └── embeddings/
│   └── documents/       Per-dataset entity documents (auto-generated)
│       ├── asinou/entity_documents/
│       └── museum/entity_documents/
├── docs/                Documentation
│   ├── ARCHITECTURE.md
│   ├── LOCAL_EMBEDDINGS.md   # Local embeddings guide
│   ├── CLUSTER_EMBEDDINGS.md # GPU cluster processing guide
│   └── REORGANIZATION_PLAN.md
├── scripts/             Utility scripts
│   └── extract_ontology_labels.py
├── logs/                Application logs
├── static/              Web interface CSS and JavaScript
├── templates/           Web interface HTML templates
├── main.py              Flask application entry point
├── universal_rag_system.py  Core RAG logic
├── graph_document_store.py  Graph-based document storage
├── llm_providers.py     LLM abstraction (OpenAI, Claude, local embeddings)
├── embedding_cache.py   Embedding cache for resumability
├── dataset_manager.py   Multi-dataset management
└── config_loader.py     Configuration loading
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create configuration files from templates in the `config/` directory:

```bash
# Copy the secrets template (for API keys)
cp config/.env.secrets.example config/.env.secrets

# Edit config/.env.secrets and add your actual API keys
OPENAI_API_KEY=your_actual_openai_key_here
ANTHROPIC_API_KEY=your_actual_anthropic_key_here
R1_API_KEY=your_actual_r1_key_here

# Copy the provider configuration you want to use
cp config/.env.openai.example config/.env.openai
# OR
cp config/.env.claude.example config/.env.claude
# OR
cp config/.env.r1.example config/.env.r1
# OR
cp config/.env.ollama.example config/.env.ollama
```

### 3. Configure Datasets

Create `config/datasets.yaml` to define your SPARQL datasets:

```yaml
# config/datasets.yaml
default_dataset: asinou  # Which dataset to load by default

datasets:
  asinou:
    name: asinou
    display_name: "Asinou Church"
    description: "Asinou church dataset with frescoes and iconography"
    endpoint: "http://localhost:3030/asinou/sparql"
    # Optional: use local embeddings for this small dataset
    embedding:
      provider: local
      model: BAAI/bge-m3
    interface:  # Optional: override interface.yaml settings
      page_title: "Asinou Dataset Chat"
      welcome_message: "Ask me about Asinou church..."
      example_questions:
        - "Where is Panagia Phorbiottisa located?"
        - "What frescoes are in the church?"

  museum:
    name: museum
    display_name: "Museum Collection"
    description: "Museum artworks, artists, and exhibitions"
    endpoint: "http://localhost:3030/museum/sparql"
    # Optional: use OpenAI embeddings for this dataset (inherits from .env if not specified)
    embedding:
      provider: openai
    interface:
      page_title: "Museum Collection Chat"
      example_questions:
        - "Which pieces from Swiss Artists are in the museum?"
```

Each dataset gets its own cache directory under `data/cache/<dataset_id>/`.

**Per-dataset embedding configuration:**

You can configure different embedding providers for each dataset:

```yaml
datasets:
  small_dataset:
    endpoint: "http://localhost:3030/small/sparql"
    embedding:
      provider: local              # Use local embeddings (fast)
      model: BAAI/bge-m3
      batch_size: 64

  large_dataset:
    endpoint: "http://localhost:3030/large/sparql"
    embedding:
      provider: openai             # Use OpenAI embeddings
      # model inherited from .env config
```

Available embedding options per dataset:
- `provider`: `local`, `sentence-transformers`, `openai`, `ollama`
- `model`: Embedding model name
- `batch_size`: Batch size for local embeddings (default: 64)
- `device`: `auto`, `cuda`, `mps`, `cpu`
- `use_cache`: `true` or `false`

### 4. Extract Ontology Labels

Extract English labels from ontology files (required on first run):

```bash
python scripts/extract_ontology_labels.py
```

This creates label files in `data/labels/` used by the RAG system.

### 5. Configure Event Classes (Optional)

The system uses event-aware graph traversal to build entity documents. In CIDOC-CRM, events (activities, productions, etc.) are the "glue" connecting things, actors, places, and times. Multi-hop context only traverses THROUGH events, preventing unrelated entities from polluting documents.

Event classes are configured in [`config/event_classes.json`](config/event_classes.json):

```json
{
  "_comment": "Add or remove event class URIs as needed",

  "cidoc_crm": [
    "http://www.cidoc-crm.org/cidoc-crm/E5_Event",
    "http://www.cidoc-crm.org/cidoc-crm/E12_Production",
    ...
  ],
  "crmdig": [...],
  "crmsci": [...],
  "vir": [...],
  "crminf": [...]
}
```

To customize:
- Add URIs to existing categories or create new ones
- Keys starting with `_` are ignored (use for comments)
- Changes take effect on next restart

### 6. Customize Chat Interface (Optional)

Customize the chatbot title, welcome message, and example questions by editing `config/interface.yaml`:

```yaml
page_title: "Your Dataset Chat"
header_title: "Your Custom Chatbot"
welcome_message: "Hello! Ask me about your dataset..."
example_questions:
  - "Your first example question?"
  - "Your second example question?"
```

See `config/README.md` for detailed customization options.

### 7. Start Your SPARQL Endpoint

Ensure your SPARQL server is running with your CIDOC-CRM dataset loaded at the configured endpoint.

## Usage

### Basic Usage

```bash
# Run with OpenAI
python main.py --env .env.openai

# Run with Claude
python main.py --env .env.claude

# Run with R1
python main.py --env .env.r1

# Run with Ollama (no API key needed)
python main.py --env .env.ollama

# Force rebuild of document graph and vector store
python main.py --env .env.openai --rebuild
```

Access the chat interface at `http://localhost:5001`

### Local Embeddings (Recommended for Large Datasets)

For datasets with 5,000+ entities, use local embeddings to avoid API rate limits and reduce processing time from days to minutes.

```bash
# Set up local embeddings config
cp config/.env.local.example config/.env.local

# Process a dataset with local embeddings
python main.py --env .env.local --dataset asinou --rebuild --process-only
```

| Method | 50,000 entities | Cost |
|--------|-----------------|------|
| OpenAI API | 2-4 days | ~$10-20 |
| Local (CPU) | 1-2 hours | Free |
| Local (GPU) | 10-20 minutes | Free |

**See [docs/LOCAL_EMBEDDINGS.md](docs/LOCAL_EMBEDDINGS.md) for complete documentation** including:
- Configuration options
- Per-dataset embedding providers
- Model recommendations
- Hardware acceleration (GPU/CPU)
- Troubleshooting

### Multi-Dataset Mode

When `config/datasets.yaml` is configured, the chat interface displays a dataset selector dropdown. Select a dataset to:
- Load its cached embeddings (or build them on first access)
- Update the interface with dataset-specific titles and example questions
- Query only that dataset's knowledge graph

Datasets are lazily loaded - they initialize only when first selected, saving memory and startup time.

### Clearing Cache

To rebuild a specific dataset's cache:

```bash
# Clear cache for a specific dataset and rebuild
rm -rf data/cache/asinou/
rm -rf data/documents/asinou/
python main.py --env .env.openai --rebuild
```

For single-dataset mode (legacy):
```bash
rm -rf data/cache/document_graph.pkl data/cache/vector_index/
rm -rf data/documents/entity_documents/
python main.py --env .env.openai --rebuild
```

### CLI Reference

| Flag | Description |
|------|-------------|
| `--env <file>` | Path to environment config file (e.g., `.env.openai`) |
| `--dataset <id>` | Dataset ID to process (from datasets.yaml) |
| `--process-only` | Process dataset and exit without starting web server |
| `--rebuild` | Force rebuild of document graph and vector store |
| `--embedding-provider <name>` | Embedding provider: `openai`, `local`, `sentence-transformers`, `ollama` |
| `--embedding-model <model>` | Embedding model name (e.g., `BAAI/bge-m3`) |
| `--no-embedding-cache` | Disable embedding cache (force re-embedding) |
| `--generate-docs-only` | Generate documents from SPARQL without embedding (for cluster workflow) |
| `--embed-from-docs` | Generate embeddings from existing documents (no SPARQL needed) |

### Processing Specific Datasets

Process a single dataset from the command line:

```bash
# Process dataset with local embeddings (recommended)
python main.py --env .env.local --dataset asinou --rebuild --process-only

# Process dataset with OpenAI embeddings
python main.py --env .env.openai --dataset museum --rebuild --process-only

# Process and start web server
python main.py --env .env.local --dataset asinou --rebuild
```

See [docs/LOCAL_EMBEDDINGS.md](docs/LOCAL_EMBEDDINGS.md) for detailed examples of processing multiple datasets with different embedding providers.

## API Endpoints

### Dataset Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/datasets` | GET | List all available datasets with their status |
| `/api/datasets/<id>/select` | POST | Initialize and select a dataset, returns interface config |

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send a question. Body: `{"question": "...", "dataset_id": "..."}` |
| `/api/info` | GET | Get system information (LLM provider, model, etc.) |
| `/api/entity/<uri>/wikidata` | GET | Get Wikidata info for an entity |

**Note:** In multi-dataset mode, `dataset_id` is required for `/api/chat`.

## Architecture

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Key Components

- **DatasetManager** (`dataset_manager.py`): Manages multiple RAG system instances with lazy loading
- **UniversalRagSystem** (`universal_rag_system.py`): Core RAG logic with CIDOC-CRM aware retrieval
- **GraphDocumentStore** (`graph_document_store.py`): Graph-based document storage with FAISS vectors
- **LLM Providers** (`llm_providers.py`): Abstraction layer for OpenAI, Anthropic, R1, Ollama, and local embeddings (sentence-transformers)
- **EmbeddingCache** (`embedding_cache.py`): Disk-based embedding cache for resumable processing

### Retrieval Pipeline

1. **Vector Search**: FAISS similarity search for initial candidates
2. **CIDOC-CRM Scoring**: Relationship-aware scoring based on ontology semantics
3. **PageRank**: Graph-based importance scoring
4. **Coherent Subgraph Extraction**: Selects connected documents balancing relevance and connectivity
