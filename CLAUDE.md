# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a graph-based RAG (Retrieval-Augmented Generation) system for querying CIDOC-CRM RDF data about Byzantine art. The system uses coherent subgraph extraction with weighted CIDOC-CRM relationships to enhance document retrieval, providing semantically-rich context for LLM-based question answering.

**Key Innovation**: Unlike traditional vector-based RAG, this system creates a document graph where nodes represent entities and edges represent CIDOC-CRM relationships with semantic weights. This allows the system to retrieve not just similar documents, but entire coherent subgraphs of related information.

## Running the Application

```bash
# Run with different LLM providers
python main.py --env .env.openai
python main.py --env .env.claude
python main.py --env .env.r1
python main.py --env .env.ollama

# Force rebuild of document graph and vector store (required after RDF data changes)
python main.py --env .env.openai --rebuild
```

The application starts a Flask web server on port 5001 (configurable via PORT in .env files). Access the chat interface at `http://localhost:5001`.

## Prerequisites

- Apache Jena Fuseki SPARQL endpoint running (default: `http://localhost:3030/asinou/sparql`)
- Python dependencies: Flask, SPARQLWrapper, langchain, langchain-community, langchain-openai, FAISS, networkx, tqdm
- API keys for chosen LLM provider (except Ollama)

## Architecture

### Core Components

**main.py** (main.py:1)
- Flask application entry point with CLI argument parsing
- Routes: `/` (chat interface), `/api/chat` (query processing), `/api/entity/<uri>/wikidata` (entity lookup), `/api/info` (system info)
- Handles initialization logic including cache detection and rate limit warnings (main.py:111-141)

**universal_rag_system.py** (universal_rag_system.py:25)
- `UniversalRagSystem` class: core RAG orchestration
- Key methods:
  - `initialize()`: loads or builds document graph and vector index
  - `answer_question(question)`: retrieves relevant documents and generates LLM response
  - `process_cidoc_relationship()` (universal_rag_system.py:207): converts RDF predicates to natural language
  - `get_entity_context(entity_uri, depth)` (universal_rag_system.py:261): traverses graph bidirectionally to extract coherent subgraphs

**graph_document_store.py** (graph_document_store.py:33)
- `GraphDocumentStore` class: manages document graph structure
- `GraphDocument` dataclass (graph_document_store.py:16): stores document text, embeddings, metadata, and neighbor connections
- Maintains both graph structure (for context expansion) and FAISS vector index (for initial retrieval)
- Persistence: saves to `document_graph.pkl` and `vector_index/`

**llm_providers.py** (llm_providers.py:12)
- `BaseLLMProvider` abstract class with two methods: `generate()` and `get_embeddings()`
- Implementations: `OpenAIProvider`, `AnthropicProvider`, `R1Provider`, `OllamaProvider`
- Note: Anthropic uses OpenAI for embeddings since Claude doesn't provide embedding models

**config_loader.py** (config_loader.py:13)
- `ConfigLoader.load_config(env_file)`: loads provider-specific configuration from .env files
- Supports openai, anthropic, r1, ollama providers

### Data Flow

1. **Initialization** (first run or with `--rebuild`):
   - Query SPARQL endpoint for all entities
   - For each entity, extract RDF triples and convert to natural language using `process_cidoc_relationship()`
   - Create `GraphDocument` nodes with embeddings
   - Build weighted edges between documents based on RDF relationships
   - Save graph to `document_graph.pkl` and vector index to `vector_index/`

2. **Query Processing**:
   - User submits question via `/api/chat`
   - System embeds question and retrieves top-k similar documents from FAISS index
   - For each retrieved document, traverse graph to gather neighbor context using `get_entity_context()`
   - Aggregate retrieved subgraph into context
   - Generate LLM response using provider's `generate()` method

3. **Caching**:
   - On subsequent runs, loads from `document_graph.pkl` and `vector_index/` instead of querying SPARQL
   - Use `--rebuild` to force regeneration after RDF data updates

### CIDOC-CRM Relationship Processing

The system maps CIDOC-CRM predicates to natural language in `process_cidoc_relationship()` (universal_rag_system.py:207-259):

- **Spatial**: P89_falls_within, P55_has_current_location, P53_has_former_or_current_location
- **Temporal**: P4_has_time-span, P117_occurs_during, P114_is_equal_in_time_to
- **Physical**: P46_is_composed_of, P56_bears_feature, P128_carries
- **Conceptual**: P2_has_type, P1_is_identified_by, P67_refers_to, P129_is_about, P138_represents
- **Production**: P108i_was_produced_by, P94i_was_created_by
- **VIR ontology**: K1i_is_denoted_by, K17_has_attribute, K24_portray

Each predicate is converted to a natural language statement (e.g., "P89_falls_within" → "{subject} is located within {object}").

## Environment Configuration

Each `.env.*` file configures a different LLM provider. Required variables:

```bash
# SPARQL endpoint
FUSEKI_ENDPOINT=http://localhost:3030/asinou/sparql

# LLM provider (openai, anthropic, r1, ollama)
LLM_PROVIDER=openai

# Provider-specific settings
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MAX_TOKENS=4096

# Common settings
TEMPERATURE=0.7
PORT=5001
```

**Provider-specific notes**:
- **OpenAI**: Requires `OPENAI_API_KEY`, uses `text-embedding-3-small` for embeddings
- **Anthropic**: Requires both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` (for embeddings)
- **R1**: Requires `R1_API_KEY`, uses `e5-small-v2` for embeddings
- **Ollama**: No API key needed, requires local Ollama server at `OLLAMA_HOST` (default: `http://localhost:11434`)

## Important Implementation Details

### Graph-Based Retrieval

Unlike traditional RAG systems that retrieve isolated documents, this system:
1. Uses FAISS for initial similarity-based retrieval
2. Expands retrieved documents by traversing graph edges to gather related entities
3. Uses `get_entity_context(entity_uri, depth=2)` to explore bidirectional relationships
4. Weights edges based on CIDOC-CRM relationship types to prioritize semantically important connections

### Rate Limiting and Large Datasets

When processing >500 entities on first initialization (main.py:120-135):
- System warns about potential API rate limits
- Prompts for confirmation unless `--rebuild` flag is used
- Consider adjusting batch sizes and sleep times in embedding calls for large datasets

### Caching and Persistence

- **Document graph**: `document_graph.pkl` (pickled dictionary of GraphDocument objects)
- **Vector index**: `vector_index/index.faiss` (FAISS index file)
- Cache is automatically loaded if both files exist
- Use `--rebuild` to force regeneration after:
  - RDF data updates
  - Changes to relationship processing logic
  - Switching embedding models

### Web Interface

- **Templates**: `templates/chat.html` (main interface), `templates/base.html` (layout)
- **Static files**: `static/css/`, `static/js/`
- **API endpoints**:
  - `POST /api/chat`: Submit questions, returns `{"answer": "...", "sources": [...]}`
  - `GET /api/entity/<uri>/wikidata`: Get Wikidata info for entity
  - `GET /api/info`: System configuration (provider, model, embedding model)

### Legacy Code

The `legacy_approach/` directory contains archived experimental implementations:
- `ontology_processor.py`: Earlier approaches using RDFLib and document processing
- `rag_system.py`: Previous RAG implementations
- Not used by current system, kept for reference/comparison

## Logging

- Log file: `app.log` (rotating, max 10MB, 5 backups)
- Console logging enabled
- Key logged events:
  - SPARQL connection tests
  - Document graph building/loading
  - Query processing
  - Provider initialization
