# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) research project for testing different approaches to querying CIDOC-CRM RDF data about Byzantine art. The project contains two main implementations and legacy code for comparison:

1. **CRM+NETWORK+Extraction**: Advanced RAG system using graph-based document retrieval with CIDOC-CRM relationship weights and network extraction
2. **Simple_RAG**: Baseline implementation using traditional vector-based retrieval without graph processing
3. **legacy/**: Archived experimental implementations including GNN and network metrics approaches

## Architecture

### Core Components

Both implementations share these architectural patterns:

- **main.py / main_simple.py**: Flask web application entry point with CLI argument handling
- **universal_rag_system.py / universal_rag_system_simple.py**: Core RAG logic and SPARQL query processing
- **llm_providers.py**: Abstraction layer supporting multiple LLM providers (OpenAI, Anthropic, R1, Ollama)
- **config_loader.py**: Environment-based configuration loader
- **graph_document_store.py** (CRM+NETWORK+Extraction only): Graph-based document storage with neighbor relationships

### Key Architectural Differences

**CRM+NETWORK+Extraction** uses a graph-based approach:
- Creates a document graph with weighted edges based on CIDOC-CRM relationship semantics
- Processes RDF triples bidirectionally to extract coherent subgraphs
- Uses `GraphDocumentStore` to maintain document relationships
- Stores graph structure in `document_graph.pkl` and vector index in `vector_index/`
- Implements `process_cidoc_relationship()` to convert RDF predicates to natural language
- Traverses entity context with `get_entity_context()` supporting bidirectional graph exploration

**Simple_RAG** uses traditional vector retrieval:
- Flat document structure with basic FAISS vector search
- Stores documents in `documents_simple.pkl` and vector index in `vector_index_simple/`
- No graph processing or relationship weighting
- Serves as baseline for performance comparison

### LLM Provider System

The `llm_providers.py` module implements a `BaseLLMProvider` abstract class with concrete implementations:
- `OpenAIProvider`: GPT models with OpenAI embeddings
- `AnthropicProvider`: Claude models (uses OpenAI for embeddings)
- `R1Provider`: R1 models
- `OllamaProvider`: Local Ollama models

All providers expose two methods:
- `generate(system_prompt, user_prompt)`: Generate LLM responses
- `get_embeddings(text)`: Generate text embeddings

### Data Flow

1. System connects to Apache Jena Fuseki SPARQL endpoint (typically `http://localhost:3030/asinou/sparql`)
2. On first run, extracts entities from RDF data and builds document representations
3. Creates embeddings and stores both documents and vector index to disk
4. On subsequent runs, loads from cache unless `--rebuild` flag is used
5. Flask web interface serves chat UI at configured port (5001 for main, 5002 for simple)

## Running the Applications

### CRM+NETWORK+Extraction (Advanced System)

```bash
cd CRM+NETWORK+Extraction

# Run with different LLM providers
python main.py --env .env.openai
python main.py --env .env.claude
python main.py --env .env.r1
python main.py --env .env.ollama

# Force rebuild of document graph and vector store
python main.py --env .env.openai --rebuild
```

Default port: 5001

### Simple_RAG (Baseline System)

```bash
cd Simple_RAG

# Currently only configured for OpenAI
python main_simple.py --env .env.openai

# Force rebuild
python main_simple.py --env .env.openai --rebuild
```

Default port: 5002

## Environment Configuration

Each implementation directory contains `.env.*` files for different LLM providers. Required environment variables:

- `FUSEKI_ENDPOINT`: SPARQL endpoint URL
- `LLM_PROVIDER`: Provider name (openai, anthropic, r1, ollama)
- `[PROVIDER]_API_KEY`: API key for the provider (except Ollama)
- `[PROVIDER]_MODEL`: Model name to use
- `TEMPERATURE`: LLM temperature (typically 0.7)
- `PORT`: Flask server port

For OpenAI embeddings are specified via `OPENAI_EMBEDDING_MODEL` (typically `text-embedding-3-small`). Anthropic uses OpenAI embeddings as Claude doesn't provide embedding models.

## Important Implementation Details

### CIDOC-CRM Relationships

The CRM+NETWORK+Extraction system maps CIDOC-CRM predicates to natural language in `process_cidoc_relationship()`. Categories include:
- Spatial: P89_falls_within, P55_has_current_location
- Temporal: P4_has_time-span, P117_occurs_during
- Physical: P46_is_composed_of, P128_carries
- Conceptual: P2_has_type, P129_is_about
- Production: P108i_was_produced_by, P94i_was_created_by
- VIR ontology: K1i_is_denoted_by, K24_portray

### Cached Data

The systems cache processed data to avoid repeated SPARQL queries and embedding calls:
- Document structures are pickled to `.pkl` files
- FAISS vector indices are stored in `vector_index/` or `vector_index_simple/`
- Use `--rebuild` flag to force regeneration (required after RDF data updates)

### Rate Limiting

The main system warns about rate limits when processing large datasets (>500 entities). On first initialization with large datasets, the system will prompt for confirmation unless `--rebuild` is explicitly used.

## Development Notes

- All systems use Flask with `debug=False` for production-like behavior
- Logging configured with rotating file handlers (`app.log`, `app_simple.log`)
- Static files and templates are in `static/` and `templates/` directories
- The chat interface provides entity Wikidata lookup via `/api/entity/<uri>/wikidata`
- System information available at `/api/info` endpoint
