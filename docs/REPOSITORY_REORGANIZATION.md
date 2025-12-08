# Repository Reorganization Proposal

## Current Issues

1. **Everything in root directory** - Hard to navigate and find files
2. **Mixed concerns** - Config files, code, data, and docs all together
3. **Generated files in root** - Makes it unclear what's source vs. generated
4. **No clear structure** - Difficult for new developers to understand
5. **No tests directory** - Testing infrastructure not organized
6. **Multiple .env files** scattered in root

---

## Proposed New Structure

```
CRM_RAG/
│
├── README.md                          # Project overview & quick start
├── ARCHITECTURE.md                    # Architecture documentation (move to docs/)
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation (NEW)
├── .gitignore                         # Git ignore rules
│
├── config/                            # Configuration files (NEW)
│   ├── .env.example                   # Combined example for all providers
│   ├── .env.secrets.example           # Secrets template
│   ├── providers/                     # Provider-specific configs
│   │   ├── openai.env.example
│   │   ├── claude.env.example
│   │   ├── r1.env.example
│   │   └── ollama.env.example
│   └── README.md                      # Config documentation
│
├── src/                               # Source code (NEW)
│   └── crm_rag/                       # Main package
│       ├── __init__.py
│       ├── app.py                     # Flask application (renamed from main.py)
│       ├── rag_system.py              # Core RAG (renamed from universal_rag_system.py)
│       ├── config.py                  # Config loader (renamed from config_loader.py)
│       │
│       ├── core/                      # Core RAG components
│       │   ├── __init__.py
│       │   ├── graph_store.py         # Graph document store
│       │   ├── retrieval.py           # Retrieval algorithms
│       │   └── coherence.py           # Coherent subgraph extraction
│       │
│       ├── ontology/                  # Ontology processing
│       │   ├── __init__.py
│       │   ├── extractor.py           # Label extraction (from extract_ontology_labels.py)
│       │   └── validator.py           # Ontology validation
│       │
│       ├── llm/                       # LLM providers
│       │   ├── __init__.py
│       │   ├── base.py                # Base provider class
│       │   ├── openai.py              # OpenAI provider
│       │   ├── anthropic.py           # Anthropic provider
│       │   ├── r1.py                  # R1 provider
│       │   └── ollama.py              # Ollama provider
│       │
│       ├── web/                       # Web interface
│       │   ├── __init__.py
│       │   ├── routes.py              # Flask routes
│       │   ├── security.py            # Security utilities (path validation, etc.)
│       │   ├── static/                # Static files
│       │   │   ├── css/
│       │   │   │   ├── base.css
│       │   │   │   └── chat.css
│       │   │   └── js/
│       │   │       └── chat.js
│       │   └── templates/             # HTML templates
│       │       ├── base.html
│       │       └── chat.html
│       │
│       └── utils/                     # Utilities
│           ├── __init__.py
│           ├── sparql.py              # SPARQL utilities
│           └── wikidata.py            # Wikidata integration
│
├── data/                              # Data directory (NEW)
│   ├── ontologies/                    # Ontology files (renamed from ontology/)
│   │   ├── README.md                  # Ontology documentation
│   │   ├── CIDOC_CRM_v7.1.3.rdf
│   │   ├── CRMdig_v3.2.1.rdf
│   │   ├── frbroo.rdf
│   │   ├── skos_2009-08-18.n3
│   │   └── vir.ttl
│   │
│   ├── labels/                        # Extracted labels (NEW)
│   │   ├── property_labels.json       # (generated, gitignored)
│   │   ├── class_labels.json          # (generated, gitignored)
│   │   └── ontology_classes.json      # (generated, gitignored)
│   │
│   ├── cache/                         # Cached data (NEW)
│   │   ├── document_graph.pkl         # (generated, gitignored)
│   │   └── vector_index/              # (generated, gitignored)
│   │
│   └── documents/                     # Entity documents (NEW)
│       └── entity_documents/          # (generated, gitignored)
│
├── docs/                              # Documentation (NEW)
│   ├── ARCHITECTURE.md                # Architecture overview (moved)
│   ├── SETUP.md                       # Setup & installation guide
│   ├── CONFIGURATION.md               # Configuration guide
│   ├── API.md                         # API documentation
│   ├── ONTOLOGIES.md                  # Ontology guide & troubleshooting
│   ├── SECURITY.md                    # Security best practices
│   └── images/                        # Architecture diagrams, screenshots
│
├── scripts/                           # Utility scripts (NEW)
│   ├── extract_labels.py              # Extract ontology labels (from extract_ontology_labels.py)
│   ├── rebuild_cache.py               # Rebuild document graph & vector index
│   ├── validate_ontologies.py         # Validate ontology files
│   └── generate_secret_key.py         # Generate Flask secret key
│
├── tests/                             # Tests (NEW)
│   ├── __init__.py
│   ├── test_rag_system.py
│   ├── test_graph_store.py
│   ├── test_ontology_extractor.py
│   ├── test_llm_providers.py
│   ├── test_retrieval.py
│   ├── test_security.py
│   └── fixtures/                      # Test data
│       └── sample_ontology.ttl
│
├── logs/                              # Log files (NEW)
│   └── app.log                        # (gitignored)
│
└── .env                               # Active environment (gitignored, not in repo)
```

---

## Key Improvements

### 1. **Clear Separation of Concerns**

| Directory | Purpose | Gitignored? |
|-----------|---------|-------------|
| `src/crm_rag/` | Source code | No |
| `config/` | Configuration templates | No (except `.env`) |
| `data/` | Ontologies & generated data | Partial (only generated) |
| `docs/` | Documentation | No |
| `scripts/` | Utility scripts | No |
| `tests/` | Test suite | No |
| `logs/` | Application logs | Yes |

### 2. **Package Structure**

Source code is now a proper Python package:
```python
# Install in development mode
pip install -e .

# Use as a package
from crm_rag import RAGSystem
from crm_rag.llm import OpenAIProvider
```

### 3. **Clearer Entry Points**

```bash
# Run web application
python -m crm_rag.app --env config/.env

# Extract ontology labels
python scripts/extract_labels.py

# Rebuild cache
python scripts/rebuild_cache.py --force

# Run tests
pytest tests/
```

### 4. **Better Configuration Management**

**Old (confusing):**
```
.env.openai
.env.openai.example
.env.claude
.env.claude.example
.env.r1
.env.r1.example
.env.ollama
.env.ollama.example
.env.secrets
.env.secrets.example
```

**New (clear):**
```
config/
├── .env.example              # Combined example
├── .env.secrets.example      # Secrets template
└── providers/                # Provider-specific examples
    ├── openai.env.example
    ├── claude.env.example
    ├── r1.env.example
    └── ollama.env.example
```

Users copy one provider example to `.env` in root:
```bash
# For OpenAI
cp config/providers/openai.env.example .env
cp config/.env.secrets.example .env.secrets
# Edit .env and .env.secrets with your keys
```

### 5. **Generated Files Clearly Separated**

**Source (checked into git):**
- `data/ontologies/*.rdf, *.ttl, *.owl`
- `src/crm_rag/*.py`
- `config/*.example`

**Generated (gitignored):**
- `data/labels/*.json`
- `data/cache/*`
- `data/documents/*`
- `logs/*`

### 6. **Better Documentation Organization**

Instead of one huge README:
- `README.md` - Quick start & overview
- `docs/SETUP.md` - Detailed installation
- `docs/CONFIGURATION.md` - Configuration guide
- `docs/ARCHITECTURE.md` - System architecture
- `docs/ONTOLOGIES.md` - Ontology troubleshooting
- `docs/API.md` - API reference
- `docs/SECURITY.md` - Security practices

---

## Migration Plan

### Phase 1: Create New Structure (No Breaking Changes)

```bash
# 1. Create new directories
mkdir -p src/crm_rag/{core,ontology,llm,web,utils}
mkdir -p config/providers
mkdir -p data/{ontologies,labels,cache,documents}
mkdir -p docs/images
mkdir -p scripts
mkdir -p tests/fixtures
mkdir -p logs

# 2. Move ontology files
mv ontology/* data/ontologies/

# 3. Move static/templates into package
mv static src/crm_rag/web/
mv templates src/crm_rag/web/

# 4. Move generated data
mv *_labels.json ontology_classes.json data/labels/
mv document_graph.pkl data/cache/
mv vector_index data/cache/
mv entity_documents data/documents/

# 5. Move config files
mv .env.*.example config/providers/
mv .env.secrets.example config/

# 6. Move logs
mv app.log logs/

# 7. Move docs
mv ARCHITECTURE.md docs/
```

### Phase 2: Refactor Code

1. **Split `llm_providers.py` into separate files:**
   ```
   src/crm_rag/llm/
   ├── base.py          # BaseLLMProvider
   ├── openai.py        # OpenAIProvider
   ├── anthropic.py     # AnthropicProvider
   ├── r1.py            # R1Provider
   └── ollama.py        # OllamaProvider
   ```

2. **Split `universal_rag_system.py` into logical modules:**
   ```
   src/crm_rag/
   ├── rag_system.py         # Main orchestrator
   ├── core/
   │   ├── graph_store.py    # GraphDocumentStore (from graph_document_store.py)
   │   ├── retrieval.py      # Retrieval algorithms
   │   └── coherence.py      # Coherent subgraph extraction
   └── utils/
       ├── sparql.py         # SPARQL utilities
       └── wikidata.py       # Wikidata integration
   ```

3. **Extract web interface from `main.py`:**
   ```
   src/crm_rag/
   ├── app.py               # Flask app initialization
   └── web/
       ├── routes.py        # Flask routes
       └── security.py      # Path validation, etc.
   ```

4. **Rename scripts:**
   ```
   scripts/
   ├── extract_labels.py          # From extract_ontology_labels.py
   ├── rebuild_cache.py           # New utility
   └── generate_secret_key.py     # New utility
   ```

### Phase 3: Update Imports & Paths

1. **Update imports in all files:**
   ```python
   # Old
   from universal_rag_system import UniversalRagSystem

   # New
   from crm_rag import RAGSystem
   ```

2. **Update file paths:**
   ```python
   # Old
   ontology_dir = 'ontology'
   labels_file = 'property_labels.json'

   # New
   ontology_dir = 'data/ontologies'
   labels_file = 'data/labels/property_labels.json'
   ```

3. **Update .gitignore:**
   ```gitignore
   # Environment
   .env
   .env.secrets

   # Generated data
   data/labels/*.json
   data/cache/*
   data/documents/entity_documents/

   # Logs
   logs/*.log

   # Python
   __pycache__/
   *.pyc
   *.egg-info/
   dist/
   build/

   # IDE
   .vscode/
   .idea/
   .claude/

   # OS
   .DS_Store
   ```

### Phase 4: Add Package Files

1. **Create `setup.py`:**
   ```python
   from setuptools import setup, find_packages

   setup(
       name="crm-rag",
       version="1.0.0",
       packages=find_packages(where="src"),
       package_dir={"": "src"},
       install_requires=[
           # From requirements.txt
       ],
       python_requires=">=3.8",
   )
   ```

2. **Add `__init__.py` files:**
   ```python
   # src/crm_rag/__init__.py
   from .rag_system import RAGSystem
   from .config import ConfigLoader

   __version__ = "1.0.0"
   ```

### Phase 5: Update Documentation

1. Create comprehensive docs in `docs/`:
   - SETUP.md
   - CONFIGURATION.md
   - API.md
   - ONTOLOGIES.md
   - SECURITY.md

2. Update README.md to be concise with links to detailed docs

---

## Benefits of New Structure

### For Developers
✅ **Clear organization** - Easy to find relevant code
✅ **Modular design** - Components can be tested/developed independently
✅ **Standard Python package** - Can be installed with pip
✅ **Easier testing** - Test directory mirrors source structure
✅ **Better IDE support** - Standard structure works with autocomplete

### For Users
✅ **Clearer setup** - Documentation is organized and easy to follow
✅ **Simpler configuration** - Clear separation of config templates
✅ **Better error messages** - Can reference specific docs (e.g., "See docs/ONTOLOGIES.md")
✅ **Easier troubleshooting** - Generated files clearly separated from source

### For Maintenance
✅ **Easier to understand** - New contributors can navigate quickly
✅ **Better git history** - Changes grouped by concern
✅ **Safer updates** - Generated files clearly marked in .gitignore
✅ **Professional appearance** - Follows Python community standards

---

## Alternative: Minimal Reorganization

If a full restructure is too much, here's a **minimal cleanup**:

```
CRM_RAG/
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/                    # Application code
│   ├── main.py
│   ├── rag_system.py
│   ├── graph_store.py
│   ├── llm_providers.py
│   ├── config_loader.py
│   ├── static/
│   └── templates/
│
├── config/                 # Config files
│   └── *.env.example
│
├── data/                   # Data files
│   ├── ontologies/
│   ├── labels/            # Generated
│   └── cache/             # Generated
│
├── docs/                   # Documentation
│   └── *.md
│
└── scripts/                # Utility scripts
    └── extract_labels.py
```

This keeps the benefits of organization while requiring fewer code changes.

---

## Recommendation

**Start with the Minimal Reorganization**, then gradually refactor toward the full structure:

1. **Week 1:** Create directory structure, move files (no code changes)
2. **Week 2:** Split large modules (llm_providers.py, universal_rag_system.py)
3. **Week 3:** Update documentation
4. **Week 4:** Add tests, finalize package structure

This allows continuous use of the system while improving organization incrementally.
