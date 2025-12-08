# Repository Reorganization Implementation Plan

**Goal**: Minimal reorganization for easier navigation with code path updates to prevent link rot

**Approach**: One-time migration, focus on findability

---

## Proposed New Structure

```
CRM_RAG/
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/                           # Configuration files (NEW)
│   ├── .env.openai.example
│   ├── .env.claude.example
│   ├── .env.r1.example
│   ├── .env.ollama.example
│   └── .env.secrets.example
│
├── data/                             # Data files (NEW)
│   ├── ontologies/                   # Ontology files (moved from ontology/)
│   │   ├── CIDOC_CRM_v7.1.3.rdf
│   │   ├── CRMdig_v3.2.1.rdf
│   │   ├── frbroo.rdf
│   │   ├── skos_2009-08-18.n3
│   │   └── vir.ttl
│   │
│   ├── labels/                       # Extracted labels (NEW - generated files)
│   │   ├── property_labels.json      # (gitignored)
│   │   ├── class_labels.json         # (gitignored)
│   │   └── ontology_classes.json     # (gitignored)
│   │
│   ├── cache/                        # Cached data (NEW - generated files)
│   │   ├── document_graph.pkl        # (gitignored)
│   │   └── vector_index/             # (gitignored)
│   │
│   └── documents/                    # Entity documents (NEW - generated files)
│       └── entity_documents/         # (gitignored)
│
├── docs/                             # Documentation (NEW)
│   ├── ARCHITECTURE.md               # (moved from root)
│   └── REPOSITORY_REORGANIZATION.md  # (moved from root)
│
├── scripts/                          # Utility scripts (NEW)
│   └── extract_ontology_labels.py    # (moved from root)
│
├── logs/                             # Log files (NEW)
│   └── app.log                       # (gitignored)
│
├── static/                           # Keep as is (Flask serves from here)
│   ├── css/
│   │   ├── base.css
│   │   └── chat.css
│   └── js/
│       └── chat.js
│
├── templates/                        # Keep as is (Flask serves from here)
│   ├── base.html
│   └── chat.html
│
└── [Python files remain in root]     # Keep as is for easy imports
    ├── main.py
    ├── universal_rag_system.py
    ├── graph_document_store.py
    ├── llm_providers.py
    └── config_loader.py
```

---

## Key Design Decisions

1. **Python files stay in root**: No changes to import paths, easier to run `python main.py`
2. **Static/templates stay in root**: Flask default locations, no code changes needed
3. **Organize data by type**: config/, data/, docs/, scripts/, logs/
4. **Separate source from generated**: data/ontologies/ vs data/labels/, data/cache/
5. **Minimal disruption**: Only move files that improve organization significantly

---

## Step-by-Step Migration Plan

### Phase 1: Create Directory Structure

```bash
# Create new directories
mkdir -p config
mkdir -p data/ontologies
mkdir -p data/labels
mkdir -p data/cache
mkdir -p data/documents
mkdir -p docs
mkdir -p scripts
mkdir -p logs
```

### Phase 2: Move Files

**Config files:**
```bash
mv .env.openai.example config/
mv .env.claude.example config/
mv .env.r1.example config/
mv .env.ollama.example config/
mv .env.secrets.example config/
```

**Ontology files:**
```bash
mv ontology/*.rdf data/ontologies/
mv ontology/*.ttl data/ontologies/
mv ontology/*.n3 data/ontologies/
rmdir ontology  # Remove empty directory
```

**Generated label files:**
```bash
mv property_labels.json data/labels/ 2>/dev/null || true
mv class_labels.json data/labels/ 2>/dev/null || true
mv ontology_classes.json data/labels/ 2>/dev/null || true
```

**Cache files:**
```bash
mv document_graph.pkl data/cache/ 2>/dev/null || true
mv document_graph_temp.pkl data/cache/ 2>/dev/null || true
mv vector_index data/cache/ 2>/dev/null || true
```

**Entity documents:**
```bash
mv entity_documents data/documents/ 2>/dev/null || true
```

**Documentation:**
```bash
mv ARCHITECTURE.md docs/
mv REPOSITORY_REORGANIZATION.md docs/
```

**Scripts:**
```bash
mv extract_ontology_labels.py scripts/
```

**Logs:**
```bash
mv app.log logs/ 2>/dev/null || true
```

### Phase 3: Update Code Paths

#### File: `universal_rag_system.py`

**Lines to update:**

1. **Ontology directory** (multiple locations):
```python
# OLD:
ontology_dir='ontology'

# NEW:
ontology_dir='data/ontologies'
```

2. **Label files** (~line 213-234, 235-293):
```python
# OLD:
labels_file = 'property_labels.json'
classes_file = 'ontology_classes.json'
labels_file = 'class_labels.json'

# NEW:
labels_file = 'data/labels/property_labels.json'
classes_file = 'data/labels/ontology_classes.json'
labels_file = 'data/labels/class_labels.json'
```

3. **Document graph cache** (~line 147-155, 166-190):
```python
# OLD:
cache_file = 'document_graph.pkl'
temp_cache = 'document_graph_temp.pkl'

# NEW:
cache_file = 'data/cache/document_graph.pkl'
temp_cache = 'data/cache/document_graph_temp.pkl'
```

4. **Vector index** (~line 195):
```python
# OLD:
index_file = 'vector_index'

# NEW:
index_file = 'data/cache/vector_index'
```

5. **Entity documents** (~line 559):
```python
# OLD:
output_dir = 'entity_documents'

# NEW:
output_dir = 'data/documents/entity_documents'
```

6. **Validation report** (~line 959):
```python
# OLD:
output_file = 'ontology_validation_report.txt'

# NEW:
output_file = 'logs/ontology_validation_report.txt'
```

#### File: `scripts/extract_ontology_labels.py` (after moving)

**Lines to update:**

1. **Default ontology directory** (~line 17, 155, 301, 347):
```python
# OLD:
def extract_property_labels(ontology_dir='ontology'):
def extract_ontology_classes(ontology_dir='ontology'):
def run_extraction(ontology_dir='ontology', ...):
success = run_extraction('ontology', ...)

# NEW:
def extract_property_labels(ontology_dir='data/ontologies'):
def extract_ontology_classes(ontology_dir='data/ontologies'):
def run_extraction(ontology_dir='data/ontologies', ...):
success = run_extraction('data/ontologies', ...)
```

2. **Default output files** (~line 278, 285, 294, 301, 347):
```python
# OLD:
output_file='property_labels.json'
output_file='ontology_classes.json'
output_file='class_labels.json'

# NEW:
output_file='data/labels/property_labels.json'
classes_file='data/labels/ontology_classes.json'
class_labels_file='data/labels/class_labels.json'
```

#### File: `main.py`

**Lines to update:**

1. **Log file** (~line 33):
```python
# OLD:
filename='app.log'

# NEW:
filename='logs/app.log'
```

2. **Entity documents path validation** (~line 179):
```python
# OLD:
base_dir = os.path.abspath('entity_documents')

# NEW:
base_dir = os.path.abspath('data/documents/entity_documents')
```

#### File: `config_loader.py`

**No changes needed** - this file loads .env from current directory, user will copy config/.env.example to .env in root

---

## Phase 4: Update .gitignore Patterns

**Update these patterns in `.gitignore`:**

```gitignore
# OLD patterns to REMOVE:
property_labels.json
class_labels.json
ontology_classes.json
document_graph.pkl
document_graph_temp.pkl
vector_index/
entity_documents/
ontology_validation_report.txt
app.log

# NEW patterns to ADD:
data/labels/*.json
data/cache/
data/documents/entity_documents/
logs/*.log
logs/*.txt
```

**Note**: Keep all other patterns as-is (environment files, Python, IDE, etc.)

---

## Phase 5: Update Documentation

### Update README.md

Add section at the top:

```markdown
## Repository Structure

- `config/` - Configuration file templates (.env examples)
- `data/` - All data files (ontologies, generated files)
  - `ontologies/` - CIDOC-CRM, VIR, CRMdig ontology files
  - `labels/` - Extracted labels (auto-generated)
  - `cache/` - Document graph and vector index (auto-generated)
  - `documents/` - Entity documents (auto-generated)
- `docs/` - Documentation (architecture, guides)
- `scripts/` - Utility scripts (label extraction, etc.)
- `logs/` - Application logs
- `static/`, `templates/` - Web interface assets
```

Update setup instructions:

```markdown
## Setup

1. Copy configuration template:
   ```bash
   cp config/.env.openai.example .env
   cp config/.env.secrets.example .env.secrets
   ```

2. Edit `.env` and `.env.secrets` with your API keys

3. Extract ontology labels:
   ```bash
   python scripts/extract_ontology_labels.py
   ```

4. Run the application:
   ```bash
   python main.py
   ```
```

### Update docs/ARCHITECTURE.md

Update file paths in all examples to reflect new structure:
- `ontology/` → `data/ontologies/`
- `property_labels.json` → `data/labels/property_labels.json`
- etc.

---

## Phase 6: Testing After Migration

### Test Checklist

1. **Label extraction:**
   ```bash
   python scripts/extract_ontology_labels.py
   # Verify files created in data/labels/
   ```

2. **Cache creation:**
   ```bash
   python main.py
   # Verify document_graph.pkl and vector_index/ in data/cache/
   ```

3. **Entity documents:**
   ```bash
   # Run a query in the web interface
   # Verify entity_documents/ created in data/documents/
   ```

4. **Logs:**
   ```bash
   # Check logs/app.log exists and is being written
   ```

5. **Web interface:**
   ```bash
   # Verify static/css and templates/ still load correctly
   ```

---

## Complete Migration Script

```bash
#!/bin/bash
# migrate_repository.sh

set -e  # Exit on error

echo "=== CRM_RAG Repository Reorganization ==="
echo ""

# Phase 1: Create directories
echo "[1/6] Creating directory structure..."
mkdir -p config
mkdir -p data/ontologies
mkdir -p data/labels
mkdir -p data/cache
mkdir -p data/documents
mkdir -p docs
mkdir -p scripts
mkdir -p logs
echo "✓ Directories created"
echo ""

# Phase 2: Move files
echo "[2/6] Moving files..."

# Config files
echo "  Moving config files..."
mv .env.openai.example config/ 2>/dev/null || true
mv .env.claude.example config/ 2>/dev/null || true
mv .env.r1.example config/ 2>/dev/null || true
mv .env.ollama.example config/ 2>/dev/null || true
mv .env.secrets.example config/ 2>/dev/null || true

# Ontology files
echo "  Moving ontology files..."
if [ -d "ontology" ]; then
    mv ontology/*.rdf data/ontologies/ 2>/dev/null || true
    mv ontology/*.ttl data/ontologies/ 2>/dev/null || true
    mv ontology/*.n3 data/ontologies/ 2>/dev/null || true
    mv ontology/*.owl data/ontologies/ 2>/dev/null || true
    rmdir ontology 2>/dev/null || true
fi

# Generated label files
echo "  Moving generated label files..."
mv property_labels.json data/labels/ 2>/dev/null || true
mv class_labels.json data/labels/ 2>/dev/null || true
mv ontology_classes.json data/labels/ 2>/dev/null || true

# Cache files
echo "  Moving cache files..."
mv document_graph.pkl data/cache/ 2>/dev/null || true
mv document_graph_temp.pkl data/cache/ 2>/dev/null || true
mv vector_index data/cache/ 2>/dev/null || true

# Entity documents
echo "  Moving entity documents..."
mv entity_documents data/documents/ 2>/dev/null || true

# Documentation
echo "  Moving documentation..."
mv ARCHITECTURE.md docs/ 2>/dev/null || true
mv REPOSITORY_REORGANIZATION.md docs/ 2>/dev/null || true

# Scripts
echo "  Moving scripts..."
mv extract_ontology_labels.py scripts/ 2>/dev/null || true

# Logs
echo "  Moving logs..."
mv app.log logs/ 2>/dev/null || true

echo "✓ Files moved"
echo ""

echo "[3/6] Files moved successfully!"
echo "✓ Next steps:"
echo "  1. Update code paths in Python files (see REORGANIZATION_PLAN.md Phase 3)"
echo "  2. Update .gitignore patterns (see REORGANIZATION_PLAN.md Phase 4)"
echo "  3. Update README.md (see REORGANIZATION_PLAN.md Phase 5)"
echo "  4. Run tests (see REORGANIZATION_PLAN.md Phase 6)"
echo ""
echo "=== Migration script completed ==="
```

---

## Rollback Plan (If Needed)

If something goes wrong, reverse the migration:

```bash
#!/bin/bash
# rollback_migration.sh

# Recreate ontology directory
mkdir -p ontology
mv data/ontologies/* ontology/

# Move config files back
mv config/.env*.example .

# Move generated files back
mv data/labels/*.json . 2>/dev/null || true
mv data/cache/*.pkl . 2>/dev/null || true
mv data/cache/vector_index . 2>/dev/null || true
mv data/documents/entity_documents . 2>/dev/null || true

# Move docs back
mv docs/*.md .

# Move scripts back
mv scripts/*.py .

# Move logs back
mv logs/*.log . 2>/dev/null || true

# Remove empty directories
rmdir data/ontologies data/labels data/cache data/documents docs scripts logs config data 2>/dev/null || true

echo "Rollback complete"
```

---

## Summary

**What Changes:**
- 7 new directories created
- ~30 files moved to organized locations
- 15 code path updates across 3 Python files
- .gitignore pattern updates
- README.md structure documentation

**What Stays the Same:**
- Python files remain in root (no import changes)
- Flask static/templates in root (no template path changes)
- .env files in root (user copies from config/)
- Git history preserved (file moves tracked by git)

**Benefits:**
- ✅ Easier to find ontology files (data/ontologies/)
- ✅ Clear separation: source vs generated files
- ✅ Config templates organized in one place
- ✅ Documentation in dedicated folder
- ✅ Logs separated from code
- ✅ No broken imports or link rot

**Time Estimate:**
- File migration: 5 minutes (automated script)
- Code updates: 15 minutes (manual edits)
- Testing: 10 minutes
- **Total: ~30 minutes**
