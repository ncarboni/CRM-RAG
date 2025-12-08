# RAG Architecture for CIDOC-CRM

Graph-based RAG (Retrieval-Augmented Generation) system for querying CIDOC-CRM RDF data about Byzantine art. Uses coherent subgraph extraction with weighted CIDOC-CRM relationships to enhance document retrieval.

## Repository Structure

```
CRM_RAG/
├── config/              Configuration files
│   ├── .env.openai.example
│   ├── .env.claude.example
│   ├── .env.r1.example
│   ├── .env.ollama.example
│   ├── .env.secrets.example
│   ├── interface.yaml        # Chat interface customization
│   └── README.md             # Configuration guide
├── data/                All data files
│   ├── ontologies/      CIDOC-CRM, VIR, CRMdig ontology files
│   ├── labels/          Extracted labels (auto-generated)
│   ├── cache/           Document graph and vector index (auto-generated)
│   └── documents/       Entity documents (auto-generated)
├── docs/                Documentation
│   ├── ARCHITECTURE.md
│   └── REORGANIZATION_PLAN.md
├── scripts/             Utility scripts
│   └── extract_ontology_labels.py
├── logs/                Application logs
├── static/              Web interface CSS and JavaScript
├── templates/           Web interface HTML templates
└── *.py                 Python application files
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

### 3. Configure SPARQL Endpoint

The default SPARQL endpoint is `http://localhost:3030/asinou/sparql`.

**To change the endpoint**: Edit the `FUSEKI_ENDPOINT` variable in your `config/.env.*` files:

```bash
# Example: config/.env.openai, config/.env.claude, etc.
FUSEKI_ENDPOINT=http://your-server:3030/your-dataset/sparql
```

### 4. Extract Ontology Labels

Extract English labels from ontology files (required on first run):

```bash
python scripts/extract_ontology_labels.py
```

This creates label files in `data/labels/` used by the RAG system.

### 5. Customize Chat Interface (Optional)

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

### 6. Start Your SPARQL Endpoint

Ensure your SPARQL server is running with your CIDOC-CRM dataset loaded at the configured endpoint.

## Usage

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