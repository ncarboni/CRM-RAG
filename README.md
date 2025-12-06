# RAG Architecture for CIDOC-CRM

Graph-based RAG (Retrieval-Augmented Generation) system for querying CIDOC-CRM RDF data about Byzantine art. Uses coherent subgraph extraction with weighted CIDOC-CRM relationships to enhance document retrieval.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env.secrets` file and add your API keys:

```bash
# Copy the example file
cp .env.secrets.example .env.secrets

# Edit .env.secrets and add your actual API keys
OPENAI_API_KEY=your_actual_openai_key_here
ANTHROPIC_API_KEY=your_actual_anthropic_key_here
R1_API_KEY=your_actual_r1_key_here
```

### 3. Configure SPARQL Endpoint

The default SPARQL endpoint is `http://localhost:3030/asinou/sparql`.

**To change the endpoint**: Edit the `FUSEKI_ENDPOINT` variable in your `.env.*` files:

```bash
# Example: .env.openai, .env.claude, .env.r1, .env.ollama
FUSEKI_ENDPOINT=http://your-server:3030/your-dataset/sparql
```

### 4. Customize Settings (Optional)

The repository includes pre-configured `.env.*` files for each provider. You can also customize:
- Model names
- Temperature
- Max tokens
- Port number

### 5. Start Your SPARQL Endpoint

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