# Configuration Files

This directory contains configuration files for the CRM RAG system.

## Configuration Architecture

Settings are organized across different files by purpose:

| File | Purpose |
|------|---------|
| `datasets.yaml` | **SPARQL endpoints** and per-dataset settings |
| `.env.*` files | **LLM/embedding providers** and API settings |
| `.env.secrets` | **API keys** (keep out of version control) |
| `interface.yaml` | **UI customization** (titles, messages, questions) |

**Important:** SPARQL endpoints are defined **only** in `datasets.yaml`, not in `.env` files.

## Environment Configuration (.env files)

Configuration files for different LLM providers:

- `.env.openai.example` - OpenAI (GPT) configuration template
- `.env.claude.example` - Anthropic Claude configuration template
- `.env.r1.example` - DeepSeek R1 configuration template
- `.env.ollama.example` - Ollama (local) configuration template
- `.env.local.example` - **Local embeddings** with sentence-transformers (fast, no API costs)
- `.env.secrets.example` - API keys template

### Setup

1. Copy the provider template you want to use:
   ```bash
   cp config/.env.openai.example config/.env.openai
   ```

2. Copy the secrets template:
   ```bash
   cp config/.env.secrets.example config/.env.secrets
   ```

3. Edit the files in `config/` and add your API keys

## Local Embeddings (Fast Processing for Large Datasets)

For large datasets (5,000+ entities), local embeddings are **highly recommended**. They are:
- **10-100x faster** than API-based embeddings (no rate limits, no network latency)
- **Free** (no API costs)
- **Private** (data never leaves your machine)
- **Resumable** (embedding cache allows stopping and continuing later)

### Setup for Local Embeddings

1. Copy the local embeddings template:
   ```bash
   cp config/.env.local.example config/.env.local
   ```

2. Edit `config/.env.local` and set your OpenAI API key (still needed for chat/LLM)

3. Run with local embeddings:
   ```bash
   python main.py --env .env.local
   ```

Or use CLI flags with any config file:
```bash
python main.py --env .env.openai --embedding-provider local
```

### Recommended Embedding Models

| Model | Quality | Speed | Size | Best For |
|-------|---------|-------|------|----------|
| `BAAI/bge-m3` | Best | Medium | 2.3GB | Multilingual datasets |
| `BAAI/bge-base-en-v1.5` | Excellent | Fast | 440MB | English-only datasets |
| `all-MiniLM-L6-v2` | Good | Very Fast | 90MB | Quick testing |

### Performance Comparison

For a 50,000 entity dataset:
- **OpenAI API**: 2-4 days (rate limited)
- **Local (CPU)**: 1-2 hours
- **Local (GPU)**: 10-20 minutes

### Embedding Cache

By default, embeddings are cached to disk. This allows:
- **Resumability**: Stop processing and continue later
- **Incremental updates**: Only new entities need embedding
- **Fast rebuilds**: Cached embeddings are reused

Disable cache with: `--no-embedding-cache`

## Interface Customization (interface.yaml)

The `interface.yaml` file controls the chat interface appearance and text.

### What You Can Customize

- **Page title** - Browser tab title
- **Header title** - Main chatbot title shown in the interface
- **Welcome message** - Initial greeting message
- **Input placeholder** - Placeholder text in the input box
- **Example questions** - Questions shown below the input (can add/remove/modify)
- **About section** - Description and features list

### How to Customize

1. Open `config/interface.yaml` in any text editor

2. Edit the values:
   ```yaml
   page_title: "My Custom Dataset Chat"
   header_title: "My Custom Chatbot"
   welcome_message: "Hello! Ask me about..."

   example_questions:
     - "Your first question?"
     - "Your second question?"
     - "Add as many as you want!"
   ```

3. Save the file and restart the application:
   ```bash
   python main.py --env .env.openai
   ```

4. The interface will update automatically with your customizations

### Example: Customizing for a Different Dataset

If you're using this system for a different cultural heritage dataset (not Asinou), update the interface.yaml:

```yaml
page_title: "Pompeii Frescoes Chat"
header_title: "Pompeii Dataset Chatbot"
welcome_message: "Hello! I can answer questions about Pompeii frescoes and artifacts. How can I help you?"
input_placeholder: "Ask about Pompeii frescoes, artifacts, locations..."

example_questions:
  - "What frescoes are in the House of the Vettii?"
  - "Tell me about Fourth Style paintings"
  - "Which houses contain mythological scenes?"
  - "Describe the Villa of Mysteries frescoes"
```

### Backup

Keep a copy of `interface.yaml.example` as reference. If you want to reset to defaults, you can copy it:

```bash
cp config/interface.yaml.example config/interface.yaml
```
