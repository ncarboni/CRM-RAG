# Configuration Files

This directory contains configuration files for the CRM RAG system.

## Environment Configuration (.env files)

Configuration files for different LLM providers:

- `.env.openai.example` - OpenAI (GPT) configuration template
- `.env.claude.example` - Anthropic Claude configuration template
- `.env.r1.example` - DeepSeek R1 configuration template
- `.env.ollama.example` - Ollama (local) configuration template
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
