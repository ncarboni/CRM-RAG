


# Run with OpenAI
python main.py --env .env.openai

# Run with Claude
python main.py --env .env.claude

# Run with R1
python main.py --env .env.r1

# Run with Ollama
python main.py --env .env.ollama

# Force rebuild of document graph and vector store
python main.py --env .env.openai --rebuild