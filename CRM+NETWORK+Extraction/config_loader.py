"""
Configuration loader for the Asinou Dataset Chatbot.
This module handles loading configuration from environment files.
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Configuration loader for the Asinou Dataset Chatbot"""
    
    @staticmethod
    def load_config(env_file: str = None) -> Dict[str, Any]:
        """Load configuration from environment file"""
        if env_file and os.path.exists(env_file):
            logger.info(f"Loading configuration from {env_file}")
            # Load the specified .env file
            load_dotenv(env_file)
        else:
            logger.info("Loading configuration from default .env file")
            # Load the default .env file
            load_dotenv()
        
        # Load LLM provider configuration
        llm_provider = os.environ.get("LLM_PROVIDER", "openai").lower()
        
        config = {
            "llm_provider": llm_provider,
            "fuseki_endpoint": os.environ.get("FUSEKI_ENDPOINT", "http://localhost:3030/asinou/sparql"),
            "temperature": float(os.environ.get("TEMPERATURE", "0.7")),
            "port": int(os.environ.get("PORT", "5001"))
        }
        
        # Add provider-specific configuration
        if llm_provider == "openai":
            config.update({
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                "embedding_model": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                "max_tokens": int(os.environ.get("OPENAI_MAX_TOKENS", "4096"))
            })
        elif llm_provider == "anthropic":
            config.update({
                "api_key": os.environ.get("ANTHROPIC_API_KEY"),
                "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet"),
                "max_tokens": int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096")),
                # For embeddings with Claude (which doesn't have its own)
                "openai_api_key": os.environ.get("OPENAI_API_KEY")
            })
        elif llm_provider == "r1":
            config.update({
                "api_key": os.environ.get("R1_API_KEY"),
                "model": os.environ.get("R1_MODEL", "rank-r1-v16"),
                "embedding_model": os.environ.get("R1_EMBEDDING_MODEL", "e5-small-v2"),
                "max_tokens": int(os.environ.get("R1_MAX_TOKENS", "4096"))
            })
        elif llm_provider == "ollama":
            config.update({
                "model": os.environ.get("OLLAMA_MODEL", "mistral"),
                "embedding_model": os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
                "host": os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            })
        
        # Validate required configuration
        if llm_provider != "ollama" and not config.get("api_key"):
            logger.warning(f"{llm_provider.upper()}_API_KEY environment variable is not set! The application may not function correctly.")
        
        return config