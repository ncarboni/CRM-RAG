"""
Configuration loader for the CIDOC-CRM RAG system.
This module handles loading configuration from environment files.
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import yaml

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Configuration loader for the CIDOC-CRM RAG system"""
    
    @staticmethod
    def load_config(env_file: str = None) -> Dict[str, Any]:
        """Load configuration from environment file"""
        # Determine base directory (repo root)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_dir = os.path.join(base_dir, "config")

        if env_file:
            # If relative path provided, check both config/ and absolute path
            if not os.path.isabs(env_file):
                config_path = os.path.join(config_dir, env_file)
                if os.path.exists(config_path):
                    env_file = config_path

            if os.path.exists(env_file):
                logger.info(f"Loading configuration from {env_file}")
                load_dotenv(env_file)
            else:
                logger.warning(f"Specified env file not found: {env_file}, loading default")
                default_env = os.path.join(config_dir, ".env")
                if os.path.exists(default_env):
                    load_dotenv(default_env)
        else:
            # Load the default .env file from config/
            default_env = os.path.join(config_dir, ".env")
            if os.path.exists(default_env):
                logger.info(f"Loading configuration from {default_env}")
                load_dotenv(default_env)
            else:
                logger.warning("No .env file found in config/ directory")

        # Load secrets from config/.env.secrets (if it exists)
        secrets_file = os.path.join(config_dir, ".env.secrets")
        if os.path.exists(secrets_file):
            logger.info(f"Loading secrets from {secrets_file}")
            load_dotenv(secrets_file, override=True)
        else:
            logger.warning(".env.secrets file not found. API keys should be set in environment or config/.env.secrets")

        # Load LLM provider configuration
        llm_provider = os.environ.get("LLM_PROVIDER", "openai").lower()

        # Load embedding provider (can be different from LLM provider)
        # Options: openai, sentence-transformers (or "local"), ollama
        embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "").lower()
        if not embedding_provider:
            embedding_provider = llm_provider  # Default to same as LLM provider

        config = {
            "llm_provider": llm_provider,
            "embedding_provider": embedding_provider,
            "temperature": float(os.environ.get("TEMPERATURE", "0.7")),
            "port": int(os.environ.get("PORT", "5001")),
            # Embedding cache (default enabled)
            "use_embedding_cache": os.environ.get("USE_EMBEDDING_CACHE", "true").lower() == "true",
        }

        # Note: SPARQL endpoints are configured in config/datasets.yaml, not here
        
        # Add provider-specific configuration
        if llm_provider == "openai":
            config.update({
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
                "embedding_model": os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
                "max_tokens": int(os.environ.get("OPENAI_MAX_TOKENS", "4096")),
                # Concurrent embedding configuration (for parallel API calls)
                "embedding_max_concurrent": int(os.environ.get("EMBEDDING_MAX_CONCURRENT", "10")),
                "embedding_retry_attempts": int(os.environ.get("EMBEDDING_RETRY_ATTEMPTS", "3")),
                "embedding_retry_delay": float(os.environ.get("EMBEDDING_RETRY_DELAY", "1.0")),
                "embedding_chunk_size": int(os.environ.get("EMBEDDING_CHUNK_SIZE", "100")),
                # Outer batch size for processing entities (before checkpoint save)
                "embedding_batch_size": int(os.environ.get("EMBEDDING_BATCH_SIZE", "500")),
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

        # Add embedding-specific configuration (for separate embedding provider)
        if embedding_provider in ("sentence-transformers", "local"):
            # For local embeddings, always use sentence-transformers compatible model
            # Override any OpenAI model that might have been set
            local_embedding_model = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
            # Don't use OpenAI model names with sentence-transformers
            if local_embedding_model.startswith("text-embedding"):
                local_embedding_model = "BAAI/bge-m3"

            config.update({
                "embedding_model": local_embedding_model,
                "embedding_batch_size": int(os.environ.get("EMBEDDING_BATCH_SIZE", "64")),
                "embedding_device": os.environ.get("EMBEDDING_DEVICE", "auto"),
            })

        # Validate required configuration
        if llm_provider != "ollama" and not config.get("api_key"):
            logger.warning(f"{llm_provider.upper()}_API_KEY environment variable is not set! The application may not function correctly.")

        return config

    @staticmethod
    def load_datasets_config() -> Dict[str, Any]:
        """
        Load datasets configuration from config/datasets.yaml.

        Returns:
            dict: Datasets configuration with 'datasets' dict and optional 'default_dataset'
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config", "datasets.yaml")

        # Default configuration if file doesn't exist (backward compatibility)
        default_config = {
            "datasets": {},
            "default_dataset": None
        }

        if not os.path.exists(config_path):
            logger.warning(f"Datasets config not found at {config_path}")
            logger.info("Running in single-dataset mode (backward compatibility)")
            return default_config

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config or "datasets" not in config:
                logger.warning("Invalid datasets.yaml: missing 'datasets' key")
                return default_config

            logger.info(f"Loaded {len(config['datasets'])} datasets from {config_path}")
            return config

        except Exception as e:
            logger.error(f"Error loading datasets config: {str(e)}")
            return default_config