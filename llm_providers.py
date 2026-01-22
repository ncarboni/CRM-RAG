"""
LLM provider abstraction layer to support multiple AI model providers.
This module provides a unified interface for different LLM APIs.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for text"""
        pass

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch.
        Default implementation calls get_embeddings for each text.
        Subclasses can override for more efficient batch processing.
        """
        return [self.get_embeddings(text) for text in texts]

    def supports_batch_embedding(self) -> bool:
        """Return True if this provider supports efficient batch embedding."""
        return False

class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", embedding_model: str = "text-embedding-3-small", 
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        """Initialize OpenAI provider"""
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._embeddings = None
        self._llm = None
        
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using OpenAI API"""
        from langchain_openai import ChatOpenAI
        
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                openai_api_key=self.api_key
            )
        
        response = self._llm.invoke(system_prompt + "\n\n" + user_prompt)
        return response.content
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using OpenAI API"""
        from langchain_openai import OpenAIEmbeddings

        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=self.api_key
            )

        return self._embeddings.embed_query(text)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts using OpenAI's native batch API.
        This is significantly faster than calling embed_query in a loop.

        OpenAI's embed_documents handles batching internally with chunk_size=1000.
        """
        from langchain_openai import OpenAIEmbeddings

        if not texts:
            return []

        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=self.api_key
            )

        # embed_documents sends texts in batches to the API (default chunk_size=1000)
        return self._embeddings.embed_documents(texts)

    def supports_batch_embedding(self) -> bool:
        """OpenAI supports efficient batch embedding via embed_documents."""
        return True

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet", 
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        """Initialize Anthropic provider"""
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        # Claude doesn't have its own embeddings, so we'll use a default embedding provider
        self._openai_embeddings = None
        self.openai_api_key = None  # Will need to be set separately
        
    def set_embedding_provider(self, provider: str = "openai", **kwargs):
        """Set embedding provider for Claude"""
        if provider == "openai":
            self.openai_api_key = kwargs.get("api_key")
        
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using Anthropic API"""
        try:
            from langchain_anthropic import ChatAnthropic  # Requires langchain_anthropic package
            
            if self._llm is None:
                self._llm = ChatAnthropic(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    anthropic_api_key=self.api_key
                )
            
            response = self._llm.invoke(system_prompt + "\n\n" + user_prompt)
            return response.content
        except ImportError:
            logger.error("langchain_anthropic package not installed. Please install it with 'pip install langchain-anthropic'")
            raise
        
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings (using OpenAI as Claude doesn't offer embeddings)"""
        from langchain_openai import OpenAIEmbeddings

        if self.openai_api_key is None:
            raise ValueError("OpenAI API key not set for embeddings. Use set_embedding_provider().")

        if self._openai_embeddings is None:
            self._openai_embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key
            )

        return self._openai_embeddings.embed_query(text)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts using OpenAI's native batch API.
        Claude doesn't have embeddings, so we use OpenAI's embed_documents.
        """
        from langchain_openai import OpenAIEmbeddings

        if not texts:
            return []

        if self.openai_api_key is None:
            raise ValueError("OpenAI API key not set for embeddings. Use set_embedding_provider().")

        if self._openai_embeddings is None:
            self._openai_embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key
            )

        return self._openai_embeddings.embed_documents(texts)

    def supports_batch_embedding(self) -> bool:
        """Anthropic uses OpenAI embeddings which support batch embedding."""
        return True

class R1Provider(BaseLLMProvider):
    """R1 API provider"""

    def __init__(self, api_key: str, model: str = "rank-r1-v16", embedding_model: str = "e5-small-v2",
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        """Initialize R1 provider"""
        import requests

        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        self._embeddings = None

        # Create a secure session that doesn't use .netrc credentials
        # This prevents CVE-related .netrc credential leaks
        self._session = requests.Session()
        self._session.trust_env = False
        
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using R1 API"""
        import json

        endpoint = "https://api.cohere.ai/v1/chat"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "message": user_prompt,
            "model": self.model,
            "temperature": self.temperature,
            "chat_history": [],
            "system_prompt": system_prompt
        }

        if self.max_tokens:
            data["max_tokens"] = self.max_tokens

        # Use secure session instead of direct requests call
        response = self._session.post(endpoint, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["text"]
        else:
            raise Exception(f"R1 API request failed: {response.text}")
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using R1 (Cohere) API"""
        endpoint = "https://api.cohere.ai/v1/embed"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "texts": [text],
            "model": self.embedding_model,
            "input_type": "search_query"
        }

        # Use secure session instead of direct requests call
        response = self._session.post(endpoint, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()["embeddings"][0]
        else:
            raise Exception(f"R1 embedding request failed: {response.text}")

class OllamaProvider(BaseLLMProvider):
    """Local Ollama API provider"""
    
    def __init__(self, model: str = "mistral", embedding_model: str = "nomic-embed-text",
                 temperature: float = 0.7, host: str = "http://localhost:11434"):
        """Initialize Ollama provider"""
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.host = host
        self._llm = None
        self._embeddings = None
        
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using local Ollama API"""
        try:
            from langchain_community.llms import Ollama
            
            if self._llm is None:
                self._llm = Ollama(
                    model=self.model,
                    temperature=self.temperature,
                    base_url=self.host
                )
            
            # Combine system and user prompts for Ollama
            combined_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self._llm.invoke(combined_prompt)
            return response
        except ImportError:
            logger.error("langchain_community package not installed. Please install it with 'pip install langchain-community'")
            raise
        
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using Ollama"""
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            
            if self._embeddings is None:
                self._embeddings = OllamaEmbeddings(
                    model=self.embedding_model,
                    base_url=self.host
                )
            
            return self._embeddings.embed_query(text)
        except ImportError:
            logger.error("langchain_community package not installed. Please install it with 'pip install langchain-community'")
            raise


class SentenceTransformersProvider(BaseLLMProvider):
    """
    Local embedding provider using sentence-transformers library.
    Provides high-speed batch embedding with optional GPU acceleration.

    Note: This provider only supports embeddings, not text generation.
    For LLM generation, use a separate provider (OpenAI, Anthropic, etc.)
    and combine with this for embeddings via the embedding_provider config.
    """

    # Default model choices with trade-offs:
    # - BAAI/bge-m3: Best quality, multilingual, 1024 dims, ~2.3GB, slower
    # - BAAI/bge-base-en-v1.5: Best English, 768 dims, ~440MB, fast
    # - all-MiniLM-L6-v2: Good balance, 384 dims, ~90MB, very fast
    DEFAULT_MODEL = "BAAI/bge-m3"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SentenceTransformers provider.

        Args:
            config: Configuration dictionary with optional keys:
                - embedding_model: Model name (default: BAAI/bge-m3)
                - embedding_batch_size: Batch size for encoding (default: 64)
                - embedding_device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_name = config.get("embedding_model", self.DEFAULT_MODEL)

        # Validate model name - don't use OpenAI model names
        if self.model_name.startswith("text-embedding") or "openai" in self.model_name.lower():
            logger.warning(f"Invalid model for sentence-transformers: {self.model_name}")
            logger.warning(f"Using default model: {self.DEFAULT_MODEL}")
            self.model_name = self.DEFAULT_MODEL

        self.batch_size = int(config.get("embedding_batch_size", 64))
        self._device = config.get("embedding_device", "auto")
        self._model = None  # Lazy load to avoid import overhead

        # For LLM generation, we delegate to another provider if configured
        self._llm_provider = None
        llm_provider_name = config.get("llm_provider")
        if llm_provider_name and llm_provider_name not in ("sentence-transformers", "local"):
            # We'll initialize the LLM provider later to avoid circular dependency
            self._llm_config = config
            self._llm_provider_name = llm_provider_name
        else:
            self._llm_config = None
            self._llm_provider_name = None

    def _detect_device(self) -> str:
        """Auto-detect CUDA GPU or fallback to CPU."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA GPU detected: {device_name}")
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Apple MPS (Metal) GPU detected")
                return "mps"
            else:
                logger.info("No GPU detected, using CPU for embeddings")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not available, using CPU")
            return "cpu"

    @property
    def device(self) -> str:
        """Get the device to use for computation."""
        if self._device == "auto":
            self._device = self._detect_device()
        return self._device

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers package not installed. "
                    "Install with: pip install sentence-transformers"
                )

            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully. Embedding dimension: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for a single text."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts efficiently in batch.
        This is the primary performance advantage of local embeddings.
        """
        if not texts:
            return []

        logger.info(f"Batch embedding {len(texts)} texts with batch_size={self.batch_size}")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def supports_batch_embedding(self) -> bool:
        """SentenceTransformers supports efficient batch embedding."""
        return True

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a response using the delegated LLM provider.
        SentenceTransformers doesn't support generation directly.
        """
        if self._llm_provider is None and self._llm_provider_name:
            # Lazy initialize the LLM provider
            self._llm_provider = get_llm_provider(self._llm_provider_name, self._llm_config)

        if self._llm_provider:
            return self._llm_provider.generate(system_prompt, user_prompt)
        else:
            raise NotImplementedError(
                "SentenceTransformersProvider does not support text generation. "
                "Configure a separate llm_provider (openai, anthropic, etc.) in your config."
            )


def get_llm_provider(provider_name: str, config: Dict[str, Any]) -> BaseLLMProvider:
    """Factory function to get LLM provider based on name and config"""
    if provider_name == "openai":
        return OpenAIProvider(
            api_key=config.get("api_key"),
            model=config.get("model", "gpt-4o"),
            embedding_model=config.get("embedding_model", "text-embedding-3-small"),
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=int(config.get("max_tokens")) if "max_tokens" in config else None
        )
    elif provider_name == "anthropic":
        provider = AnthropicProvider(
            api_key=config.get("api_key"),
            model=config.get("model", "claude-3-5-sonnet"),
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=int(config.get("max_tokens")) if "max_tokens" in config else None
        )
        # For embeddings, Anthropic doesn't have its own, so we use OpenAI
        if "openai_api_key" in config:
            provider.set_embedding_provider("openai", api_key=config.get("openai_api_key"))
        return provider
    elif provider_name == "r1":
        return R1Provider(
            api_key=config.get("api_key"),
            model=config.get("model", "rank-r1-v16"),
            embedding_model=config.get("embedding_model", "e5-small-v2"),
            temperature=float(config.get("temperature", 0.7)),
            max_tokens=int(config.get("max_tokens")) if "max_tokens" in config else None
        )
    elif provider_name == "ollama":
        return OllamaProvider(
            model=config.get("model", "mistral"),
            embedding_model=config.get("embedding_model", "nomic-embed-text"),
            temperature=float(config.get("temperature", 0.7)),
            host=config.get("host", "http://localhost:11434")
        )
    elif provider_name in ("sentence-transformers", "local"):
        return SentenceTransformersProvider(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")


def get_embedding_provider(provider_name: str, config: Dict[str, Any]) -> BaseLLMProvider:
    """
    Factory function to get embedding-only provider.
    This allows using a separate provider for embeddings vs LLM generation.

    Args:
        provider_name: Name of the embedding provider
        config: Configuration dictionary

    Returns:
        BaseLLMProvider instance (used only for embeddings)
    """
    if provider_name in ("sentence-transformers", "local"):
        return SentenceTransformersProvider(config)
    else:
        # For other providers, use the standard LLM provider factory
        return get_llm_provider(provider_name, config)