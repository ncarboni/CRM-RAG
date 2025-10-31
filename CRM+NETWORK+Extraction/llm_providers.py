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

class R1Provider(BaseLLMProvider):
    """R1 API provider"""
    
    def __init__(self, api_key: str, model: str = "rank-r1-v16", embedding_model: str = "e5-small-v2",
                 temperature: float = 0.7, max_tokens: Optional[int] = None):
        """Initialize R1 provider"""
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
        self._embeddings = None
        
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using R1 API"""
        import os
        import requests
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
        
        response = requests.post(endpoint, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()["text"]
        else:
            raise Exception(f"R1 API request failed: {response.text}")
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using R1 (Cohere) API"""
        import requests
        
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
        
        response = requests.post(endpoint, headers=headers, json=data)
        
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
    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")