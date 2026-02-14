"""
LLM provider abstraction layer to support multiple AI model providers.
This module provides a unified interface for different LLM APIs.
"""

import json
import logging
import os
import platform
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Semaphore
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


def get_hardware_info() -> Dict[str, Any]:
    """
    Gather hardware information for stats logging.
    Returns CPU, RAM, and GPU details.
    """
    info = {
        'cpu': {
            'model': platform.processor() or 'Unknown',
            'cores_physical': None,
            'cores_logical': None,
        },
        'ram_gb': None,
        'gpu': [],
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'python_version': platform.python_version(),
        }
    }

    # CPU cores
    try:
        import psutil
        info['cpu']['cores_physical'] = psutil.cpu_count(logical=False)
        info['cpu']['cores_logical'] = psutil.cpu_count(logical=True)
        info['ram_gb'] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        # psutil not available, try os
        try:
            info['cpu']['cores_logical'] = os.cpu_count()
        except:
            pass

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info['gpu'].append({
                    'name': props.name,
                    'memory_gb': round(props.total_memory / (1024**3), 1),
                    'compute_capability': f"{props.major}.{props.minor}",
                })
    except ImportError:
        pass

    # Check for Apple Silicon MPS
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['gpu'].append({
                'name': 'Apple Silicon (MPS)',
                'memory_gb': 'shared',
            })
    except:
        pass

    return info

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
                 temperature: float = 0.7, max_tokens: Optional[int] = None,
                 embedding_max_concurrent: int = 10, embedding_retry_attempts: int = 3,
                 embedding_retry_delay: float = 1.0, embedding_chunk_size: int = 100,
                 tokens_per_minute: int = 1_000_000, requests_per_minute: int = 500):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: LLM model name (default: gpt-4o)
            embedding_model: Embedding model name (default: text-embedding-3-small)
            temperature: Temperature for generation (default: 0.7)
            max_tokens: Max tokens for generation (default: None)
            embedding_max_concurrent: Max parallel embedding API requests (default: 10)
            embedding_retry_attempts: Retry attempts per chunk on failure (default: 3)
            embedding_retry_delay: Base delay for exponential backoff in seconds (default: 1.0)
            embedding_chunk_size: Number of texts per API request (default: 100)
            tokens_per_minute: TPM rate limit for embeddings (default: 1,000,000)
            requests_per_minute: RPM rate limit for embeddings (default: 500)
        """
        self.api_key = api_key
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._embeddings = None
        self._llm = None

        # Concurrent embedding configuration
        self.embedding_max_concurrent = embedding_max_concurrent
        self.embedding_retry_attempts = embedding_retry_attempts
        self.embedding_retry_delay = embedding_retry_delay
        self.embedding_chunk_size = embedding_chunk_size
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute

        # Minimum delay between API requests derived from RPM (avoids burst 429s)
        self._min_request_interval = 60.0 / requests_per_minute

        # Persistent TPM tracking across calls
        self._tpm_token_count = 0
        self._tpm_window_start = time.time()
        
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
                openai_api_key=self.api_key,
                chunk_size=2048
            )

        return self._embeddings.embed_query(text)

    def get_embeddings_batch(self, texts: List[str], stats_file: Optional[str] = None) -> List[List[float]]:
        """
        Get embeddings for multiple texts using OpenAI's native batch API.
        This is significantly faster than calling embed_query in a loop.

        OpenAI's embed_documents handles batching internally with chunk_size=1000.

        Args:
            texts: List of texts to embed
            stats_file: Optional path to save embedding stats (JSON format)
        """
        from langchain_openai import OpenAIEmbeddings

        if not texts:
            return []

        # Track stats
        start_time = datetime.now()
        start_timestamp = time.time()
        doc_lengths = [len(t) for t in texts]

        logger.info(f"=== OpenAI Embedding Started ===")
        logger.info(f"Start time: {start_time.isoformat()}")
        logger.info(f"Documents: {len(texts)}")
        logger.info(f"Document lengths: min={min(doc_lengths)}, max={max(doc_lengths)}, "
                   f"avg={sum(doc_lengths)//len(doc_lengths)} chars")
        logger.info(f"Model: {self.embedding_model}")

        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=self.api_key,
                chunk_size=2048
            )

        # embed_documents sends texts in batches to the API (default chunk_size=1000)
        embeddings = self._embeddings.embed_documents(texts)

        # Calculate final stats
        end_time = datetime.now()
        elapsed_seconds = time.time() - start_timestamp
        throughput = len(texts) / elapsed_seconds if elapsed_seconds > 0 else 0

        logger.info(f"=== OpenAI Embedding Completed ===")
        logger.info(f"End time: {end_time.isoformat()}")
        logger.info(f"Total time: {elapsed_seconds:.1f}s")
        logger.info(f"Throughput: {throughput:.2f} docs/sec")

        return embeddings

    def supports_batch_embedding(self) -> bool:
        """OpenAI supports efficient batch embedding via embed_documents."""
        return True

    def supports_concurrent_embedding(self) -> bool:
        """OpenAI supports concurrent batch embedding with ThreadPoolExecutor."""
        return True

    def _get_openai_length_thresholds(self) -> Dict[str, int]:
        """
        Calculate character length thresholds for OpenAI embeddings.

        OpenAI text-embedding-3-small has 8191 token limit per text.
        We use conservative character estimates (~4 chars/token).

        Dynamic batching helps with:
        - Per-request token limits
        - Timeout risk with large payloads
        - Error recovery (smaller batches lose less on failure)
        """
        # OpenAI embedding models max tokens
        max_tokens = 8191  # text-embedding-3-small/large
        chars_per_token = 4  # Conservative estimate
        max_chars = max_tokens * chars_per_token  # ~32K chars

        # Thresholds based on document length
        thresholds = {
            'very_long': int(max_chars * 0.50),   # >16K chars → chunk_size ÷ 8
            'long': int(max_chars * 0.25),        # >8K chars → chunk_size ÷ 4
            'medium': int(max_chars * 0.0625),    # >2K chars → chunk_size ÷ 2
        }

        return thresholds

    def _get_effective_chunk_size(self, max_text_length: int, thresholds: Dict[str, int]) -> int:
        """
        Calculate effective chunk size based on the longest text in the batch.

        Uses relative scaling: chunk sizes are fractions of the configured chunk_size.
        This respects user configuration while adapting to document length.

        Args:
            max_text_length: Length of longest text in potential chunk (in characters)
            thresholds: Dictionary of length thresholds

        Returns:
            Effective chunk size (minimum 1)
        """
        if max_text_length > thresholds['very_long']:
            divisor = 8
        elif max_text_length > thresholds['long']:
            divisor = 4
        elif max_text_length > thresholds['medium']:
            divisor = 2
        else:
            divisor = 1

        effective_size = max(1, self.embedding_chunk_size // divisor)
        return effective_size

    def get_embeddings_batch_concurrent(self, texts: List[str], stats_file: Optional[str] = None) -> List[List[float]]:
        """
        Get embeddings for multiple texts using concurrent API calls with dynamic batching.

        This method provides ~5-10x speedup over sequential processing by:
        1. Sorting texts by length for efficient batching
        2. Using dynamic chunk sizes based on document length
        3. Processing chunks in parallel using ThreadPoolExecutor
        4. Using a Semaphore for rate limiting
        5. Implementing exponential backoff for rate limit errors (429)

        Args:
            texts: List of texts to embed
            stats_file: Optional path to save embedding stats (JSON format)

        Returns:
            List of embeddings in the same order as input texts
        """
        from langchain_openai import OpenAIEmbeddings

        if not texts:
            return []

        # For small batches, use sequential processing
        if len(texts) <= self.embedding_chunk_size:
            return self.get_embeddings_batch(texts, stats_file=stats_file)

        # Track stats
        start_time = datetime.now()
        start_timestamp = time.time()
        doc_lengths = [len(t) for t in texts]

        logger.info(f"=== OpenAI Concurrent Embedding Started ===")
        logger.info(f"Start time: {start_time.isoformat()}")
        logger.info(f"Documents: {len(texts)}")
        logger.info(f"Document lengths: min={min(doc_lengths)}, max={max(doc_lengths)}, "
                   f"avg={sum(doc_lengths)//len(doc_lengths)} chars")
        logger.info(f"Model: {self.embedding_model}")
        logger.info(f"Max concurrent: {self.embedding_max_concurrent}")
        logger.info(f"Base chunk size: {self.embedding_chunk_size}")

        # Initialize embeddings model (lazy initialization for thread safety)
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=self.api_key,
                chunk_size=2048
            )

        # Get length thresholds for dynamic batching
        thresholds = self._get_openai_length_thresholds()

        # Sort texts by length (with original indices) for efficient batching
        indexed_texts = sorted(enumerate(texts), key=lambda x: len(x[1]))

        # Build chunks with dynamic sizing based on document length
        chunks = []  # List of (chunk_id, [(original_idx, text), ...])
        chunk_sizes_used = []
        current_pos = 0
        chunk_id = 0

        while current_pos < len(indexed_texts):
            # Look ahead to find max length in potential chunk
            lookahead_end = min(current_pos + self.embedding_chunk_size, len(indexed_texts))
            max_len_in_chunk = len(indexed_texts[lookahead_end - 1][1])  # Sorted, so last is longest

            # Get effective chunk size based on document length
            effective_chunk_size = self._get_effective_chunk_size(max_len_in_chunk, thresholds)
            chunk_sizes_used.append(effective_chunk_size)

            # Extract chunk
            chunk_end = min(current_pos + effective_chunk_size, len(indexed_texts))
            chunk_items = indexed_texts[current_pos:chunk_end]
            chunks.append((chunk_id, chunk_items))

            current_pos = chunk_end
            chunk_id += 1

        logger.info(f"Dynamic batching: {len(chunks)} chunks, sizes min={min(chunk_sizes_used)}, "
                   f"max={max(chunk_sizes_used)}, avg={sum(chunk_sizes_used)//len(chunk_sizes_used)}")

        logger.info(f"Using concurrent batch embedding: {len(texts)} texts in {len(chunks)} chunks "
                    f"(base_chunk_size={self.embedding_chunk_size}, max_concurrent={self.embedding_max_concurrent})")

        # Semaphore for rate limiting
        semaphore = Semaphore(self.embedding_max_concurrent)

        # Results storage: maps original text index -> embedding
        results: Dict[int, List[float]] = {}
        errors: List[Tuple[int, Exception]] = []

        def process_chunk(chunk_idx: int, chunk_items: List[Tuple[int, str]]) -> Tuple[int, Optional[List[Tuple[int, List[float]]]]]:
            """Process a single chunk with retries and exponential backoff."""
            with semaphore:
                last_exception = None
                chunk_texts = [text for _, text in chunk_items]
                original_indices = [idx for idx, _ in chunk_items]

                for attempt in range(self.embedding_retry_attempts):
                    try:
                        # Create a new embeddings instance for thread safety
                        embeddings = OpenAIEmbeddings(
                            model=self.embedding_model,
                            openai_api_key=self.api_key,
                            chunk_size=2048
                        )
                        result = embeddings.embed_documents(chunk_texts)
                        # Pair embeddings with their original indices
                        indexed_results = list(zip(original_indices, result))
                        return (chunk_idx, indexed_results)
                    except Exception as e:
                        last_exception = e
                        error_str = str(e).lower()

                        # Check for rate limit error (429)
                        is_rate_limit = "429" in error_str or "rate" in error_str or "limit" in error_str

                        if is_rate_limit and attempt < self.embedding_retry_attempts - 1:
                            # Exponential backoff: delay * 2^attempt
                            backoff_time = self.embedding_retry_delay * (2 ** attempt)
                            logger.warning(f"Rate limit hit for chunk {chunk_idx}, "
                                           f"retrying in {backoff_time:.1f}s (attempt {attempt + 1}/{self.embedding_retry_attempts})")
                            time.sleep(backoff_time)
                        elif attempt < self.embedding_retry_attempts - 1:
                            # Other errors - shorter backoff
                            backoff_time = self.embedding_retry_delay * (attempt + 1)
                            logger.warning(f"Error on chunk {chunk_idx}: {e}, "
                                           f"retrying in {backoff_time:.1f}s (attempt {attempt + 1}/{self.embedding_retry_attempts})")
                            time.sleep(backoff_time)

                # All retries exhausted
                logger.error(f"Failed to embed chunk {chunk_idx} after {self.embedding_retry_attempts} attempts: {last_exception}")
                return (chunk_idx, None)

        # Process chunks with RPM + TPM pacing (persistent across calls)
        completed_chunks = 0
        last_submit_time = 0.0

        logger.info(f"Rate limits: {self.requests_per_minute} RPM, {self.tokens_per_minute:,} TPM "
                    f"(min interval: {self._min_request_interval*1000:.0f}ms)")

        with ThreadPoolExecutor(max_workers=self.embedding_max_concurrent) as executor:
            futures = {}
            for chunk_id, chunk_items in chunks:
                chunk_tokens = sum(len(text) // 4 for _, text in chunk_items)

                # TPM gate: reset window if 60s elapsed, otherwise wait if over budget
                now = time.time()
                elapsed = now - self._tpm_window_start
                if elapsed >= 60:
                    self._tpm_token_count = 0
                    self._tpm_window_start = now
                elif self._tpm_token_count + chunk_tokens > self.tokens_per_minute:
                    wait_time = 60 - elapsed + 1
                    logger.info(f"TPM throttle: {self._tpm_token_count:,}+{chunk_tokens:,} > "
                                f"{self.tokens_per_minute:,}, waiting {wait_time:.0f}s")
                    time.sleep(wait_time)
                    self._tpm_token_count = 0
                    self._tpm_window_start = time.time()

                # RPM pacing: wait between submissions to avoid burst 429s
                since_last = time.time() - last_submit_time
                if since_last < self._min_request_interval:
                    time.sleep(self._min_request_interval - since_last)

                self._tpm_token_count += chunk_tokens
                last_submit_time = time.time()
                futures[executor.submit(process_chunk, chunk_id, chunk_items)] = chunk_id

                # Backpressure: drain a completed future before submitting more
                if len(futures) >= self.embedding_max_concurrent:
                    done = next(as_completed(futures))
                    chunk_idx, indexed_embeddings = done.result()
                    completed_chunks += 1
                    del futures[done]
                    if indexed_embeddings is not None:
                        for orig_idx, embedding in indexed_embeddings:
                            results[orig_idx] = embedding
                    else:
                        errors.append((chunk_idx, Exception("Failed after retries")))

            # Collect remaining futures
            for future in as_completed(futures):
                chunk_idx, indexed_embeddings = future.result()
                completed_chunks += 1

                if indexed_embeddings is not None:
                    for orig_idx, embedding in indexed_embeddings:
                        results[orig_idx] = embedding
                    if completed_chunks % 5 == 0 or completed_chunks == len(chunks):
                        logger.info(f"Embedding progress: {completed_chunks}/{len(chunks)} chunks completed")
                else:
                    errors.append((chunk_idx, Exception("Failed after retries")))

        # Check for errors
        if errors:
            failed_indices = [idx for idx, _ in errors]
            logger.error(f"Failed to embed chunks: {failed_indices}")
            raise RuntimeError(f"Failed to embed {len(errors)} chunks: {failed_indices}. "
                               f"Consider reducing embedding_max_concurrent or increasing retry_attempts.")

        # Reassemble results in original order
        all_embeddings = [results[i] for i in range(len(texts))]

        # Calculate final stats
        end_time = datetime.now()
        elapsed_seconds = time.time() - start_timestamp
        throughput = len(texts) / elapsed_seconds if elapsed_seconds > 0 else 0

        logger.info(f"=== OpenAI Concurrent Embedding Completed ===")
        logger.info(f"End time: {end_time.isoformat()}")
        logger.info(f"Total time: {elapsed_seconds:.1f}s")
        logger.info(f"Documents: {len(all_embeddings)}")
        logger.info(f"Throughput: {throughput:.2f} docs/sec")
        logger.info(f"Chunks processed: {len(chunks)} (dynamic sizes: {min(chunk_sizes_used)}-{max(chunk_sizes_used)})")

        return all_embeddings


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

    def get_embeddings_batch(self, texts: List[str], stats_file: Optional[str] = None) -> List[List[float]]:
        """
        Get embeddings for multiple texts using OpenAI's native batch API.
        Claude doesn't have embeddings, so we use OpenAI's embed_documents.

        Args:
            texts: List of texts to embed
            stats_file: Optional path to save embedding stats (JSON format)
        """
        from langchain_openai import OpenAIEmbeddings

        if not texts:
            return []

        if self.openai_api_key is None:
            raise ValueError("OpenAI API key not set for embeddings. Use set_embedding_provider().")

        # Track stats
        start_time = datetime.now()
        start_timestamp = time.time()
        doc_lengths = [len(t) for t in texts]

        logger.info(f"=== Anthropic (OpenAI) Embedding Started ===")
        logger.info(f"Start time: {start_time.isoformat()}")
        logger.info(f"Documents: {len(texts)}")
        logger.info(f"Document lengths: min={min(doc_lengths)}, max={max(doc_lengths)}, "
                   f"avg={sum(doc_lengths)//len(doc_lengths)} chars")
        logger.info(f"Model: text-embedding-3-small (via OpenAI)")

        if self._openai_embeddings is None:
            self._openai_embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self.openai_api_key
            )

        embeddings = self._openai_embeddings.embed_documents(texts)

        # Calculate final stats
        end_time = datetime.now()
        elapsed_seconds = time.time() - start_timestamp
        throughput = len(texts) / elapsed_seconds if elapsed_seconds > 0 else 0

        logger.info(f"=== Anthropic (OpenAI) Embedding Completed ===")
        logger.info(f"End time: {end_time.isoformat()}")
        logger.info(f"Total time: {elapsed_seconds:.1f}s")
        logger.info(f"Throughput: {throughput:.2f} docs/sec")

        return embeddings

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

        # Create a secure session that doesn't use .netrc credentials
        # This prevents CVE-related .netrc credential leaks
        self._session = requests.Session()
        self._session.trust_env = False
        
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using R1 API"""
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

    def get_embeddings_batch(self, texts: List[str], stats_file: Optional[str] = None) -> List[List[float]]:
        """
        Get embeddings for multiple texts efficiently in batch.

        Uses dynamic batching based on text length to prevent OOM errors.
        Longer texts get smaller batch sizes since attention memory is O(batch × seq²).

        The algorithm:
        1. Sorts texts by length to group similar-sized documents
        2. Derives thresholds from the model's max_seq_length (model-aware)
        3. Uses relative batch sizes (fractions of configured batch_size)
        4. Clears GPU cache after processing long documents

        Args:
            texts: List of texts to embed
            stats_file: Optional path to save embedding stats (JSON format)

        See docs/TECHNICAL_REPORT.md for detailed explanation.
        """
        if not texts:
            return []

        # Track stats
        start_time = datetime.now()
        start_timestamp = time.time()
        batch_sizes_used = []
        doc_lengths = [len(t) for t in texts]

        # Log start
        logger.info(f"=== Embedding Started ===")
        logger.info(f"Start time: {start_time.isoformat()}")
        logger.info(f"Documents: {len(texts)}")
        logger.info(f"Document lengths: min={min(doc_lengths)}, max={max(doc_lengths)}, "
                   f"avg={sum(doc_lengths)//len(doc_lengths)} chars")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Base batch size: {self.batch_size}")
        logger.info(f"Device: {self.device}")

        # Get model-aware thresholds
        thresholds = self._get_length_thresholds()

        # Sort texts by length, keeping track of original indices
        indexed_texts = [(i, text, len(text)) for i, text in enumerate(texts)]
        indexed_texts.sort(key=lambda x: x[2])  # Sort by length

        # Process in length-adaptive batches
        all_embeddings = [None] * len(texts)  # Preallocate for reordering
        current_batch_start = 0
        total_batches = 0

        while current_batch_start < len(indexed_texts):
            # Determine batch size based on max text length in potential batch
            max_len_in_batch = indexed_texts[min(current_batch_start + self.batch_size - 1,
                                                   len(indexed_texts) - 1)][2]

            # Get effective batch size using relative scaling
            effective_batch_size = self._get_effective_batch_size(max_len_in_batch, thresholds)
            batch_sizes_used.append(effective_batch_size)

            # Extract batch
            batch_end = min(current_batch_start + effective_batch_size, len(indexed_texts))
            batch_items = indexed_texts[current_batch_start:batch_end]
            batch_texts = [item[1] for item in batch_items]
            batch_indices = [item[0] for item in batch_items]

            # Log batch info
            batch_max_len = max(len(t) for t in batch_texts)
            logger.info(f"Processing batch {total_batches + 1} of {len(batch_texts)} texts "
                       f"(max_len={batch_max_len} chars, batch_size={effective_batch_size})")

            # Encode batch
            batch_embeddings = self.model.encode(
                batch_texts,
                batch_size=effective_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False  # Disable per-batch progress bar
            )

            # Place embeddings back in original order
            for idx, embedding in zip(batch_indices, batch_embeddings):
                all_embeddings[idx] = embedding.tolist()

            # Clear GPU cache after processing long documents (>25% of max)
            if max_len_in_batch > thresholds['medium']:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            current_batch_start = batch_end
            total_batches += 1

        # Calculate final stats
        end_time = datetime.now()
        elapsed_seconds = time.time() - start_timestamp
        throughput = len(texts) / elapsed_seconds if elapsed_seconds > 0 else 0

        # Log summary
        logger.info(f"=== Embedding Completed ===")
        logger.info(f"End time: {end_time.isoformat()}")
        logger.info(f"Total time: {elapsed_seconds:.1f}s")
        logger.info(f"Documents: {len(texts)}")
        logger.info(f"Throughput: {throughput:.2f} docs/sec")
        logger.info(f"Batches: {total_batches}")
        logger.info(f"Batch sizes: min={min(batch_sizes_used)}, max={max(batch_sizes_used)}, "
                   f"avg={sum(batch_sizes_used)//len(batch_sizes_used)}")

        # Build stats dict
        stats = self._build_embedding_stats(
            start_time=start_time,
            end_time=end_time,
            elapsed_seconds=elapsed_seconds,
            num_documents=len(texts),
            doc_lengths=doc_lengths,
            batch_sizes_used=batch_sizes_used,
            total_batches=total_batches,
            throughput=throughput,
        )

        # Save stats file if requested
        if stats_file:
            self._save_embedding_stats(stats, stats_file)

        return all_embeddings

    def _build_embedding_stats(
        self,
        start_time: datetime,
        end_time: datetime,
        elapsed_seconds: float,
        num_documents: int,
        doc_lengths: List[int],
        batch_sizes_used: List[int],
        total_batches: int,
        throughput: float,
    ) -> Dict[str, Any]:
        """Build a comprehensive stats dictionary."""
        # Get model info
        try:
            max_seq_length = self.model.max_seq_length
            embedding_dim = self.model.get_sentence_embedding_dimension()
        except:
            max_seq_length = None
            embedding_dim = None

        # Get hardware info (with fallback if it fails)
        try:
            hardware_info = get_hardware_info()
        except Exception as e:
            logger.warning(f"Failed to get hardware info: {e}")
            hardware_info = {'error': str(e)}

        return {
            'timing': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'elapsed_seconds': round(elapsed_seconds, 2),
            },
            'documents': {
                'count': num_documents,
                'length_chars': {
                    'min': min(doc_lengths),
                    'max': max(doc_lengths),
                    'avg': sum(doc_lengths) // len(doc_lengths),
                    'total': sum(doc_lengths),
                },
            },
            'batching': {
                'base_batch_size': self.batch_size,
                'total_batches': total_batches,
                'effective_batch_sizes': {
                    'min': min(batch_sizes_used),
                    'max': max(batch_sizes_used),
                    'avg': round(sum(batch_sizes_used) / len(batch_sizes_used), 1),
                },
            },
            'performance': {
                'throughput_docs_per_sec': round(throughput, 2),
                'avg_time_per_doc_ms': round((elapsed_seconds / num_documents) * 1000, 2) if num_documents > 0 else 0,
            },
            'model': {
                'name': self.model_name,
                'max_seq_length': max_seq_length,
                'embedding_dimension': embedding_dim,
            },
            'hardware': hardware_info,
        }

    def _save_embedding_stats(self, stats: Dict[str, Any], file_path: str) -> None:
        """Save embedding stats to a JSON file."""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                json.dump(stats, f, indent=2)

            logger.info(f"Embedding stats saved to: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to save embedding stats: {e}")

    def _get_length_thresholds(self) -> Dict[str, int]:
        """
        Calculate character length thresholds based on model's max sequence length.

        Returns thresholds as fractions of the model's maximum capacity.
        This makes the batching strategy model-agnostic.
        """
        # Get model's max sequence length (in tokens)
        try:
            max_seq_length = self.model.max_seq_length
        except AttributeError:
            # Fallback for models that don't expose this
            max_seq_length = 512
            logger.warning(f"Model doesn't expose max_seq_length, using default: {max_seq_length}")

        # Convert to characters (conservative estimate: ~4 chars per token)
        # This ratio varies by language/content but 4 is a safe average
        chars_per_token = 4
        max_chars = max_seq_length * chars_per_token

        # Define thresholds as fractions of max capacity
        # These fractions determine when to reduce batch size
        thresholds = {
            'very_long': int(max_chars * 0.75),   # 75% of max → batch_size ÷ 16
            'long': int(max_chars * 0.50),        # 50% of max → batch_size ÷ 8
            'medium_long': int(max_chars * 0.25), # 25% of max → batch_size ÷ 4
            'medium': int(max_chars * 0.125),     # 12.5% of max → batch_size ÷ 2
        }

        logger.debug(f"Model max_seq_length: {max_seq_length} tokens, "
                    f"thresholds (chars): {thresholds}")

        return thresholds

    def _get_effective_batch_size(self, max_text_length: int, thresholds: Dict[str, int]) -> int:
        """
        Calculate effective batch size based on the longest text in the batch.

        Uses relative scaling: batch sizes are fractions of the configured batch_size.
        This respects the user's configuration while adapting to document length.

        Args:
            max_text_length: Length of longest text in potential batch (in characters)
            thresholds: Dictionary of length thresholds from _get_length_thresholds()

        Returns:
            Effective batch size (minimum 1)
        """
        if max_text_length > thresholds['very_long']:
            # Very long documents: use 1/16 of configured batch size
            divisor = 16
        elif max_text_length > thresholds['long']:
            # Long documents: use 1/8 of configured batch size
            divisor = 8
        elif max_text_length > thresholds['medium_long']:
            # Medium-long documents: use 1/4 of configured batch size
            divisor = 4
        elif max_text_length > thresholds['medium']:
            # Medium documents: use 1/2 of configured batch size
            divisor = 2
        else:
            # Short documents: use full configured batch size
            divisor = 1

        effective_size = max(1, self.batch_size // divisor)

        return effective_size

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
            max_tokens=int(config.get("max_tokens")) if "max_tokens" in config else None,
            embedding_max_concurrent=int(config.get("embedding_max_concurrent", 10)),
            embedding_retry_attempts=int(config.get("embedding_retry_attempts", 3)),
            embedding_retry_delay=float(config.get("embedding_retry_delay", 1.0)),
            embedding_chunk_size=int(config.get("embedding_chunk_size", 100)),
            tokens_per_minute=int(config.get("tokens_per_minute", 1_000_000)),
            requests_per_minute=int(config.get("requests_per_minute", 500))
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