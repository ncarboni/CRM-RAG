"""
Embedding cache for resumable processing of large datasets.
Stores embeddings to disk so processing can be stopped and resumed.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Cache embeddings to disk for resumability.

    Embeddings are stored as numpy arrays in .npy files, organized by
    a hash of the document ID. This allows stopping and resuming
    embedding generation for large datasets.
    """

    def __init__(self, cache_dir: str):
        """
        Initialize the embedding cache.

        Args:
            cache_dir: Directory to store cached embeddings
        """
        self.cache_dir = cache_dir
        self._ensure_dir()

        # In-memory index of cached document IDs for fast lookup
        self._cached_ids: Optional[set] = None

    def _ensure_dir(self):
        """Create cache directory if it doesn't exist."""
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_hash(self, doc_id: str) -> str:
        """Generate a safe filename hash for a document ID."""
        return hashlib.md5(doc_id.encode('utf-8')).hexdigest()

    def _get_path(self, doc_id: str) -> str:
        """Get the cache file path for a document ID."""
        hash_id = self._get_hash(doc_id)
        # Use subdirectories to avoid too many files in one folder
        subdir = hash_id[:2]
        return os.path.join(self.cache_dir, subdir, f"{hash_id}.npy")

    def _get_metadata_path(self) -> str:
        """Get the path to the cache metadata file."""
        return os.path.join(self.cache_dir, "cache_metadata.json")

    def get(self, doc_id: str) -> Optional[List[float]]:
        """
        Get a cached embedding for a document.

        Args:
            doc_id: Document identifier (typically entity URI)

        Returns:
            Embedding as list of floats, or None if not cached
        """
        path = self._get_path(doc_id)
        if os.path.exists(path):
            try:
                embedding = np.load(path)
                return embedding.tolist()
            except Exception as e:
                logger.warning(f"Error loading cached embedding for {doc_id}: {e}")
                return None
        return None

    def set(self, doc_id: str, embedding: List[float]):
        """
        Cache an embedding for a document.

        Args:
            doc_id: Document identifier (typically entity URI)
            embedding: Embedding vector as list of floats
        """
        path = self._get_path(doc_id)
        try:
            # Ensure subdirectory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, np.array(embedding, dtype=np.float32))

            # Invalidate the cached IDs set
            self._cached_ids = None
        except Exception as e:
            logger.error(f"Error caching embedding for {doc_id}: {e}")

    def get_cached_ids(self) -> set:
        """
        Get all cached document IDs.
        Uses in-memory caching for performance.
        """
        if self._cached_ids is not None:
            return self._cached_ids

        cached_ids = set()
        if not os.path.exists(self.cache_dir):
            return cached_ids

        # Load from metadata index file
        metadata_path = self._get_metadata_path()
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    cached_ids = set(data.get('cached_ids', []))
            except Exception as e:
                logger.warning(f"Error loading cache metadata: {e}")

        self._cached_ids = cached_ids
        return cached_ids

    def update_metadata(self, doc_ids: List[str]):
        """
        Update the metadata file with newly cached document IDs.

        Args:
            doc_ids: List of document IDs that were cached
        """
        metadata_path = self._get_metadata_path()
        cached_ids = self.get_cached_ids()
        cached_ids.update(doc_ids)

        try:
            with open(metadata_path, 'w') as f:
                json.dump({'cached_ids': list(cached_ids)}, f)
            self._cached_ids = cached_ids
        except Exception as e:
            logger.error(f"Error updating cache metadata: {e}")

    def count(self) -> int:
        """Get the number of cached embeddings."""
        count = 0
        if not os.path.exists(self.cache_dir):
            return 0

        for subdir in os.listdir(self.cache_dir):
            subdir_path = os.path.join(self.cache_dir, subdir)
            if os.path.isdir(subdir_path) and len(subdir) == 2:
                count += len([f for f in os.listdir(subdir_path) if f.endswith('.npy')])
        return count

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        count = self.count()
        size_bytes = 0

        if os.path.exists(self.cache_dir):
            for root, dirs, files in os.walk(self.cache_dir):
                for file in files:
                    if file.endswith('.npy'):
                        size_bytes += os.path.getsize(os.path.join(root, file))

        return {
            'count': count,
            'size_bytes': size_bytes,
            'size_mb': round(size_bytes / (1024 * 1024), 2),
            'cache_dir': self.cache_dir
        }
