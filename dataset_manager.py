"""
Dataset Manager for multi-dataset RAG system support.
Manages multiple RAG system instances with lazy loading.
"""

import logging
import os
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages multiple RAG system instances with lazy loading"""

    def __init__(self, datasets_config: Dict[str, Any], llm_config: Dict[str, Any]):
        """
        Initialize the DatasetManager.

        Args:
            datasets_config: Configuration from datasets.yaml containing 'datasets' dict
                             and optional 'default_dataset'
            llm_config: LLM configuration from .env files
        """
        self.datasets = datasets_config.get('datasets', {})
        self.llm_config = llm_config
        self._rag_systems: Dict[str, Any] = {}  # Lazy-loaded RAG instances

        logger.info(f"DatasetManager initialized with {len(self.datasets)} datasets")

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        Return list of available datasets with their status.

        Returns:
            List of dataset info dictionaries
        """
        result = []
        for dataset_id, dataset_config in self.datasets.items():
            result.append({
                "id": dataset_id,
                "name": dataset_config.get("name", dataset_id),
                "display_name": dataset_config.get("display_name", dataset_id),
                "description": dataset_config.get("description", ""),
                "endpoint": dataset_config.get("endpoint", ""),
                "initialized": self.is_initialized(dataset_id),
                "has_cache": self._has_cache(dataset_id)
            })
        return result

    def get_dataset(self, dataset_id: str) -> Any:
        """
        Get or lazy-initialize a RAG system for the specified dataset.

        Args:
            dataset_id: The dataset identifier

        Returns:
            UniversalRagSystem instance for the dataset

        Raises:
            ValueError: If dataset_id is not found
        """
        if dataset_id not in self.datasets:
            raise ValueError(f"Dataset '{dataset_id}' not found. Available: {list(self.datasets.keys())}")

        # Return cached instance if already initialized
        if dataset_id in self._rag_systems:
            logger.info(f"Returning cached RAG system for dataset: {dataset_id}")
            return self._rag_systems[dataset_id]

        # Lazy initialize the RAG system
        logger.info(f"Initializing RAG system for dataset: {dataset_id}")
        dataset_config = self.datasets[dataset_id]

        # Import here to avoid circular imports
        from universal_rag_system import UniversalRagSystem

        # Create config with dataset-specific settings
        config = self.llm_config.copy()

        # Get SPARQL endpoint from dataset config (required)
        endpoint = dataset_config.get('endpoint')
        if not endpoint:
            raise ValueError(f"Dataset '{dataset_id}' is missing required 'endpoint' configuration")

        # Merge dataset-specific embedding configuration
        embedding_config = dataset_config.get('embedding', {})
        if embedding_config:
            logger.info(f"Dataset '{dataset_id}' has custom embedding config: {embedding_config}")
            if 'provider' in embedding_config:
                config['embedding_provider'] = embedding_config['provider']
            if 'model' in embedding_config:
                config['embedding_model'] = embedding_config['model']
            if 'batch_size' in embedding_config:
                config['embedding_batch_size'] = embedding_config['batch_size']
            if 'device' in embedding_config:
                config['embedding_device'] = embedding_config['device']
            if 'use_cache' in embedding_config:
                config['use_embedding_cache'] = embedding_config['use_cache']

        # Create and initialize the RAG system
        rag_system = UniversalRagSystem(
            endpoint_url=endpoint,
            config=config,
            dataset_id=dataset_id,
            dataset_config=dataset_config
        )

        # Initialize the system (load or build cache)
        if not rag_system.initialize():
            raise RuntimeError(f"Failed to initialize RAG system for dataset: {dataset_id}")

        # Cache the instance
        self._rag_systems[dataset_id] = rag_system
        logger.info(f"Successfully initialized RAG system for dataset: {dataset_id}")

        return rag_system

    def is_initialized(self, dataset_id: str) -> bool:
        """
        Check if a dataset is loaded in memory.

        Args:
            dataset_id: The dataset identifier

        Returns:
            True if the RAG system is initialized in memory
        """
        return dataset_id in self._rag_systems

    def _has_cache(self, dataset_id: str) -> bool:
        """
        Check if a dataset has cached data on disk.

        Args:
            dataset_id: The dataset identifier

        Returns:
            True if cache files exist for this dataset
        """
        paths = self.get_cache_paths(dataset_id)
        return (
            os.path.exists(paths['document_graph']) and
            os.path.exists(paths['vector_index'])
        )

    def get_cache_paths(self, dataset_id: str) -> Dict[str, str]:
        """
        Return dataset-specific cache paths.

        Args:
            dataset_id: The dataset identifier

        Returns:
            Dictionary with paths for 'cache_dir', 'document_graph', 'vector_index', 'documents_dir'
        """
        cache_dir = f'data/cache/{dataset_id}'
        return {
            'cache_dir': cache_dir,
            'document_graph': f'{cache_dir}/document_graph.pkl',
            'vector_index': f'{cache_dir}/vector_index/index.faiss',
            'vector_index_dir': f'{cache_dir}/vector_index',
            'documents_dir': f'data/documents/{dataset_id}/entity_documents'
        }

    def get_interface_config(self, dataset_id: str, default_interface: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge default interface config with dataset-specific overrides.

        Args:
            dataset_id: The dataset identifier
            default_interface: Default interface configuration from interface.yaml

        Returns:
            Merged interface configuration
        """
        if dataset_id not in self.datasets:
            return default_interface

        dataset_config = self.datasets[dataset_id]
        dataset_interface = dataset_config.get('interface', {})

        # Deep merge: start with defaults, override with dataset-specific values
        merged = default_interface.copy()

        for key, value in dataset_interface.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                # Merge nested dicts (like 'about' section)
                merged[key] = {**merged[key], **value}
            else:
                # Override scalar values and lists
                merged[key] = value

        return merged

