"""
Universal RAG system with graph-based document retrieval.
This system can be applied to any RDF dataset and uses coherent subgraph extraction
to enhance document retrieval using CIDOC-CRM relationship weights.
"""

# Standard library imports
import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import quote

import yaml

# Third-party imports
import numpy as np
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON, TSV, POST

# Langchain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# Third-party data fetching
import requests

# Local imports
from crm_rag import PROJECT_ROOT
from crm_rag.document_store import GraphDocumentStore
from crm_rag.llm_providers import get_llm_provider, get_embedding_provider
from crm_rag.embedding_cache import EmbeddingCache
from scripts.extract_ontology_labels import run_extraction
from crm_rag.fr_traversal import FRTraversal, classify_satellite
from crm_rag.document_formatter import (
    is_schema_predicate as _is_schema_predicate,
    is_technical_class_name as _is_technical_class_name,
    get_relationship_weight as _get_relationship_weight,
)
from crm_rag.sparql_helpers import BatchSparqlClient

logger = logging.getLogger(__name__)

from dataclasses import dataclass

@dataclass
class QueryAnalysis:
    """Result of LLM-based query analysis."""
    query_type: str  # SPECIFIC, ENUMERATION, or AGGREGATION
    categories: List[str]  # Target FC categories: Thing, Actor, Place, Event, Concept, Time

QUERY_ANALYSIS_PROMPT = """Classify this question about cultural heritage data.

Query type (pick one):
- SPECIFIC: asks about a particular entity ("tell me about X", "where is X")
- ENUMERATION: asks to list entities ("which paintings…", "list all…", "what artists…")
- AGGREGATION: asks to count or rank ("how many…", "top 10…", "most…")

Target categories (pick 1-3 from this list):
- Thing (objects, artworks, buildings, features, inscriptions)
- Actor (people, artists, groups, organizations)
- Place (locations, cities, regions)
- Event (activities, creation, production, exhibitions)
- Concept (types, materials, techniques)
- Time (dates, periods)

Return ONLY valid JSON with no extra text: {{"query_type": "...", "categories": ["..."]}}

Question: "{question}"
"""


class RetrievalConfig:
    """Configuration constants for the RAG retrieval system"""

    # Score combination weights
    RELEVANCE_CONNECTIVITY_ALPHA = 0.7  # Weight for combining relevance and connectivity scores

    # Rate limiting
    TOKENS_PER_MINUTE_LIMIT = 950_000  # Token limit for rate limiting (TPM)

    # Retrieval parameters
    DEFAULT_RETRIEVAL_K = 10  # Default number of documents to retrieve
    SPECIFIC_K = 10            # k for specific entity queries
    ENUMERATION_K = 20         # k for listing/enumeration queries
    AGGREGATION_K = 25         # k for counting/ranking queries
    POOL_MULTIPLIER = 6        # initial_pool_size = k * POOL_MULTIPLIER

    # Processing parameters
    DEFAULT_BATCH_SIZE = 50  # Default batch size for processing entities
    ENTITY_CONTEXT_DEPTH = 2  # Depth for entity context traversal
    MAX_ADJACENCY_HOPS = 2  # Maximum hops for adjacency matrix construction

    # Batch SPARQL query parameters
    BATCH_QUERY_SIZE = 1000  # Number of URIs per VALUES clause in batch queries
    BATCH_QUERY_TIMEOUT = 60000  # Timeout for batch queries in ms (60 seconds)
    BATCH_QUERY_RETRY_SIZE = 100  # Smaller batch size for retry on timeout

    # Diversity penalty (MMR-style) for coherent subgraph extraction
    DIVERSITY_PENALTY = 0.2  # Weight for penalizing embedding similarity to already-selected docs

    # Graph context enrichment for LLM prompt
    GRAPH_CONTEXT_MAX_NEIGHBORS = 5   # Max neighbor relationships to show per entity
    GRAPH_CONTEXT_MAX_LINES = 50      # Total line cap for graph context section

    # Type-based score modifiers for retrieval ranking
    # Positive = boost, negative = penalty. Applied as multiplier: score * (1 + modifier)
    # Keys match the human-readable type stored in doc.metadata["type"]
    # and also checked against doc.metadata["all_types"] for E-coded variants
    TYPE_SCORE_MODIFIERS = {
        # Primary entity types - slight boost
        "E22_Man-Made_Object": 0.05,
        "E22_Human-Made_Object": 0.05,
        "Human-Made Object": 0.05,
        "Man-Made Object": 0.05,
        "E24_Physical_Human-Made_Thing": 0.05,
        "Physical Human-Made Thing": 0.05,
        "Physical Man-Made Thing": 0.05,
        "E39_Actor": 0.05,
        "Actor": 0.05,
        "E21_Person": 0.05,
        "Person": 0.05,
        "E74_Group": 0.05,
        "Group": 0.05,
        "E7_Activity": 0.03,
        "Activity": 0.03,
        "E5_Event": 0.03,
        "Event": 0.03,
        "E53_Place": 0.03,
        "Place": 0.03,
        "E18_Physical_Thing": 0.03,
        "Physical Thing": 0.03,
        # Non-informative types - penalty
        "Linguistic Object": -0.15,
        "E33_Linguistic_Object": -0.15,
        "E33_E41_Linguistic_Appellation": -0.15,
        "E41_E33_Linguistic_Appellation": -0.15,
        "Linguistic Appellation": -0.15,
        "Inscription": -0.15,
        "E34_Inscription": -0.15,
        "E31_Document": -0.10,
        "Document": -0.10,
        "Appellation": -0.10,
        "E41_Appellation": -0.10,
    }


class UniversalRagSystem:
    """Universal RAG system with graph-based document retrieval"""

    # Class-level cache for property labels and ontology classes
    _property_labels = None
    _ontology_classes = None
    _class_labels = None  # Cache for class URI -> English label mapping
    _inverse_properties = None  # Cache for property URI -> inverse property URI mapping
    _event_classes = None  # Cache for event class URIs (loaded from config/event_classes.json)
    _extraction_attempted = False  # Track if we've tried extraction to avoid infinite loops
    _missing_properties = set()  # Track properties that couldn't be found
    _missing_classes = set()  # Track classes that couldn't be found in ontology files

    def __init__(self, endpoint_url, config=None, dataset_id=None, data_dir=None, dataset_config=None):
        """
        Initialize the universal RAG system.

        Args:
            endpoint_url: SPARQL endpoint URL
            config: Configuration dictionary for LLM provider
            dataset_id: Optional dataset identifier for multi-dataset support.
                        Used to create dataset-specific cache directories.
            data_dir: Optional override for data directory (e.g., for cluster storage).
                      If not provided, uses 'data/' relative to current directory.
            dataset_config: Optional dataset configuration dict from datasets.yaml.
                           Used to access dataset-specific settings like image SPARQL queries.
        """
        self.endpoint_url = endpoint_url
        self.dataset_id = dataset_id or "default"
        self.data_dir = data_dir  # None means use default 'data/' directory
        self.dataset_config = dataset_config or {}
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)
        self.batch_sparql = BatchSparqlClient(self.sparql)

        # Reset per-dataset tracking sets to avoid cross-dataset contamination
        # These track missing ontology elements for validation reports
        UniversalRagSystem._missing_properties = set()
        UniversalRagSystem._missing_classes = set()

        # Initialize configuration
        self.config = config or {}

        # Initialize LLM provider (for text generation)
        provider_name = self.config.get("llm_provider", "openai")
        try:
            self.llm_provider = get_llm_provider(provider_name, self.config)
        except Exception as e:
            logger.error(f"Error initializing LLM provider: {str(e)}")
            raise

        # Initialize embedding provider (can be different from LLM provider)
        embedding_provider_name = self.config.get("embedding_provider", provider_name)
        if embedding_provider_name != provider_name:
            logger.info(f"Using separate embedding provider: {embedding_provider_name}")
            try:
                self.embedding_provider = get_embedding_provider(embedding_provider_name, self.config)
            except Exception as e:
                logger.warning(f"Error initializing embedding provider: {str(e)}, falling back to LLM provider")
                self.embedding_provider = self.llm_provider
        else:
            self.embedding_provider = self.llm_provider

        # Check if batch embedding is supported
        self.use_batch_embedding = (
            hasattr(self.embedding_provider, 'supports_batch_embedding') and
            self.embedding_provider.supports_batch_embedding()
        )
        if self.use_batch_embedding:
            logger.info("Batch embedding is supported and enabled")

        # Initialize embedding cache for resumability
        self.use_embedding_cache = self.config.get("use_embedding_cache", True)
        if self.use_embedding_cache:
            cache_dir = os.path.join(self._get_cache_dir(), "embeddings")
            self.embedding_cache = EmbeddingCache(cache_dir)
            logger.info(f"Embedding cache enabled at {cache_dir}")
        else:
            self.embedding_cache = None
            logger.info("Embedding cache disabled")

        # Initialize document store
        self.document_store = None

        # Create a secure session for HTTP requests (e.g., Wikidata API)
        # Disable trust_env to prevent .netrc credential leaks (CVE fix)
        self._http_session = requests.Session()
        self._http_session.trust_env = False

        # Load property labels and ontology classes from ontology extraction (cached at class level)
        if UniversalRagSystem._property_labels is None:
            UniversalRagSystem._property_labels = self._load_property_labels()

        if UniversalRagSystem._ontology_classes is None:
            UniversalRagSystem._ontology_classes = self._load_ontology_classes()

        if UniversalRagSystem._class_labels is None:
            UniversalRagSystem._class_labels = self._load_class_labels()

        if UniversalRagSystem._inverse_properties is None:
            UniversalRagSystem._inverse_properties = self._load_inverse_properties()

        # Initialize FR traversal for FR-based document generation
        self.fr_traversal = self._init_fr_traversal()

    # ==================== Path Helper Methods ====================
    # These methods return dataset-specific paths for multi-dataset support

    def _get_cache_dir(self) -> str:
        """Return the cache directory for this dataset."""
        base = self.data_dir if self.data_dir else str(PROJECT_ROOT / 'data')
        return f'{base}/cache/{self.dataset_id}'

    def _get_document_graph_path(self) -> str:
        """Return the document graph pickle file path for this dataset."""
        return f'{self._get_cache_dir()}/document_graph.pkl'

    def _get_document_graph_temp_path(self) -> str:
        """Return the temporary document graph pickle file path for this dataset."""
        return f'{self._get_cache_dir()}/document_graph_temp.pkl'

    def _get_vector_index_dir(self) -> str:
        """Return the vector index directory for this dataset."""
        return f'{self._get_cache_dir()}/vector_index'

    def _get_vector_index_path(self) -> str:
        """Return the vector index file path for this dataset."""
        return f'{self._get_vector_index_dir()}/index.faiss'

    def _get_bm25_index_dir(self) -> str:
        """Return the BM25 index directory for this dataset."""
        return f'{self._get_cache_dir()}/bm25_index'

    def _get_documents_dir(self) -> str:
        """Return the entity documents directory for this dataset."""
        base = self.data_dir if self.data_dir else str(PROJECT_ROOT / 'data')
        return f'{base}/documents/{self.dataset_id}/entity_documents'

    # ==================== End Path Helper Methods ====================

    def _load_inverse_properties(self):
        """
        Load inverse property mappings from JSON file generated from ontologies.

        Returns:
            dict: Inverse property mappings (property URI -> inverse URI)
        """
        inverse_file = str(PROJECT_ROOT / 'data' / 'labels' / 'inverse_properties.json')

        if os.path.exists(inverse_file):
            try:
                with open(inverse_file, 'r', encoding='utf-8') as f:
                    inverse_map = json.load(f)
                logger.info(f"Loaded {len(inverse_map)} inverse property mappings from {inverse_file}")
                return inverse_map
            except Exception as e:
                logger.error(f"Error loading inverse properties: {str(e)}")
                return {}
        else:
            logger.warning(f"Inverse properties file not found at {inverse_file}")
            logger.info("Run: python scripts/extract_ontology_labels.py to generate it")
            return {}

    def _init_fr_traversal(self) -> Optional[FRTraversal]:
        """Initialize FR traversal module with required config files.

        Returns FRTraversal instance or None if config files are missing.
        """
        fr_json = str(PROJECT_ROOT / 'config' / 'fundamental_relationships_cidoc_crm.json')
        inverse_props = str(PROJECT_ROOT / 'data' / 'labels' / 'inverse_properties.json')
        fc_mapping = str(PROJECT_ROOT / 'config' / 'fc_class_mapping.json')

        if not os.path.exists(fr_json):
            logger.warning(f"FR JSON not found: {fr_json} — FR traversal disabled")
            return None
        if not os.path.exists(inverse_props):
            logger.warning(f"Inverse properties not found: {inverse_props} — FR traversal disabled")
            return None
        if not os.path.exists(fc_mapping):
            logger.warning(f"FC class mapping not found: {fc_mapping} — FR traversal disabled")
            return None

        traversal = FRTraversal(
            fr_json_path=fr_json,
            inverse_properties_path=inverse_props,
            fc_mapping_path=fc_mapping,
            property_labels=UniversalRagSystem._property_labels
        )
        logger.info("FR traversal initialized for document generation")
        return traversal

    def _load_property_labels(self, force_extract=False):
        """
        Load property labels from JSON file generated from ontologies.
        Automatically extracts labels from ontology files if JSON doesn't exist.

        Args:
            force_extract: If True, force re-extraction even if JSON exists

        Returns:
            dict: Property labels mapping
        """
        labels_file = str(PROJECT_ROOT / 'data' / 'labels' / 'property_labels.json')
        ontology_dir = str(PROJECT_ROOT / 'data' / 'ontologies')

        # Check if we need to extract
        should_extract = force_extract or not os.path.exists(labels_file)

        if should_extract:
            # Check if ontology directory exists
            if not os.path.exists(ontology_dir):
                logger.error(f"Ontology directory not found at '{ontology_dir}'")
                logger.error("Cannot extract property labels without ontology files")
                return {}

            # Check if ontology files exist
            ontology_files = [f for f in os.listdir(ontology_dir) if f.endswith(('.ttl', '.rdf', '.owl', '.n3'))]
            if not ontology_files:
                logger.error(f"No ontology files found in '{ontology_dir}'")
                logger.error("Add CIDOC-CRM, VIR, CRMdig ontology files to the ontology directory")
                return {}

            # Run extraction
            logger.info("Extracting property labels from ontology files...")
            logger.info(f"Found {len(ontology_files)} ontology files: {', '.join(ontology_files)}")

            try:
                success = run_extraction(ontology_dir, labels_file)
                if not success:
                    logger.error("Failed to extract property labels from ontologies")
                    return {}
                else:
                    logger.info(f"✓ Successfully extracted property labels to {labels_file}")
                    UniversalRagSystem._extraction_attempted = True
            except Exception as e:
                logger.error(f"Error during property label extraction: {str(e)}")
                return {}

        # Load the JSON file
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                logger.info(f"Loaded {len(labels)} property labels from {labels_file}")
                return labels
            except Exception as e:
                logger.error(f"Error loading property labels from {labels_file}: {str(e)}")
                return {}
        else:
            logger.error(f"Property labels file not found at {labels_file}")
            return {}

    def _load_ontology_classes(self, force_extract=False):
        """
        Load ontology classes from JSON file generated from ontologies.
        Automatically extracts classes from ontology files if JSON doesn't exist.

        Args:
            force_extract: If True, force re-extraction even if JSON exists

        Returns:
            set: Set of ontology class names (both URIs and local names)
        """
        classes_file = str(PROJECT_ROOT / 'data' / 'labels' / 'ontology_classes.json')
        ontology_dir = str(PROJECT_ROOT / 'data' / 'ontologies')

        # Check if we need to extract (the extraction is done together with properties)
        should_extract = force_extract or not os.path.exists(classes_file)

        if should_extract:
            # Check if ontology directory exists
            if not os.path.exists(ontology_dir):
                logger.error(f"Ontology directory not found at '{ontology_dir}'")
                logger.error("Cannot extract ontology classes without ontology files")
                return set()

            # Check if ontology files exist
            ontology_files = [f for f in os.listdir(ontology_dir) if f.endswith(('.ttl', '.rdf', '.owl', '.n3'))]
            if not ontology_files:
                logger.error(f"No ontology files found in '{ontology_dir}'")
                logger.error("Add CIDOC-CRM, VIR, CRMdig ontology files to the ontology directory")
                return set()

            # Run extraction (this extracts both properties and classes)
            logger.info("Extracting ontology classes from ontology files...")
            logger.info(f"Found {len(ontology_files)} ontology files: {', '.join(ontology_files)}")

            try:
                # This will extract both properties and classes
                labels_file = str(PROJECT_ROOT / 'data' / 'labels' / 'property_labels.json')
                success = run_extraction(ontology_dir, labels_file, classes_file)
                if not success:
                    logger.error("Failed to extract ontology classes from ontologies")
                    return set()
                else:
                    logger.info(f"✓ Successfully extracted ontology classes to {classes_file}")
            except Exception as e:
                logger.error(f"Error during ontology class extraction: {str(e)}")
                return set()

        # Load the JSON file
        if os.path.exists(classes_file):
            try:
                with open(classes_file, 'r', encoding='utf-8') as f:
                    classes_list = json.load(f)
                classes_set = set(classes_list)
                logger.info(f"Loaded {len(classes_set)} ontology classes from {classes_file}")
                return classes_set
            except Exception as e:
                logger.error(f"Error loading ontology classes from {classes_file}: {str(e)}")
                return set()
        else:
            logger.error(f"Ontology classes file not found at {classes_file}")
            return set()

    def _load_class_labels(self, force_extract=False):
        """
        Load class labels (URI -> English label mapping) from JSON file generated from ontologies.
        Automatically extracts labels from ontology files if JSON doesn't exist.

        Args:
            force_extract: If True, force re-extraction even if JSON exists

        Returns:
            dict: Class labels mapping (URI -> English label)
        """
        labels_file = str(PROJECT_ROOT / 'data' / 'labels' / 'class_labels.json')
        ontology_dir = str(PROJECT_ROOT / 'data' / 'ontologies')

        # Check if we need to extract (the extraction is done together with classes)
        should_extract = force_extract or not os.path.exists(labels_file)

        if should_extract:
            # Check if ontology directory exists
            if not os.path.exists(ontology_dir):
                logger.error(f"Ontology directory not found at '{ontology_dir}'")
                logger.error("Cannot extract class labels without ontology files")
                return {}

            # Check if ontology files exist
            ontology_files = [f for f in os.listdir(ontology_dir) if f.endswith(('.ttl', '.rdf', '.owl', '.n3'))]
            if not ontology_files:
                logger.error(f"No ontology files found in '{ontology_dir}'")
                logger.error("Add CIDOC-CRM, VIR, CRMdig ontology files to the ontology directory")
                return {}

            # Run extraction (this extracts properties, classes, and class labels)
            logger.info("Extracting class labels from ontology files...")
            logger.info(f"Found {len(ontology_files)} ontology files: {', '.join(ontology_files)}")

            try:
                prop_labels = str(PROJECT_ROOT / 'data' / 'labels' / 'property_labels.json')
                ont_classes = str(PROJECT_ROOT / 'data' / 'labels' / 'ontology_classes.json')
                success = run_extraction(ontology_dir, prop_labels, ont_classes, labels_file)
                if not success:
                    logger.error("Failed to extract class labels from ontologies")
                    return {}
                else:
                    logger.info(f"✓ Successfully extracted class labels to {labels_file}")
            except Exception as e:
                logger.error(f"Error during class label extraction: {str(e)}")
                return {}

        # Load the JSON file
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
                logger.info(f"Loaded {len(labels)} class labels from {labels_file}")
                return labels
            except Exception as e:
                logger.error(f"Error loading class labels from {labels_file}: {str(e)}")
                return {}
        else:
            logger.error(f"Class labels file not found at {labels_file}")
            return {}

    def _load_event_classes(self):
        """
        Load event classes from config/event_classes.json.

        Event classes are CIDOC-CRM and extension classes that represent events
        (activities, processes, etc.). These are used for event-aware graph traversal
        where multi-hop context only goes THROUGH events.

        Returns:
            set: Set of event class URIs
        """
        config_file = str(PROJECT_ROOT / 'config' / 'event_classes.json')

        if not os.path.exists(config_file):
            logger.warning(f"Event classes config not found at {config_file}")
            logger.warning("Event-aware traversal will be disabled (all entities treated equally)")
            return set()

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Combine all event class lists (ignore _comment key)
            event_classes = set()
            for key, value in data.items():
                if key.startswith('_'):  # Skip comment fields
                    continue
                if isinstance(value, list):
                    event_classes.update(value)

            logger.info(f"Loaded {len(event_classes)} event classes from {config_file}")
            return event_classes

        except Exception as e:
            logger.error(f"Error loading event classes from {config_file}: {str(e)}")
            return set()

    @classmethod
    def get_event_classes(cls):
        """
        Get the cached event classes, loading from JSON if needed.

        Returns:
            set: Set of event class URIs
        """
        if cls._event_classes is None:
            # Create a temporary instance just to load event classes
            # This is a bit awkward but maintains consistency with other label loading
            instance = object.__new__(cls)
            cls._event_classes = instance._load_event_classes()
        return cls._event_classes

    @property
    def embeddings(self):
        """
        Return an embedding object compatible with FAISS and the rest of the code.
        This property maintains backward compatibility with existing code.
        Uses the embedding_provider (which may be different from llm_provider).
        """
        class EmbeddingFunction:
            def __init__(self, provider):
                self.provider = provider

            def __call__(self, text):
                """Make the object callable for FAISS"""
                return self.provider.get_embeddings(text)

            def embed_query(self, text):
                """For code that explicitly calls embed_query"""
                return self.provider.get_embeddings(text)

            def embed_documents(self, texts):
                """For code that needs to embed multiple documents"""
                if hasattr(self.provider, 'get_embeddings_batch'):
                    return self.provider.get_embeddings_batch(texts)
                return [self.provider.get_embeddings(text) for text in texts]

            def get_embeddings_batch(self, texts):
                """Batch embedding for efficiency"""
                if hasattr(self.provider, 'get_embeddings_batch'):
                    return self.provider.get_embeddings_batch(texts)
                return [self.provider.get_embeddings(text) for text in texts]

        return EmbeddingFunction(self.embedding_provider)
    
    def test_connection(self):
        """Test connection to SPARQL endpoint"""
        try:
            query = """
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o
            } LIMIT 1
            """
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            logger.info("Successfully connected to SPARQL endpoint")
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False
    
    def initialize(self):
        """Initialize the system"""

        # Initialize document store
        self.document_store = GraphDocumentStore(self.embeddings)

        # Check if saved data exists (using dataset-specific paths)
        doc_graph_path = self._get_document_graph_path()
        vector_index_path = self._get_vector_index_path()
        vector_index_dir = self._get_vector_index_dir()

        logger.info(f"Checking for saved data at {doc_graph_path} and {vector_index_path}")

        if os.path.exists(doc_graph_path):
            logger.info(f"Found document graph at {doc_graph_path}")
        else:
            logger.info(f"Document graph file not found at {doc_graph_path}")

        if os.path.exists(vector_index_path):
            logger.info(f"Found vector index at {vector_index_path}")
        else:
            logger.info(f"Vector index not found at {vector_index_path}")

        if os.path.exists(doc_graph_path) and os.path.exists(vector_index_path):
            logger.info("Found both document graph and vector store, attempting to load...")

            # Try to load document graph
            graph_loaded = self.document_store.load_document_graph(doc_graph_path)

            # Try to load vector store
            vector_loaded = False
            try:
                self.document_store.vector_store = FAISS.load_local(
                    vector_index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                vector_loaded = True
                logger.info("Vector store loaded successfully")
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")

            if graph_loaded and vector_loaded:
                logger.info("Successfully loaded existing document graph and vector store")
                self._load_triples_index()
                # Load or build BM25 index
                bm25_dir = self._get_bm25_index_dir()
                if not self.document_store.load_bm25_index(bm25_dir):
                    logger.info("BM25 index not cached, building from loaded documents...")
                    if self.document_store.build_bm25_index():
                        self.document_store.save_bm25_index(bm25_dir)
                return True
            else:
                logger.warning("Failed to load saved data completely, rebuilding...")
        else:
            logger.info("No saved data found, building from scratch...")

        # SPARQL connection only needed when rebuilding
        if not self.test_connection():
            logger.error("Failed to connect to SPARQL endpoint (needed to rebuild data)")
            return False

        logger.info("Building document graph from RDF data...")

        # Process RDF data
        self.process_rdf_data()

        # Save document graph
        self.document_store.save_document_graph(doc_graph_path)
        os.makedirs(vector_index_dir, exist_ok=True)

        # Save the vector store
        if self.document_store.vector_store:
            self.document_store.vector_store.save_local(vector_index_dir)
            logger.info(f"Vector store saved to {vector_index_dir}")

        # Build and save BM25 index
        if self.document_store.build_bm25_index():
            bm25_dir = self._get_bm25_index_dir()
            self.document_store.save_bm25_index(bm25_dir)

        return True

    def _identify_satellites_from_prefetched(
        self,
        all_types: Dict[str, set],
        fr_incoming: Dict[str, List[Tuple[str, str]]],
        entity_labels_map: Dict[str, str],
        all_literals: Dict[str, Dict[str, List[str]]] = None,
    ) -> Tuple[set, Dict[str, Dict[str, list]]]:
        """Identify satellite entities and map them to parents using pre-fetched data.

        Two-pass: first identify all satellites, then find parents.

        Args:
            all_types: entity_uri -> set of type URIs
            fr_incoming: entity_uri -> [(pred, subj), ...] (incoming edges)
            entity_labels_map: entity_uri -> label
            all_literals: entity_uri -> {prop_name: [values]} for looking up
                time-span date values (P82a, P82b, P82)

        Returns:
            (satellite_uris, parent_satellites) where parent_satellites is
            parent_uri -> {kind: [label_or_dict, ...]}
            For time satellites, entries are dicts: {"label": str, "begin": str|None,
            "end": str|None, "within": str|None}
        """
        from collections import defaultdict

        if not self.fr_traversal:
            return set(), {}

        # Date property local names to look for on E52_Time-Span entities
        _TIME_PROPS = {
            "P82a_begin_of_the_begin": "begin",
            "P82b_end_of_the_end": "end",
            "P82_at_some_time_within": "within",
        }

        # Pass 1: identify all satellites
        satellite_uris = set()
        for uri, types in all_types.items():
            if self.fr_traversal.is_minimal_doc_entity(types):
                satellite_uris.add(uri)

        # Pass 2: find parent for each satellite
        parent_satellites = defaultdict(lambda: defaultdict(list))

        for sat_uri in satellite_uris:
            sat_label = entity_labels_map.get(sat_uri, sat_uri.split('/')[-1])
            sat_kind = classify_satellite(all_types.get(sat_uri, set()))

            # For time satellites, look up actual date values from literals
            if sat_kind == "time" and all_literals:
                sat_lits = all_literals.get(sat_uri, {})
                date_info = {"label": sat_label, "begin": None, "end": None, "within": None}
                for prop_name, key in _TIME_PROPS.items():
                    vals = sat_lits.get(prop_name, [])
                    if vals:
                        date_info[key] = vals[0]
                sat_entry = date_info
            else:
                sat_entry = sat_label

            for pred, subj in fr_incoming.get(sat_uri, []):
                if subj not in satellite_uris:
                    parent_satellites[subj][sat_kind].append(sat_entry)
                    break

        logger.info(f"Satellite absorption: {len(satellite_uris)} satellites → "
                    f"{len(parent_satellites)} parent entities enriched")

        return satellite_uris, parent_satellites

    def _save_edges_parquet(self, all_triples: List[Dict]) -> None:
        """Save collected triples to a Parquet edges file alongside entity_documents/.

        Stores 6 columns: URIs (s, p, o) and labels (s_label, p_label, o_label).

        Args:
            all_triples: List of dicts with 'subject', 'subject_label', 'predicate',
                        'predicate_label', 'object', 'object_label' keys
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        # edges.parquet lives in data/documents/{dataset_id}/ (parent of entity_documents/)
        edges_path = os.path.join(os.path.dirname(self._get_documents_dir()), "edges.parquet")

        subjects = [t["subject"] for t in all_triples]
        subject_labels = [t.get("subject_label", "") for t in all_triples]
        predicates = [t["predicate"] for t in all_triples]
        predicate_labels = [t.get("predicate_label", "") for t in all_triples]
        objects = [t["object"] for t in all_triples]
        object_labels = [t.get("object_label", "") for t in all_triples]

        table = pa.table({
            "s": subjects, "s_label": subject_labels,
            "p": predicates, "p_label": predicate_labels,
            "o": objects, "o_label": object_labels,
        })
        pq.write_table(table, edges_path)

        size_mb = os.path.getsize(edges_path) / (1024 * 1024)
        logger.info(f"Saved {len(all_triples)} triples (6 columns) to {edges_path} ({size_mb:.1f} MB)")

    def _load_triples_index(self):
        """Load Parquet edges file and build entity -> triples index.

        Reads the enriched edges.parquet (6 columns: s, s_label, p, p_label, o, o_label)
        and creates a dict mapping each entity URI to its list of triple dicts.
        Each triple appears under both its subject and object URI.

        Falls back gracefully if the file is missing or uses the old 3-column format.
        """
        edges_path = os.path.join(os.path.dirname(self._get_documents_dir()), "edges.parquet")
        if not os.path.exists(edges_path):
            logger.warning(f"No edges file at {edges_path} — raw_triples will be empty in responses")
            self._triples_index = {}
            return

        import pyarrow.parquet as pq

        table = pq.read_table(edges_path)
        columns = set(table.column_names)

        # Check for enriched format (6 columns with labels)
        has_labels = {"s_label", "p_label", "o_label"}.issubset(columns)
        if not has_labels:
            logger.warning("edges.parquet uses old 3-column format — re-generate docs to get labels")
            self._triples_index = {}
            return

        index = {}
        for s, s_label, p, p_label, o, o_label in zip(
            table.column("s"), table.column("s_label"),
            table.column("p"), table.column("p_label"),
            table.column("o"), table.column("o_label"),
        ):
            triple = {
                "subject": s.as_py(),
                "subject_label": s_label.as_py(),
                "predicate": p.as_py(),
                "predicate_label": p_label.as_py(),
                "object": o.as_py(),
                "object_label": o_label.as_py(),
            }
            index.setdefault(triple["subject"], []).append(triple)
            index.setdefault(triple["object"], []).append(triple)

        self._triples_index = index
        logger.info(f"Loaded triples index from {edges_path}: "
                    f"{len(table)} triples, {len(index)} entities indexed")

    def _build_edges_from_parquet(self):
        """Build edges from Parquet edges file. O(n*r) where r is number of triples.

        Reads edges.parquet (saved during --generate-docs) and adds weighted edges
        to the document store using proper CIDOC-CRM semantic weights.
        """
        edges_path = os.path.join(os.path.dirname(self._get_documents_dir()), "edges.parquet")
        if not os.path.exists(edges_path):
            logger.warning(f"No edges file found at {edges_path}, skipping edge building")
            return

        import pyarrow.parquet as pq

        logger.info(f"Loading edges from {edges_path}...")
        table = pq.read_table(edges_path)
        total_triples = len(table)
        logger.info(f"Loaded {total_triples} triples")

        doc_uris = set(self.document_store.docs.keys())
        edges_added = 0
        skipped = 0

        for s, p, o in zip(table.column("s"), table.column("p"), table.column("o")):
            s_str, p_str, o_str = s.as_py(), p.as_py(), o.as_py()
            if s_str in doc_uris and o_str in doc_uris and s_str != o_str:
                weight = self.get_relationship_weight(p_str)
                pred_name = p_str.split('/')[-1].split('#')[-1]
                self.document_store.add_edge(s_str, o_str, pred_name, weight=weight)
                edges_added += 1
            else:
                skipped += 1

        logger.info(f"Added {edges_added} edges from Parquet file "
                    f"({skipped} triples skipped — endpoints not in document store)")

    def _get_inverse_predicate(self, predicate_uri):
        """
        Get the inverse predicate URI for a CIDOC-CRM predicate.
        Uses owl:inverseOf mappings extracted from ontology files.

        Args:
            predicate_uri: Full URI of the predicate

        Returns:
            Inverse predicate URI (or original if no inverse defined in ontology)
        """
        if UniversalRagSystem._inverse_properties:
            inverse = UniversalRagSystem._inverse_properties.get(predicate_uri)
            if inverse:
                return inverse

        # No inverse found in ontology - return original predicate
        return predicate_uri

    def process_cidoc_relationship(self, subject_uri, predicate, object_uri, subject_label=None, object_label=None):
        """Convert CIDOC-CRM RDF relationships to natural language using ontology labels"""

        # Extract predicate name handling various namespace formats
        # Handle: http://example.com/P89_falls_within, vir#K1i, crm:P89_falls_within
        # First split by '/' to get the last segment
        simple_pred = predicate.split('/')[-1]

        # If that segment contains '#', split by '#' to get the actual predicate
        if '#' in simple_pred:
            simple_pred = simple_pred.split('#')[-1]

        # Handle missing entity labels
        subject_label = subject_label or subject_uri.split('/')[-1].rstrip('/')
        object_label = object_label or object_uri.split('/')[-1].rstrip('/')

        # Look up the predicate label from the ontology-extracted labels
        # Try multiple strategies: full URI, local name with code, stripped name
        predicate_label = None

        if self._property_labels:
            # Try full predicate URI first
            predicate_label = self._property_labels.get(predicate)

            # Try local name with prefix code (e.g., "K24_portray", "L54_is_same-as")
            if not predicate_label:
                predicate_label = self._property_labels.get(simple_pred)

            # Try stripped name without prefix code (e.g., "portray", "is_same-as")
            if not predicate_label:
                stripped_pred = re.sub(r'^[A-Z]\d+[a-z]?_', '', simple_pred)
                predicate_label = self._property_labels.get(stripped_pred)

        # If no label found in ontology, handle missing property
        if not predicate_label:
            # Track this missing property
            if predicate not in UniversalRagSystem._missing_properties:
                UniversalRagSystem._missing_properties.add(predicate)
                logger.warning(f"Property label not found for: {predicate} (local: {simple_pred})")

                # If we haven't tried extraction yet, trigger it
                if not UniversalRagSystem._extraction_attempted:
                    logger.info("Attempting to re-extract property labels from ontologies...")
                    new_labels = self._load_property_labels(force_extract=True)
                    if new_labels:
                        # Update the class-level cache
                        UniversalRagSystem._property_labels = new_labels
                        # Try to find the label again
                        predicate_label = (
                            new_labels.get(predicate) or
                            new_labels.get(simple_pred) or
                            new_labels.get(re.sub(r'^[A-Z]\d+[a-z]?_', '', simple_pred))
                        )

            # If still not found, create a fallback label
            if not predicate_label:
                # Strip prefix codes and convert underscores to spaces
                stripped_pred = re.sub(r'^[A-Z]\d+[a-z]?_', '', simple_pred)
                predicate_label = stripped_pred.replace('_', ' ').lower()
                logger.debug(f"Using fallback label '{predicate_label}' for property {simple_pred}")

        # Special handling for type classification relationships (P2_has_type)
        # to distinguish instances from types in iconographic classifications
        if "P2_has_type" in predicate or "P2_has_type" in simple_pred:
            return f"{subject_label} is classified as type: {object_label}"

        # Return natural language statement using the predicate label
        return f"{subject_label} {predicate_label} {object_label}"

    def is_schema_predicate(self, predicate):
        """Check if a predicate is a schema-level predicate that should be filtered out"""
        return _is_schema_predicate(predicate)

    def is_technical_class_name(self, class_name):
        """Check if a class name is a technical ontology class that should be filtered"""
        return _is_technical_class_name(class_name, UniversalRagSystem._ontology_classes)

    def _batch_query_tsv(self, query: str) -> List[List[str]]:
        """Execute a SPARQL query using TSV format for 3x faster parsing."""
        return self.batch_sparql.batch_query_tsv(query)

    def _escape_uri_for_values(self, uri: str) -> str:
        """Escape a URI for use in SPARQL VALUES clause."""
        return self.batch_sparql.escape_uri_for_values(uri)

    def _batch_fetch_types(self, uris: List[str], batch_size: int = None) -> Dict[str, set]:
        """Batch fetch rdf:type for multiple URIs."""
        return self.batch_sparql.batch_fetch_types(uris, batch_size or RetrievalConfig.BATCH_QUERY_SIZE)

    def _batch_query_outgoing(self, uris: List[str], batch_size: int = None) -> Dict[str, List[Tuple[str, str, Optional[str]]]]:
        """Batch query outgoing relationships for multiple URIs."""
        return self.batch_sparql.batch_query_outgoing(uris, batch_size or RetrievalConfig.BATCH_QUERY_SIZE)

    def _batch_query_incoming(self, uris: List[str], batch_size: int = None) -> Dict[str, List[Tuple[str, str, Optional[str]]]]:
        """Batch query incoming relationships for multiple URIs."""
        return self.batch_sparql.batch_query_incoming(uris, batch_size or RetrievalConfig.BATCH_QUERY_SIZE)

    def _batch_fetch_literals(self, uris: List[str], batch_size: int = None) -> Dict[str, Dict[str, List[str]]]:
        """Batch fetch literal properties for multiple URIs."""
        return self.batch_sparql.batch_fetch_literals(uris, batch_size or RetrievalConfig.BATCH_QUERY_SIZE)

    def _batch_fetch_type_labels(self, type_uris: set, batch_size: int = None) -> Dict[str, str]:
        """Batch fetch labels for type URIs."""
        return self.batch_sparql.batch_fetch_type_labels(type_uris, batch_size or RetrievalConfig.BATCH_QUERY_SIZE)

    def get_entities_context_batch(
        self,
        entity_uris: List[str],
        depth: int = None,
        batch_size: int = None
    ) -> Dict[str, Tuple[List[str], List[Dict]]]:
        """
        Batch version of get_entity_context using BFS-style traversal.

        Implements event-aware filtering matching the single-entity method:
        - For EVENT entities: full multi-hop traversal
        - For NON-EVENT entities: only traverse through events

        Args:
            entity_uris: List of entity URIs to get context for
            depth: How many hops to traverse (default: ENTITY_CONTEXT_DEPTH)
            batch_size: URIs per batch query (default: BATCH_QUERY_SIZE)

        Returns:
            Dict mapping URI -> (statements list, triples list)
        """
        if depth is None:
            depth = RetrievalConfig.ENTITY_CONTEXT_DEPTH
        if batch_size is None:
            batch_size = RetrievalConfig.BATCH_QUERY_SIZE

        logger.info(f"Batch context retrieval for {len(entity_uris)} entities (depth={depth})")

        # Initialize results for all requested entities
        results = {uri: ([], []) for uri in entity_uris}

        # Track which entities are starting entities (for event-aware filtering)
        starting_entities = set(entity_uris)

        # Cache for entity types (URI -> set of type URIs)
        type_cache = {}

        # Cache for entity labels (URI -> label string)
        label_cache = {}

        # Load event classes for filtering
        event_classes = UniversalRagSystem.get_event_classes()

        # Pre-fetch types for starting entities
        logger.info(f"  Pre-fetching types for {len(entity_uris)} starting entities...")
        starting_types = self._batch_fetch_types(entity_uris, batch_size)
        type_cache.update(starting_types)

        # Determine which starting entities are events
        starting_is_event = {}
        for uri in entity_uris:
            types = type_cache.get(uri, set())
            starting_is_event[uri] = bool(types & event_classes)

        # BFS traversal
        visited = set()
        current_level = set(entity_uris)

        # Track statements and triples per starting entity
        entity_statements = {uri: [] for uri in entity_uris}
        entity_triples = {uri: [] for uri in entity_uris}

        # Map discovered URIs back to their originating starting entity
        # uri_origin[discovered_uri] = set of starting entities that led here
        uri_origin = {uri: {uri} for uri in entity_uris}

        for current_depth in range(depth + 1):
            uris_to_query = [u for u in current_level if u not in visited]
            if not uris_to_query:
                break

            visited.update(uris_to_query)
            logger.info(f"  Depth {current_depth}: querying {len(uris_to_query)} URIs...")

            # Batch query outgoing and incoming relationships
            outgoing = self._batch_query_outgoing(uris_to_query, batch_size)
            incoming = self._batch_query_incoming(uris_to_query, batch_size)

            # Collect URIs for next level
            next_level = set()
            next_level_origins = {}  # new_uri -> set of starting entities

            # Pre-fetch types for all discovered URIs (for event-aware filtering)
            discovered_uris = set()
            for rels in outgoing.values():
                for pred, obj, _ in rels:
                    if obj not in visited and obj not in discovered_uris:
                        discovered_uris.add(obj)
            for rels in incoming.values():
                for subj, pred, _ in rels:
                    if subj not in visited and subj not in discovered_uris:
                        discovered_uris.add(subj)

            if discovered_uris:
                new_types = self._batch_fetch_types(list(discovered_uris), batch_size)
                type_cache.update(new_types)

            # Process outgoing relationships
            for uri, rels in outgoing.items():
                # Find which starting entities this URI is connected to
                origins = uri_origin.get(uri, set())
                current_types = type_cache.get(uri, set())
                current_is_event = bool(current_types & event_classes)

                for pred, obj, obj_label in rels:
                    # Skip schema predicates
                    if self.is_schema_predicate(pred):
                        continue

                    # Get predicate label
                    pred_label = UniversalRagSystem._property_labels.get(pred)
                    if not pred_label:
                        pred_label = pred.split('/')[-1].split('#')[-1]

                    # Get entity labels
                    if uri not in label_cache:
                        # Try to get from literals we already have
                        label_cache[uri] = uri.split('/')[-1]
                    entity_label = label_cache[uri]

                    if not obj_label:
                        obj_label = obj.split('/')[-1]
                    label_cache[obj] = obj_label

                    # Filter self-referential
                    if uri == obj:
                        continue
                    if entity_label and obj_label and entity_label.lower() == obj_label.lower():
                        continue

                    # Event-aware filtering for each starting entity
                    target_types = type_cache.get(obj, set())


                    for start_uri in origins:
                        start_is_event = starting_is_event.get(start_uri, False)

                        # Filtering logic matching single-entity method
                        if start_is_event:
                            should_include = True
                        elif current_depth == 0:
                            should_include = True
                        elif current_is_event:
                            should_include = True
                        else:
                            should_include = False

                        if should_include:
                            # Create statement
                            statement = self.process_cidoc_relationship(
                                uri, pred, obj, entity_label, obj_label
                            )
                            entity_statements[start_uri].append(statement)
                            entity_triples[start_uri].append({
                                "subject": uri,
                                "subject_label": entity_label,
                                "predicate": pred,
                                "predicate_label": pred_label,
                                "object": obj,
                                "object_label": obj_label
                            })

                            # Only traverse further through events (matching bulk script)
                            target_is_event = bool(target_types & event_classes)
                            if current_depth < depth and obj not in visited:
                                if starting_is_event.get(start_uri, False) or target_is_event:
                                    next_level.add(obj)
                                    if obj not in next_level_origins:
                                        next_level_origins[obj] = set()
                                    next_level_origins[obj].add(start_uri)

            # Process incoming relationships
            for uri, rels in incoming.items():
                origins = uri_origin.get(uri, set())
                current_types = type_cache.get(uri, set())
                current_is_event = bool(current_types & event_classes)

                if uri not in label_cache:
                    label_cache[uri] = uri.split('/')[-1]
                entity_label = label_cache[uri]

                for subj, pred, subj_label in rels:
                    if self.is_schema_predicate(pred):
                        continue

                    pred_label = UniversalRagSystem._property_labels.get(pred)
                    if not pred_label:
                        pred_label = pred.split('/')[-1].split('#')[-1]

                    if not subj_label:
                        subj_label = subj.split('/')[-1]
                    label_cache[subj] = subj_label

                    if subj == uri:
                        continue
                    if subj_label and entity_label and subj_label.lower() == entity_label.lower():
                        continue

                    target_types = type_cache.get(subj, set())


                    for start_uri in origins:
                        start_is_event = starting_is_event.get(start_uri, False)

                        if start_is_event:
                            should_include = True
                        elif current_depth == 0:
                            should_include = True
                        elif current_is_event:
                            should_include = True
                        else:
                            should_include = False

                        if should_include:
                            # Use inverse predicate for incoming
                            inverse_pred = self._get_inverse_predicate(pred)
                            statement = self.process_cidoc_relationship(
                                uri, inverse_pred, subj, entity_label, subj_label
                            )
                            entity_statements[start_uri].append(statement)
                            entity_triples[start_uri].append({
                                "subject": subj,
                                "subject_label": subj_label,
                                "predicate": pred,
                                "predicate_label": pred_label,
                                "object": uri,
                                "object_label": entity_label
                            })

                            # Traverse from incoming IF the source is an event
                            # Critical for CIDOC-CRM: production events arrive as incoming
                            # to actors/objects, traverse through them to discover artworks
                            source_types = type_cache.get(subj, set())
                            source_is_event = bool(source_types & event_classes)
                            if current_depth < depth and subj not in visited:
                                if starting_is_event.get(start_uri, False) or source_is_event:
                                    next_level.add(subj)
                                    if subj not in next_level_origins:
                                        next_level_origins[subj] = set()
                                    next_level_origins[subj].add(start_uri)

            # Update origins for next level
            for uri, origins in next_level_origins.items():
                if uri not in uri_origin:
                    uri_origin[uri] = set()
                uri_origin[uri].update(origins)

            current_level = next_level

        # Compile results - deduplicate statements and triples
        for uri in entity_uris:
            unique_statements = list(set(entity_statements[uri]))

            seen_triples = set()
            unique_triples = []
            for triple in entity_triples[uri]:
                triple_key = (triple["subject"], triple["predicate"], triple["object"])
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    unique_triples.append(triple)

            results[uri] = (unique_statements, unique_triples)

        logger.info(f"  Batch context complete: visited {len(visited)} total URIs")
        return results

    def _create_document_from_prefetched(
        self,
        entity_uri: str,
        literals: Dict[str, List[str]],
        types: set,
        context: Tuple[List[str], List[Dict]],
        type_labels: Dict[str, str] = None
    ) -> Tuple[str, str, List[str], List[Dict]]:
        """
        Create document text from pre-fetched data (no SPARQL queries).

        Same format as create_enhanced_document() but uses pre-fetched data.

        Args:
            entity_uri: The entity URI
            literals: Dict of property_name -> [values]
            types: Set of type URIs
            context: Tuple of (statements, triples) from batch context
            type_labels: Optional dict of type_uri -> label (from batch fetch)

        Returns:
            Tuple of (doc_text, entity_label, type_labels_list, raw_triples)
        """
        # Extract label from literals
        entity_label = entity_uri.split('/')[-1]
        for label_prop in ['label', 'prefLabel', 'name', 'title']:
            if label_prop in literals and literals[label_prop]:
                entity_label = literals[label_prop][0]
                break

        # Convert type URIs to labels
        entity_types = []
        for type_uri in types:
            # Try class_labels.json first
            type_label = None
            if UniversalRagSystem._class_labels:
                type_label = UniversalRagSystem._class_labels.get(type_uri)

            # Try pre-fetched labels
            if not type_label and type_labels:
                type_label = type_labels.get(type_uri)

            # Fallback to local name
            if not type_label:
                type_label = type_uri.split('/')[-1].split('#')[-1]
                if type_uri not in UniversalRagSystem._missing_classes:
                    UniversalRagSystem._missing_classes.add(type_uri)

            entity_types.append(type_label)

        # Unpack context
        context_statements, raw_triples = context

        # Create document text (same format as create_enhanced_document)
        text = f"# {entity_label}\n\n"
        text += f"URI: {entity_uri}\n\n"

        # Add types (filter technical class names)
        if entity_types:
            human_readable_types = [
                t for t in entity_types
                if not self.is_technical_class_name(t)
            ]
            if human_readable_types:
                text += "## Types\n\n"
                for type_label in human_readable_types:
                    text += f"- {type_label}\n"
                text += "\n"

        # Add literal properties
        if literals:
            text += "## Properties\n\n"
            for prop_name, values in sorted(literals.items()):
                display_name = prop_name.replace('_', ' ').title()
                if len(values) == 1:
                    value_str = values[0]
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "... [truncated]"
                    text += f"- **{display_name}**: {value_str}\n"
                else:
                    text += f"- **{display_name}**:\n"
                    for value in values:
                        value_str = value
                        if len(value_str) > 200:
                            value_str = value_str[:200] + "... [truncated]"
                        text += f"  - {value_str}\n"
            text += "\n"

        # Add relationship statements
        if context_statements:
            text += "## Relationships\n\n"
            for statement in context_statements:
                text += f"- {statement}\n"

        return text, entity_label, entity_types, raw_triples

    def _batch_fetch_wikidata_ids(self, uris: List[str], batch_size: int = None) -> Dict[str, str]:
        """Batch fetch Wikidata IDs (crmdig:L54_is_same-as) for multiple URIs."""
        return self.batch_sparql.batch_fetch_wikidata_ids(uris, batch_size or RetrievalConfig.BATCH_QUERY_SIZE)

    def _build_image_index(self) -> Dict[str, List[str]]:
        """Build image index using SPARQL pattern from dataset configuration."""
        return self.batch_sparql.build_image_index(self.dataset_config)

    # ==================== End Batch SPARQL Query Methods ====================

    # ==================== FR-based Document Generation ====================

    def _build_fr_graph_for_chunk(
        self,
        chunk_uris: List[str],
        chunk_types: Dict[str, set],
        chunk_literals: Dict[str, Dict[str, List[str]]],
    ) -> Tuple[Dict, Dict, Dict, Dict, List[Dict]]:
        """Build the outgoing/incoming/entity_labels indexes that FR traversal needs.

        Uses SPARQL batch queries. Fetches 2 hops: direct edges from chunk entities
        plus edges from intermediate URIs reached at step 1 (events, types, etc.).

        Args:
            chunk_uris: Entity URIs in this chunk
            chunk_types: Pre-fetched entity types (uri -> set of type URIs)
            chunk_literals: Pre-fetched literals (for extracting labels)

        Returns:
            (fr_outgoing, fr_incoming, entity_labels, entity_types_map, raw_triples)
            where fr_outgoing/fr_incoming use 2-tuple format (pred, target) per FR spec.
        """
        from collections import defaultdict

        chunk_set = set(chunk_uris)

        # Step 1: Batch query outgoing and incoming for chunk entities
        raw_outgoing = self._batch_query_outgoing(chunk_uris)
        raw_incoming = self._batch_query_incoming(chunk_uris)

        # Build FR-format indexes (2-tuple: pred, target) and collect labels
        fr_outgoing = defaultdict(list)  # uri -> [(pred, obj)]
        fr_incoming = defaultdict(list)  # uri -> [(pred, subj)]
        entity_labels = {}
        raw_triples = []

        # Extract labels from chunk_literals
        for uri in chunk_uris:
            literals = chunk_literals.get(uri, {})
            label = uri.split('/')[-1]
            for label_prop in ['label', 'prefLabel', 'name', 'title']:
                if label_prop in literals and literals[label_prop]:
                    label = literals[label_prop][0]
                    break
            entity_labels[uri] = label

        # Process outgoing: raw format is (pred, obj, obj_label)
        intermediate_uris = set()
        for uri, rels in raw_outgoing.items():
            for pred, obj, obj_label in rels:
                if self.is_schema_predicate(pred):
                    continue
                fr_outgoing[uri].append((pred, obj))
                if obj_label:
                    entity_labels[obj] = obj_label
                elif obj not in entity_labels:
                    entity_labels[obj] = obj.split('/')[-1].split('#')[-1]
                # Collect intermediates not in our chunk (events, places reached at step 1)
                if obj not in chunk_set:
                    intermediate_uris.add(obj)
                # Collect raw triple for edges.parquet
                pred_label = ""
                if UniversalRagSystem._property_labels:
                    simple_pred = pred.split('/')[-1].split('#')[-1]
                    pred_label = (
                        UniversalRagSystem._property_labels.get(pred) or
                        UniversalRagSystem._property_labels.get(simple_pred) or ""
                    )
                raw_triples.append({
                    "subject": uri,
                    "subject_label": entity_labels.get(uri, ""),
                    "predicate": pred,
                    "predicate_label": pred_label,
                    "object": obj,
                    "object_label": entity_labels.get(obj, ""),
                })

        # Process incoming: raw format is (subj, pred, subj_label)
        for uri, rels in raw_incoming.items():
            for subj, pred, subj_label in rels:
                if self.is_schema_predicate(pred):
                    continue
                fr_incoming[uri].append((pred, subj))
                if subj_label:
                    entity_labels[subj] = subj_label
                elif subj not in entity_labels:
                    entity_labels[subj] = subj.split('/')[-1].split('#')[-1]
                if subj not in chunk_set:
                    intermediate_uris.add(subj)
                # Collect raw triple for edges.parquet
                pred_label = ""
                if UniversalRagSystem._property_labels:
                    simple_pred = pred.split('/')[-1].split('#')[-1]
                    pred_label = (
                        UniversalRagSystem._property_labels.get(pred) or
                        UniversalRagSystem._property_labels.get(simple_pred) or ""
                    )
                raw_triples.append({
                    "subject": subj,
                    "subject_label": entity_labels.get(subj, ""),
                    "predicate": pred,
                    "predicate_label": pred_label,
                    "object": uri,
                    "object_label": entity_labels.get(uri, ""),
                })

        # Step 2: Fetch edges for intermediate URIs (2-hop coverage for FR paths)
        if intermediate_uris:
            intermediate_list = list(intermediate_uris)
            logger.info(f"    FR: fetching edges for {len(intermediate_list)} intermediate URIs...")

            inter_outgoing = self._batch_query_outgoing(intermediate_list)
            inter_incoming = self._batch_query_incoming(intermediate_list)

            for uri, rels in inter_outgoing.items():
                for pred, obj, obj_label in rels:
                    if self.is_schema_predicate(pred):
                        continue
                    fr_outgoing[uri].append((pred, obj))
                    if obj_label:
                        entity_labels[obj] = obj_label
                    elif obj not in entity_labels:
                        entity_labels[obj] = obj.split('/')[-1].split('#')[-1]

            for uri, rels in inter_incoming.items():
                for subj, pred, subj_label in rels:
                    if self.is_schema_predicate(pred):
                        continue
                    fr_incoming[uri].append((pred, subj))
                    if subj_label:
                        entity_labels[subj] = subj_label
                    elif subj not in entity_labels:
                        entity_labels[subj] = subj.split('/')[-1].split('#')[-1]

            # Fetch types for intermediates (needed for FR range-FC filtering)
            inter_types = self._batch_fetch_types(intermediate_list)
            chunk_types.update(inter_types)

        # entity_types_map = chunk_types (already updated with intermediates)
        entity_types_map = chunk_types

        logger.info(f"    FR graph: {len(fr_outgoing)} outgoing, {len(fr_incoming)} incoming, "
                    f"{len(entity_labels)} labels, {len(raw_triples)} raw triples")

        return dict(fr_outgoing), dict(fr_incoming), entity_labels, entity_types_map, raw_triples

    def _create_fr_document_from_prefetched(
        self,
        entity_uri: str,
        entity_label: str,
        raw_types: set,
        entity_type_labels: List[str],
        literals: Dict[str, List[str]],
        fr_outgoing: Dict,
        fr_incoming: Dict,
        entity_labels: Dict[str, str],
        entity_types_map: Dict[str, set],
        absorbed_lines: List[str] = None,
    ) -> Tuple[str, str, List[str]]:
        """Create FR-organized document from pre-fetched data.

        Args:
            entity_uri: Entity URI
            entity_label: Entity label
            raw_types: Set of rdf:type URIs
            entity_type_labels: Human-readable type label strings
            literals: prop_name -> [values] dict
            fr_outgoing: Full graph outgoing index (uri -> [(pred, obj)])
            fr_incoming: Full graph incoming index (uri -> [(pred, subj)])
            entity_labels: Full graph label index
            entity_types_map: Full graph entity types (uri -> set of type URIs)
            absorbed_lines: Lines from absorbed satellite entities

        Returns:
            (text, label, type_labels)
        """
        types_display = [t for t in entity_type_labels if not self.is_technical_class_name(t)]

        # Minimal doc for vocabulary entities
        if self.fr_traversal.is_minimal_doc_entity(raw_types):
            text = self.fr_traversal.format_minimal_document(
                entity_uri, entity_label, types_display, literals
            )
            return text, entity_label, entity_type_labels

        # Full FR traversal
        fc = self.fr_traversal.get_fc(raw_types)

        fr_results = self.fr_traversal.match_fr_paths(
            entity_uri=entity_uri,
            entity_types=raw_types,
            outgoing=fr_outgoing,
            incoming=fr_incoming,
            entity_labels=entity_labels,
            entity_types_map=entity_types_map,
        )

        # Collect direct non-FR predicates (VIR extensions etc.)
        direct_preds = self.fr_traversal.collect_direct_predicates(
            entity_uri=entity_uri,
            outgoing=fr_outgoing,
            incoming=fr_incoming,
            entity_labels=entity_labels,
            entity_types=raw_types,
            schema_filter=self.is_schema_predicate,
        )

        # Build target enrichments (type tags + attributes for FR targets)
        all_target_uris = set()
        for fr in fr_results:
            for uri, _lbl in fr["targets"]:
                all_target_uris.add(uri)
        for dp in (direct_preds or []):
            for uri, _lbl in dp["targets"]:
                all_target_uris.add(uri)

        from fr_traversal import build_target_enrichments
        target_enrichments = build_target_enrichments(
            target_uris=all_target_uris,
            outgoing=fr_outgoing,
            entity_labels=entity_labels,
            entity_types_map=entity_types_map,
            class_labels=UniversalRagSystem._class_labels,
        )

        text = self.fr_traversal.format_fr_document(
            entity_uri=entity_uri,
            label=entity_label,
            types_display=types_display,
            literals=literals,
            fr_results=fr_results,
            direct_predicates=direct_preds,
            fc=fc,
            absorbed_lines=absorbed_lines,
            target_enrichments=target_enrichments,
        )

        return text, entity_label, entity_type_labels

    # ==================== End FR-based Document Generation ====================

    def save_entity_document(self, entity_uri, document_text, entity_label,
                             output_dir=None, entity_type=None, all_types=None,
                             wikidata_id=None, images=None):
        """Save entity document to disk for transparency and reuse.

        Args:
            entity_uri: The entity URI
            document_text: Document text content
            entity_label: Entity label
            output_dir: Override for output directory
            entity_type: Primary entity type (e.g. "E22_Human-Made_Object")
            all_types: List of all human-readable entity types
            wikidata_id: Wikidata Q-ID if available
            images: List of image URLs
        """

        try:
            # Use dataset-specific output directory if not specified
            if output_dir is None:
                output_dir = self._get_documents_dir()

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Create a safe filename from the entity label
            # Remove special characters and limit length
            safe_label = re.sub(r'[^\w\s-]', '', entity_label)
            safe_label = re.sub(r'[-\s]+', '_', safe_label)
            safe_label = safe_label[:100]  # Limit filename length

            # Use hash of URI to ensure uniqueness
            uri_hash = hashlib.md5(entity_uri.encode()).hexdigest()[:8]

            # Create filename: label + hash
            filename = f"{safe_label}_{uri_hash}.md"
            filepath = os.path.join(output_dir, filename)

            # Build metadata header (source of truth for all metadata)
            # Quote label to handle values containing colons (YAML special char)
            safe_entity_label = entity_label.replace('"', '\\"')
            metadata_lines = [
                "---",
                f"URI: {entity_uri}",
                f'Label: "{safe_entity_label}"',
            ]

            if entity_type:
                metadata_lines.append(f"Type: {entity_type}")

            if all_types:
                metadata_lines.append("Types:")
                for t in all_types:
                    metadata_lines.append(f"  - {t}")

            if wikidata_id:
                metadata_lines.append(f"Wikidata: {wikidata_id}")

            if images:
                metadata_lines.append("Images:")
                for img_url in images:
                    metadata_lines.append(f"  - {img_url}")

            metadata_lines.append("---")
            metadata_lines.append("")
            metadata = "\n".join(metadata_lines) + "\n"

            # Write document to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(metadata)
                f.write(document_text)

            return filepath
        except Exception as e:
            logger.error(f"Error saving entity document for {entity_uri}: {str(e)}")
            return None

    def generate_validation_report(self):
        """
        Generate a validation report showing missing classes and properties.
        Provides recommendations for handling missing ontology elements.
        """
        logger.info("\n" + "=" * 80)
        logger.info("ONTOLOGY VALIDATION REPORT")
        logger.info("=" * 80)

        # Report missing classes
        if UniversalRagSystem._missing_classes:
            logger.error("\n" + "!" * 80)
            logger.error(f"⚠ WARNING: Found {len(UniversalRagSystem._missing_classes)} classes NOT in ontology files!")
            logger.error("!" * 80)
            logger.warning("\nThese classes exist in your triplestore but their ontology definitions are")
            logger.warning("NOT present in the 'data/ontologies/' directory (CIDOC-CRM/VIR/CRMdig):")

            # Sort for consistent output
            missing_classes_sorted = sorted(UniversalRagSystem._missing_classes)
            for i, class_uri in enumerate(missing_classes_sorted[:20], 1):  # Show first 20
                local_name = class_uri.split('/')[-1].split('#')[-1]
                logger.warning(f"  {i}. {local_name}")
                logger.warning(f"     URI: {class_uri}")

            if len(UniversalRagSystem._missing_classes) > 20:
                logger.warning(f"  ... and {len(UniversalRagSystem._missing_classes) - 20} more")

            logger.error("\n" + "!" * 80)
            logger.error("⚠ ACTION REQUIRED FOR PROPER RAG FUNCTIONALITY:")
            logger.error("!" * 80)
            logger.error("\nTo proceed with optimal use of the RAG system, you MUST:")
            logger.error("\n  STEP 1: Identify missing ontology files")
            logger.error("    • Check the URIs above - the namespace tells you which ontology")
            logger.error("    • Examples: FRBRoo, CRMgeo, CRMsci, CRMarchaeo, or custom ontologies")
            logger.error("\n  STEP 2: Download or locate the ontology files (.ttl, .rdf, .owl)")
            logger.error("    • CIDOC-CRM extensions: https://www.cidoc-crm.org/extensions")
            logger.error("    • Custom ontologies: check your data provider or project docs")
            logger.error("\n  STEP 3: Add ontology files to 'data/ontologies/' directory")
            logger.error("    $ cp /path/to/ontology.ttl data/ontologies/")
            logger.error("\n  STEP 4: Extract labels from new ontology files")
            logger.error("    $ python scripts/extract_ontology_labels.py")
            logger.error("\n  STEP 5: Rebuild RAG system (delete caches and re-run)")
            logger.error("    $ rm -rf data/cache/<dataset_id>/ data/documents/<dataset_id>/")
            logger.error("\nCurrently using fallback strategy:")
            logger.error("  - Attempting to query labels from triplestore (English only)")
            logger.error("  - If no label found, deriving from URI local names")
            logger.error("  - This may result in incorrect or missing type information")
            logger.error(f"\nSee detailed instructions in: logs/ontology_validation_report.txt")
            logger.error("!" * 80)
        else:
            logger.info("\n✓ All classes found in ontology files")

        # Report missing properties
        if UniversalRagSystem._missing_properties:
            logger.error("\n" + "!" * 80)
            logger.error(f"⚠ WARNING: Found {len(UniversalRagSystem._missing_properties)} properties NOT in ontology files!")
            logger.error("!" * 80)
            logger.warning("\nThese properties exist in your triplestore but their ontology definitions are")
            logger.warning("NOT present in the 'data/ontologies/' directory (CIDOC-CRM/VIR/CRMdig):")

            # Sort for consistent output
            missing_props_sorted = sorted(UniversalRagSystem._missing_properties)
            for i, prop_uri in enumerate(missing_props_sorted[:20], 1):  # Show first 20
                local_name = prop_uri.split('/')[-1].split('#')[-1]
                logger.warning(f"  {i}. {local_name}")
                logger.warning(f"     URI: {prop_uri}")

            if len(UniversalRagSystem._missing_properties) > 20:
                logger.warning(f"  ... and {len(UniversalRagSystem._missing_properties) - 20} more")

            logger.error("\n" + "!" * 80)
            logger.error("⚠ ACTION REQUIRED FOR PROPER RAG FUNCTIONALITY:")
            logger.error("!" * 80)
            logger.error("\nTo proceed with optimal use of the RAG system, you MUST:")
            logger.error("\n  STEP 1: Identify missing ontology files")
            logger.error("    • Check the URIs above - the namespace tells you which ontology")
            logger.error("    • Examples: FRBRoo, CRMgeo, CRMsci, CRMarchaeo, or custom ontologies")
            logger.error("\n  STEP 2: Download or locate the ontology files (.ttl, .rdf, .owl)")
            logger.error("    • CIDOC-CRM extensions: https://www.cidoc-crm.org/extensions")
            logger.error("    • Custom ontologies: check your data provider or project docs")
            logger.error("\n  STEP 3: Add ontology files to 'data/ontologies/' directory")
            logger.error("    $ cp /path/to/ontology.ttl data/ontologies/")
            logger.error("\n  STEP 4: Extract labels from new ontology files")
            logger.error("    $ python scripts/extract_ontology_labels.py")
            logger.error("\n  STEP 5: Rebuild RAG system (delete caches and re-run)")
            logger.error("    $ rm -rf data/cache/<dataset_id>/ data/documents/<dataset_id>/")
            logger.error("\nCurrently using fallback strategy:")
            logger.error("  - Deriving labels from property local names")
            logger.error("    Example: 'P1_is_identified_by' → 'is identified by'")
            logger.error("  - This may result in suboptimal natural language descriptions")
            logger.error(f"\nSee detailed instructions in: logs/ontology_validation_report.txt")
            logger.error("!" * 80)
        else:
            logger.info("\n✓ All properties found in ontology files")

        # Save report to file
        report_file = "logs/ontology_validation_report.txt"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ONTOLOGY VALIDATION REPORT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Missing Classes: {len(UniversalRagSystem._missing_classes)}\n")
                f.write(f"Missing Properties: {len(UniversalRagSystem._missing_properties)}\n\n")

                has_missing = UniversalRagSystem._missing_classes or UniversalRagSystem._missing_properties

                if has_missing:
                    f.write("!" * 80 + "\n")
                    f.write("⚠ WARNING: ONTOLOGY FILES MISSING\n")
                    f.write("!" * 80 + "\n\n")
                    f.write("The ontology files for these classes/properties are NOT present in the\n")
                    f.write("'data/ontologies/' directory. To proceed with optimal use of the RAG system,\n")
                    f.write("you MUST add the missing ontology files.\n\n")

                if UniversalRagSystem._missing_classes:
                    f.write("MISSING CLASSES:\n")
                    f.write("-" * 80 + "\n")
                    for class_uri in sorted(UniversalRagSystem._missing_classes):
                        f.write(f"{class_uri}\n")
                    f.write("\n")

                if UniversalRagSystem._missing_properties:
                    f.write("MISSING PROPERTIES:\n")
                    f.write("-" * 80 + "\n")
                    for prop_uri in sorted(UniversalRagSystem._missing_properties):
                        f.write(f"{prop_uri}\n")
                    f.write("\n")

                if has_missing:
                    f.write("!" * 80 + "\n")
                    f.write("STEP-BY-STEP FIX INSTRUCTIONS:\n")
                    f.write("!" * 80 + "\n\n")

                    f.write("STEP 1: Identify the missing ontology files\n")
                    f.write("-" * 80 + "\n")
                    f.write("Look at the URIs listed above. The namespace in the URI tells you which\n")
                    f.write("ontology defines these classes/properties:\n\n")
                    f.write("Example URI patterns and their ontologies:\n")
                    f.write("  • http://www.cidoc-crm.org/cidoc-crm/...  → CIDOC-CRM (already in data/ontologies/)\n")
                    f.write("  • http://w3id.org/vir#...                  → VIR (already in data/ontologies/)\n")
                    f.write("  • http://www.ics.forth.gr/isl/CRMdig/...  → CRMdig (already in data/ontologies/)\n")
                    f.write("  • http://erlangen-crm.org/...             → Erlangen CRM\n")
                    f.write("  • http://www.cidoc-crm.org/frbroo/...     → FRBRoo\n")
                    f.write("  • http://www.cidoc-crm.org/crmgeo/...     → CRMgeo\n")
                    f.write("  • http://www.cidoc-crm.org/crmsci/...     → CRMsci\n")
                    f.write("  • http://www.cidoc-crm.org/crmarchaeo/... → CRMarchaeo\n")
                    f.write("  • http://www.cidoc-crm.org/crminf/...     → CRMinf\n")
                    f.write("  • http://www.ics.forth.gr/isl/CRMtex/...  → CRMtex\n")
                    f.write("  • http://iflastandards.info/ns/lrm/...    → LRM\n")
                    f.write("  • [Custom namespace]                      → Your custom ontology\n\n")

                    f.write("STEP 2: Download or locate the ontology files\n")
                    f.write("-" * 80 + "\n")
                    f.write("For standard CIDOC-CRM extensions:\n")
                    f.write("  • Visit: https://www.cidoc-crm.org/\n")
                    f.write("  • Or: https://cidoc-crm.org/extensions\n")
                    f.write("  • Download the .rdfs, .rdf, .owl, or .ttl file\n\n")
                    f.write("For custom/domain-specific ontologies:\n")
                    f.write("  • Contact your data provider\n")
                    f.write("  • Check your project documentation\n")
                    f.write("  • Look for ontology files alongside your RDF data\n\n")

                    f.write("STEP 3: Add ontology files to the 'data/ontologies/' directory\n")
                    f.write("-" * 80 + "\n")
                    f.write("  $ cp /path/to/downloaded/ontology.ttl data/ontologies/\n")
                    f.write("  $ cp /path/to/custom/ontology.rdf data/ontologies/\n\n")
                    f.write("Supported formats: .ttl, .rdf, .owl, .n3\n\n")

                    f.write("STEP 4: Extract labels from ontology files\n")
                    f.write("-" * 80 + "\n")
                    f.write("  $ python scripts/extract_ontology_labels.py\n\n")
                    f.write("This will regenerate:\n")
                    f.write("  • data/labels/property_labels.json (property URI → English label)\n")
                    f.write("  • data/labels/ontology_classes.json (class identifiers for filtering)\n")
                    f.write("  • data/labels/class_labels.json (class URI → English label)\n\n")

                    f.write("STEP 5: Rebuild the RAG system\n")
                    f.write("-" * 80 + "\n")
                    f.write("Delete cached data (replace <dataset_id> with your dataset):\n")
                    f.write("  $ rm -rf data/cache/<dataset_id>/\n")
                    f.write("  $ rm -rf data/documents/<dataset_id>/\n\n")
                    f.write("Re-run your initialization script to rebuild with new labels.\n\n")

                    f.write("!" * 80 + "\n")
                    f.write("CURRENT FALLBACK BEHAVIOR (until you complete the steps above):\n")
                    f.write("!" * 80 + "\n")
                    f.write("• Class labels: Querying triplestore for English labels, or deriving from URIs\n")
                    f.write("• Property labels: Deriving from property local names\n")
                    f.write("• This may result in:\n")
                    f.write("  - Incorrect or missing type information in documents\n")
                    f.write("  - Suboptimal natural language descriptions\n")
                    f.write("  - Reduced quality of RAG responses\n")
                    f.write("!" * 80 + "\n")

            logger.info(f"\n✓ Validation report saved to: {report_file}")
        except Exception as e:
            logger.error(f"Error saving validation report: {str(e)}")

        # Print final summary
        if UniversalRagSystem._missing_classes or UniversalRagSystem._missing_properties:
            logger.error("\n" + "=" * 80)
            logger.error("SUMMARY: Ontology files are missing for some classes/properties")
            logger.error(f"See detailed report: {report_file}")
            logger.error("=" * 80 + "\n")
        else:
            logger.info("=" * 80 + "\n")

    def process_rdf_data(self):
        """Process RDF data into graph documents with enhanced CIDOC-CRM understanding"""
        logger.info("Processing RDF data with enhanced CIDOC-CRM understanding...")

        # Get all entities
        entities = self.get_all_entities()
        total_entities = len(entities)
        logger.info(f"Found {total_entities} entities")

        # Clear entity_documents directory if it exists (using dataset-specific path)
        output_dir = self._get_documents_dir()
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            logger.info(f"Cleared existing {output_dir} directory")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Entity documents will be saved to: {output_dir}/")

        # Create README for entity_documents directory
        use_fr_docs = self.fr_traversal is not None
        if use_fr_docs:
            readme_content = """# Entity Documents (FR-based)

This directory contains individual markdown files for each entity processed from the RDF data
using Fundamental Relationship (FR) path traversal (Tzompanaki & Doerr, 2012).

## File Naming Convention
Files are named: `{label}_{hash}.md`
- `label`: Cleaned entity label (special chars removed, spaces replaced with underscores)
- `hash`: 8-character MD5 hash of the entity URI (ensures uniqueness)

## File Structure
Each file contains:
1. **Metadata header**: URI, label, generation timestamp
2. **[FC] Label** header line with fundamental category (Thing, Actor, Place, Event, Concept, Time)
3. **Literal properties**: labels, descriptions, dates (priority-ordered)
4. **FR relationships**: grouped by Fundamental Relationship type (e.g. "was produced by", "has current location")
5. **Direct predicates**: non-FR predicates like VIR extensions (K24_portray etc.)

Vocabulary entities (E55_Type, E30_Right, etc.) get minimal 2-5 line documents.

## Notes
- Schema-level predicates (rdf:type, rdfs:subClassOf, etc.) are filtered
- Files are regenerated on each rebuild
"""
        else:
            readme_content = """# Entity Documents (BFS-based)

This directory contains individual markdown files for each entity processed from the RDF data
using BFS relationship traversal.

## File Naming Convention
Files are named: `{label}_{hash}.md`

## File Structure
Each file contains:
1. **Metadata header**: URI, label, generation timestamp
2. **Types**: RDF types of the entity
3. **Properties**: All literal values (labels, descriptions, WKT geometries, dates, etc.)
4. **Relationships**: Filtered CIDOC-CRM relationships in natural language

## Notes
- Schema-level predicates (rdf:type, rdfs:subClassOf, etc.) are filtered from relationships
- Self-referential relationships are removed
- Files are regenerated on each rebuild
"""
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Create document nodes with enhanced content using chunked batch SPARQL
        logger.info("Creating enhanced document nodes (chunked batch SPARQL)...")

        # Determine embedding sub-batch size based on embedding provider
        if self.use_batch_embedding:
            embedding_batch_size = int(self.config.get("embedding_batch_size", 64))
            logger.info(f"Using batch embedding with embedding_batch_size={embedding_batch_size}")
        else:
            embedding_batch_size = RetrievalConfig.DEFAULT_BATCH_SIZE
            logger.info(f"Using sequential embedding with embedding_batch_size={embedding_batch_size}")

        # Global rate limit tracking (only used for API-based embeddings)
        global_token_count = 0
        tokens_per_min_limit = RetrievalConfig.TOKENS_PER_MINUTE_LIMIT
        last_reset_time = time.time()

        # Check embedding cache for already processed entities
        cached_count = 0
        if self.embedding_cache:
            cache_stats = self.embedding_cache.get_stats()
            logger.info(f"Embedding cache: {cache_stats['count']} cached embeddings ({cache_stats['size_mb']} MB)")

        # Collect all triples for edges file
        all_triples = []
        # Accumulate satellite URIs across all chunks
        all_satellite_uris = set()

        # Pre-fetch image index (single SPARQL query, shared across all chunks)
        image_index = self._build_image_index()

        # SPARQL pre-fetch chunk size (matches existing batch query infrastructure)
        chunk_size = RetrievalConfig.BATCH_QUERY_SIZE
        total_chunks = (total_entities + chunk_size - 1) // chunk_size
        logger.info(f"Processing {total_entities} entities in {total_chunks} chunks of {chunk_size} (batch SPARQL)")

        for chunk_idx in range(0, total_entities, chunk_size):
            chunk_entities = entities[chunk_idx:chunk_idx + chunk_size]
            chunk_num = chunk_idx // chunk_size + 1
            chunk_uris = [e["entity"] for e in chunk_entities]

            logger.info(f"=== Chunk {chunk_num}/{total_chunks} ({len(chunk_uris)} entities) ===")

            # Phase A: Batch pre-fetch all data for this chunk
            logger.info(f"  Phase A: Batch pre-fetching data for {len(chunk_uris)} entities...")

            chunk_literals = self._batch_fetch_literals(chunk_uris)
            logger.info(f"    Literals: {len(chunk_literals)} entities")

            chunk_types = self._batch_fetch_types(chunk_uris)
            logger.info(f"    Types: {len(chunk_types)} entities")

            # Collect all type URIs for batch label fetching
            chunk_type_uris = set()
            for types in chunk_types.values():
                chunk_type_uris.update(types)
            chunk_type_labels = self._batch_fetch_type_labels(chunk_type_uris)
            logger.info(f"    Type labels: {len(chunk_type_labels)} types")

            use_fr = self.fr_traversal is not None
            fr_outgoing = fr_incoming = entity_labels_map = entity_types_map = None
            chunk_contexts = None

            # Satellite identification variables (set per-chunk)
            chunk_satellite_uris = set()
            chunk_parent_satellites = {}

            if use_fr:
                # FR-based: build graph indexes for FR traversal (replaces BFS context)
                fr_outgoing, fr_incoming, entity_labels_map, entity_types_map, chunk_raw_triples = \
                    self._build_fr_graph_for_chunk(chunk_uris, chunk_types, chunk_literals)
                all_triples.extend(chunk_raw_triples)
                logger.info(f"    FR graph built: {len(chunk_raw_triples)} raw triples")

                # Identify satellites for this chunk
                chunk_satellite_uris, chunk_parent_satellites = self._identify_satellites_from_prefetched(
                    chunk_types, fr_incoming, entity_labels_map,
                    all_literals=chunk_literals,
                )
                all_satellite_uris.update(chunk_satellite_uris)
            else:
                # Legacy BFS context
                chunk_contexts = self.get_entities_context_batch(chunk_uris)
                logger.info(f"    Contexts: {len(chunk_contexts)} entities")

            chunk_wikidata = self._batch_fetch_wikidata_ids(chunk_uris)
            logger.info(f"    Wikidata IDs: {len(chunk_wikidata)} entities")

            # Phase B: Generate docs from pre-fetched data (zero SPARQL queries)
            # Filter out satellite entities
            doc_entities = [e for e in chunk_entities if e["entity"] not in chunk_satellite_uris]
            logger.info(f"  Phase B: Generating {len(doc_entities)} documents "
                        f"(skipping {len(chunk_satellite_uris)} satellites)"
                        f"{' (FR)' if use_fr else ' (BFS)'}...")
            chunk_docs = []  # List of (entity_uri, doc_text, metadata, cached_embedding)

            for entity in tqdm(doc_entities, desc=f"Chunk {chunk_num}", unit="entity"):
                entity_uri = entity["entity"]

                try:
                    literals = chunk_literals.get(entity_uri, {})
                    types = chunk_types.get(entity_uri, set())

                    if use_fr:
                        # FR-based document creation
                        # Build type labels for this entity
                        entity_type_labels = []
                        for type_uri in types:
                            type_label = None
                            if UniversalRagSystem._class_labels:
                                type_label = UniversalRagSystem._class_labels.get(type_uri)
                            if not type_label:
                                type_label = chunk_type_labels.get(type_uri)
                            if not type_label:
                                type_label = type_uri.split('/')[-1].split('#')[-1]
                            entity_type_labels.append(type_label)

                        # Extract entity label
                        entity_label = entity_uri.split('/')[-1]
                        for label_prop in ['label', 'prefLabel', 'name', 'title']:
                            if label_prop in literals and literals[label_prop]:
                                entity_label = literals[label_prop][0]
                                break

                        # Get absorbed satellite info
                        sat_info = chunk_parent_satellites.get(entity_uri)
                        absorbed_lines = None
                        if sat_info:
                            absorbed_lines = self.fr_traversal.format_absorbed_satellites(
                                dict(sat_info), entity_label
                            )

                        doc_text, entity_label, entity_types = self._create_fr_document_from_prefetched(
                            entity_uri, entity_label, types, entity_type_labels,
                            literals, fr_outgoing, fr_incoming,
                            entity_labels_map, entity_types_map,
                            absorbed_lines=absorbed_lines,
                        )
                    else:
                        # Legacy BFS document creation
                        context = chunk_contexts.get(entity_uri, ([], []))
                        doc_text, entity_label, entity_types, raw_triples = self._create_document_from_prefetched(
                            entity_uri, literals, types, context, chunk_type_labels
                        )
                        all_triples.extend(raw_triples)

                    # Determine primary entity type
                    primary_type = "Unknown"
                    human_readable_types = []
                    if entity_types:
                        human_readable_types = [
                            t for t in entity_types
                            if not self.is_technical_class_name(t)
                        ]
                        primary_type = human_readable_types[0] if human_readable_types else "Entity"

                    # Get Wikidata ID from pre-fetched batch data
                    wikidata_id = chunk_wikidata.get(entity_uri)

                    # Save document to disk with rich frontmatter
                    self.save_entity_document(
                        entity_uri, doc_text, entity_label,
                        entity_type=primary_type,
                        all_types=human_readable_types or None,
                        wikidata_id=wikidata_id,
                        images=image_index.get(entity_uri) or None,
                    )

                    metadata = {
                        "label": entity_label,
                        "type": primary_type,
                        "uri": entity_uri,
                        "all_types": entity_types,
                        "wikidata_id": wikidata_id,  # May be None
                        "images": image_index.get(entity_uri, [])
                    }

                    # Check embedding cache
                    cached_embedding = None
                    if self.embedding_cache:
                        cached_embedding = self.embedding_cache.get(entity_uri)
                        if cached_embedding:
                            cached_count += 1

                    chunk_docs.append((entity_uri, doc_text, metadata, cached_embedding))

                except Exception as e:
                    logger.error(f"Error processing entity {entity_uri}: {str(e)}")
                    continue

            # Phase C: Free pre-fetched data before embedding
            del chunk_literals, chunk_types, chunk_type_uris, chunk_type_labels, chunk_wikidata
            if fr_outgoing is not None:
                del fr_outgoing, fr_incoming, entity_labels_map, entity_types_map
            if chunk_contexts is not None:
                del chunk_contexts

            # Phase D: Embed in sub-batches
            logger.info(f"  Phase D: Embedding {len(chunk_docs)} documents...")
            for sub_idx in range(0, len(chunk_docs), embedding_batch_size):
                sub_batch = chunk_docs[sub_idx:sub_idx + embedding_batch_size]

                if self.use_batch_embedding:
                    self._process_batch_embeddings(sub_batch)
                else:
                    global_token_count, last_reset_time = self._process_sequential_embeddings(
                        sub_batch, global_token_count, last_reset_time, tokens_per_min_limit
                    )

                    # For API-based embeddings, pause between sub-batches
                    logger.info(f"    Completed sub-batch of {len(sub_batch)} documents, pausing for 2 seconds...")
                    time.sleep(2)

            # Phase E: Save progress after each chunk
            self.document_store.save_document_graph(self._get_document_graph_temp_path())
            logger.info(f"  Chunk {chunk_num}/{total_chunks} complete, progress saved")

        if cached_count > 0:
            logger.info(f"Used {cached_count} cached embeddings")

        # Filter satellite triples from edges before saving
        if all_satellite_uris and all_triples:
            pre_filter = len(all_triples)
            all_triples = [
                t for t in all_triples
                if t["subject"] not in all_satellite_uris and t["object"] not in all_satellite_uris
            ]
            logger.info(f"Filtered {pre_filter - len(all_triples)} satellite triples from edges")

        # Save triples and build edges from Parquet (avoids redundant SPARQL queries)
        if all_triples:
            self._save_edges_parquet(all_triples)
        logger.info("Creating document graph edges...")
        self._build_edges_from_parquet()
        
        # Rename temp file to final
        temp_path = self._get_document_graph_temp_path()
        final_path = self._get_document_graph_path()
        if os.path.exists(temp_path):
            os.replace(temp_path, final_path)
        
        # Build vector store with batched embedding requests
        logger.info("Building vector store...")
        self.build_vector_store_batched()

        # Build triples index from Parquet for query-time lookup
        self._load_triples_index()

        # Generate validation report for missing classes and properties
        self.generate_validation_report()

        logger.info("RDF data processing complete with enhanced CIDOC-CRM understanding")

    def _process_batch_embeddings(self, batch_docs):
        """
        Process embeddings for a batch of documents using batch embedding.
        Uses concurrent embedding when available (OpenAI) or standard batch (local).

        Args:
            batch_docs: List of (entity_uri, doc_text, metadata, cached_embedding)
        """
        # Separate cached and uncached documents
        cached_docs = [(uri, text, meta, emb) for uri, text, meta, emb in batch_docs if emb is not None]
        uncached_docs = [(uri, text, meta, emb) for uri, text, meta, emb in batch_docs if emb is None]

        # Add cached documents directly to store
        for entity_uri, doc_text, metadata, embedding in cached_docs:
            self.document_store.add_document_with_embedding(
                entity_uri, doc_text, embedding, metadata
            )

        # Generate embeddings for uncached documents in batch
        if uncached_docs:
            texts = [doc[1] for doc in uncached_docs]
            logger.info(f"Generating embeddings for {len(texts)} documents in batch...")

            # Check if provider supports concurrent embedding (OpenAI with ThreadPoolExecutor)
            use_concurrent = (
                hasattr(self.embedding_provider, 'supports_concurrent_embedding') and
                self.embedding_provider.supports_concurrent_embedding()
            )

            if use_concurrent:
                try:
                    logger.info("Using concurrent batch embedding for faster processing")
                    embeddings = self.embedding_provider.get_embeddings_batch_concurrent(texts)
                except Exception as e:
                    logger.warning(f"Concurrent embedding failed: {e}. Falling back to sequential batch.")
                    embeddings = self.embedding_provider.get_embeddings_batch(texts)
            else:
                embeddings = self.embedding_provider.get_embeddings_batch(texts)

            # Add documents and cache embeddings
            new_cached_ids = []
            for (entity_uri, doc_text, metadata, _), embedding in zip(uncached_docs, embeddings):
                self.document_store.add_document_with_embedding(
                    entity_uri, doc_text, embedding, metadata
                )

                # Cache the embedding for resumability
                if self.embedding_cache:
                    self.embedding_cache.set(entity_uri, embedding)
                    new_cached_ids.append(entity_uri)

            # Update cache metadata
            if self.embedding_cache and new_cached_ids:
                self.embedding_cache.update_metadata(new_cached_ids)

            logger.info(f"Added {len(uncached_docs)} documents with new embeddings")

        if cached_docs:
            logger.info(f"Added {len(cached_docs)} documents with cached embeddings")

    def _process_sequential_embeddings(self, batch_docs, global_token_count, last_reset_time, tokens_per_min_limit):
        """
        Process embeddings sequentially with rate limiting.
        This is the path for API-based embeddings (OpenAI, etc.).

        Args:
            batch_docs: List of (entity_uri, doc_text, metadata, cached_embedding)
            global_token_count: Current token count for rate limiting
            last_reset_time: Last time the token count was reset
            tokens_per_min_limit: Maximum tokens per minute

        Returns:
            Tuple of (updated_token_count, updated_reset_time)
        """
        for entity_uri, doc_text, metadata, cached_embedding in batch_docs:
            # Rate limit check - reset counter if a minute has passed
            current_time = time.time()
            if current_time - last_reset_time >= 60:
                global_token_count = 0
                last_reset_time = current_time

            # Skip if we're approaching the limit - wait just enough time
            if global_token_count > tokens_per_min_limit:
                wait_time = 60 - (current_time - last_reset_time) + 1
                if wait_time > 0:
                    logger.info(f"Approaching rate limit. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    global_token_count = 0
                    last_reset_time = time.time()

            # Use cached embedding if available
            if cached_embedding is not None:
                self.document_store.add_document_with_embedding(
                    entity_uri, doc_text, cached_embedding, metadata
                )
            else:
                # Generate embedding and add to store
                self.document_store.add_document(entity_uri, doc_text, metadata)

                # Cache the embedding for resumability
                if self.embedding_cache and entity_uri in self.document_store.docs:
                    embedding = self.document_store.docs[entity_uri].embedding
                    if embedding:
                        self.embedding_cache.set(entity_uri, embedding)

            # Estimate token count for rate limiting
            estimated_tokens = len(doc_text) / 4
            global_token_count += estimated_tokens

        return global_token_count, last_reset_time

    def build_vector_store_batched(self):
        """
        Build vector store using pre-computed embeddings from GraphDocument objects.
        This avoids redundant API calls since embeddings were already generated
        during document graph building.
        """

        vector_index_path = self._get_vector_index_dir()
        os.makedirs(vector_index_path, exist_ok=True)

        # Separate documents with and without pre-computed embeddings
        docs_with_embeddings = []
        docs_without_embeddings = []

        for doc_id, graph_doc in self.document_store.docs.items():
            if graph_doc.embedding is not None:
                docs_with_embeddings.append((doc_id, graph_doc))
            else:
                docs_without_embeddings.append((doc_id, graph_doc))

        total_docs = len(self.document_store.docs)
        logger.info(f"Building vector store with {total_docs} documents")
        logger.info(f"  - {len(docs_with_embeddings)} with pre-computed embeddings (no API calls)")
        logger.info(f"  - {len(docs_without_embeddings)} without embeddings (will generate)")

        vector_store = None

        # Build from pre-computed embeddings (fast, no API calls)
        if docs_with_embeddings:
            text_embeddings = []
            metadatas = []

            for doc_id, graph_doc in docs_with_embeddings:
                text_embeddings.append((graph_doc.text, graph_doc.embedding))
                metadatas.append({**graph_doc.metadata, "doc_id": doc_id})

            logger.info(f"Creating FAISS index from {len(text_embeddings)} pre-computed embeddings...")
            vector_store = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            logger.info("FAISS index created from pre-computed embeddings (no API calls)")

        # Add documents without embeddings (will generate via API)
        if docs_without_embeddings:
            logger.warning(f"Generating embeddings for {len(docs_without_embeddings)} documents via API...")
            docs_for_faiss = []
            for doc_id, graph_doc in docs_without_embeddings:
                doc = Document(
                    page_content=graph_doc.text,
                    metadata={**graph_doc.metadata, "doc_id": doc_id}
                )
                docs_for_faiss.append(doc)

            if vector_store is None:
                vector_store = FAISS.from_documents(docs_for_faiss, self.embeddings)
            else:
                vector_store.add_documents(docs_for_faiss)

        # Save and store
        if vector_store:
            vector_store.save_local(vector_index_path)
            self.document_store.vector_store = vector_store
            logger.info(f"Vector store built and saved with {total_docs} documents")
        else:
            logger.error("No documents to build vector store from")



    def cidoc_aware_retrieval(self, query, k=20):
        """Retrieve candidate documents using FAISS vector similarity.

        Returns (documents, scores) tuples preserving actual FAISS similarity
        scores for use as initial relevance signal in coherent subgraph extraction.
        """
        results_with_scores = self.document_store.retrieve(query, k=k)

        if not results_with_scores:
            return []

        return results_with_scores

    # ==================== Query Analysis ====================

    _fc_class_mapping = None  # Class-level cache: FC name → list of CRM class URIs

    @classmethod
    def _load_fc_class_mapping(cls) -> Dict[str, List[str]]:
        """Load FC class mapping from config/fc_class_mapping.json.

        Returns dict mapping FC names (Thing, Actor, ...) to lists of CRM class
        local names (E22_Human-Made_Object, etc.).
        """
        if cls._fc_class_mapping is not None:
            return cls._fc_class_mapping

        fc_path = str(PROJECT_ROOT / 'config' / 'fc_class_mapping.json')
        if not os.path.exists(fc_path):
            logger.warning(f"FC class mapping not found: {fc_path}")
            cls._fc_class_mapping = {}
            return cls._fc_class_mapping

        with open(fc_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        # Filter out comment keys
        cls._fc_class_mapping = {k: v for k, v in raw.items() if not k.startswith('_')}
        total = sum(len(v) for v in cls._fc_class_mapping.values())
        logger.info(f"Loaded FC class mapping: {len(cls._fc_class_mapping)} categories, {total} classes")
        return cls._fc_class_mapping

    def _analyze_query(self, question: str) -> 'QueryAnalysis':
        """Classify a user question using the LLM for query-type-aware retrieval.

        Returns a QueryAnalysis with query_type and target FC categories.
        Falls back to SPECIFIC with no categories on failure.
        """
        valid_types = {"SPECIFIC", "ENUMERATION", "AGGREGATION"}
        valid_categories = {"Thing", "Actor", "Place", "Event", "Concept", "Time"}

        try:
            prompt = QUERY_ANALYSIS_PROMPT.format(question=question)
            raw = self.llm_provider.generate("You are a query classifier. Return only valid JSON.", prompt)

            # Extract JSON from response (handle markdown code blocks)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            qtype = parsed.get("query_type", "SPECIFIC").upper()
            cats = parsed.get("categories", [])

            if qtype not in valid_types:
                qtype = "SPECIFIC"
            cats = [c for c in cats if c in valid_categories]

            result = QueryAnalysis(query_type=qtype, categories=cats)
            logger.info(f"Query analysis: type={result.query_type}, categories={result.categories}")
            return result

        except Exception as e:
            logger.warning(f"Query analysis failed ({e}), defaulting to SPECIFIC")
            return QueryAnalysis(query_type="SPECIFIC", categories=[])

    def get_cidoc_system_prompt(self):
        """Get a system prompt with CIDOC-CRM knowledge"""

        return """You are a cultural heritage expert who provides clear, accessible answers about artworks, churches, frescoes, and historical figures.

Rules:
- ONLY use information from the retrieved context provided below. Never invent or guess.
- Write in clear, natural language. Translate ontological vocabulary into everyday words: "denotes" → "depicts", "is composed of" → "consists of", "bears feature" → "has". Never use ontology codes (E22_, IC9_, D1_, P62_, etc.) or raw Wikidata codes.
- Use specific entity names (e.g. "Panagia Phorbiottisa") not generic terms.
- When describing a panel or artwork, explain WHO is depicted in it and WHAT ROLE each figure has (saint, donor, founder, etc.). If a panel depicts multiple figures, explicitly state that they appear together and describe each one.
- When asked about "other" entities of a type, list ALL matching entities from the context, even if they are from different locations. Include entities even if you cannot determine their exact location — state what you know and note what is unclear. Do not say "no other" unless you have truly found none in the context.
- Answer directly and concisely. Do not add disclaimers about data limitations unless you truly have no relevant information at all.
- If conversation history is provided, use it to understand what the user is referring to (e.g. "the church" = the church discussed earlier).
"""

    def get_all_entities(self):
        """Get all entities that have literal properties from SPARQL endpoint"""
        query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?entity ?property ?value
        WHERE {
            ?entity ?property ?value .
            FILTER(isLiteral(?value))

            # Exclude ontology schema elements (classes and properties)
            FILTER NOT EXISTS {
                ?entity rdf:type ?type .
                VALUES ?type {
                    rdfs:Class
                    owl:Class
                    rdf:Property
                    owl:ObjectProperty
                    owl:DatatypeProperty
                    owl:AnnotationProperty
                    owl:FunctionalProperty
                    owl:InverseFunctionalProperty
                    owl:TransitiveProperty
                    owl:SymmetricProperty
                }
            }
        }
        """

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            # Group literals by entity
            entity_map = {}
            for result in results["results"]["bindings"]:
                entity_uri = result["entity"]["value"]
                prop = result["property"]["value"]
                value = result["value"]["value"]

                if entity_uri not in entity_map:
                    entity_map[entity_uri] = {}

                # Store literals by property, handling multiple values
                prop_name = prop.split('/')[-1].split('#')[-1]
                if prop_name not in entity_map[entity_uri]:
                    entity_map[entity_uri][prop_name] = []
                entity_map[entity_uri][prop_name].append(value)

            # Convert to list format with labels, filtering out ontology classes
            entities = []
            ontology_classes = UniversalRagSystem._ontology_classes or set()
            skipped_classes = 0

            for entity_uri, literals in entity_map.items():
                # Skip ontology class URIs (e.g., E41_Appellation, IC10_Attribute)
                if entity_uri in ontology_classes:
                    skipped_classes += 1
                    continue

                # Also skip URIs that look like ontology definitions
                # (cidoc-crm namespace, vir namespace with class patterns)
                if any(pattern in entity_uri for pattern in [
                    'cidoc-crm.org/cidoc-crm/E',
                    'cidoc-crm.org/cidoc-crm/P',
                    'w3id.org/vir#IC',
                    'w3id.org/vir#K',
                    'ics.forth.gr/isl/CRMdig/',
                    'cidoc-crm.org/extensions/'
                ]):
                    skipped_classes += 1
                    continue

                # Try to find a label from various common properties
                label = entity_uri.split('/')[-1]  # Default to URI fragment
                for label_prop in ['label', 'prefLabel', 'name', 'title']:
                    if label_prop in literals and literals[label_prop]:
                        label = literals[label_prop][0]
                        break

                entities.append({
                    "entity": entity_uri,
                    "label": label,
                    "literals": literals
                })

            if skipped_classes > 0:
                logger.info(f"Skipped {skipped_classes} ontology class URIs")
            logger.info(f"Retrieved {len(entities)} entities with literals")
            return entities
        except Exception as e:
            logger.error(f"Error fetching entities: {str(e)}")
            return []

    def get_relationship_weight(self, predicate_uri):
        """Get weight for a CIDOC-CRM relationship predicate."""
        return _get_relationship_weight(predicate_uri)

    def normalize_scores(self, scores):
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: Dictionary of {id: score} or list of scores

        Returns:
            Dictionary or list with normalized scores in [0, 1] range
        """

        if isinstance(scores, dict):
            if not scores:
                return scores

            values = np.array(list(scores.values()))
            min_val = np.min(values)
            max_val = np.max(values)

            # Handle case where all scores are the same
            if max_val - min_val < 1e-10:
                return {k: 1.0 for k in scores.keys()}

            # Min-max normalization
            normalized = {k: float((v - min_val) / (max_val - min_val))
                         for k, v in scores.items()}
            return normalized
        else:
            # Handle list/array
            values = np.array(scores)
            min_val = np.min(values)
            max_val = np.max(values)

            if max_val - min_val < 1e-10:
                return np.ones_like(values)

            return (values - min_val) / (max_val - min_val)

    def _fetch_wikidata_id_from_sparql(self, entity_uri):
        """Fetch Wikidata ID from SPARQL endpoint (used during build phase).

        This is a private method for build-time use. At query time,
        use get_wikidata_for_entity() which checks cached metadata first.
        """
        query = f"""
        PREFIX crmdig: <http://www.ics.forth.gr/isl/CRMdig/>

        SELECT ?wikidata WHERE {{
            <{entity_uri}> crmdig:L54_is_same-as ?wikidata .
            FILTER(STRSTARTS(STR(?wikidata), "http://www.wikidata.org/entity/"))
        }}
        LIMIT 1
        """

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            if results["results"]["bindings"]:
                wikidata_uri = results["results"]["bindings"][0]["wikidata"]["value"]
                # Extract the Q-ID from the URI
                wikidata_id = wikidata_uri.split('/')[-1]
                return wikidata_id
            return None
        except Exception as e:
            logger.debug(f"Could not fetch Wikidata ID for {entity_uri}: {str(e)}")
            return None

    def get_wikidata_for_entity(self, entity_uri):
        """Get Wikidata ID for an entity if available.

        First checks document metadata (cached from build time),
        then falls back to SPARQL query if needed.
        """
        # First, check if Wikidata ID is in document metadata (no SPARQL needed)
        if entity_uri in self.document_store.docs:
            doc = self.document_store.docs[entity_uri]
            wikidata_id = doc.metadata.get("wikidata_id")
            if wikidata_id:
                return wikidata_id

        # Fall back to SPARQL query (only if endpoint is available)
        query = f"""
        PREFIX crmdig: <http://www.ics.forth.gr/isl/CRMdig/>

        SELECT ?wikidata WHERE {{
            <{entity_uri}> crmdig:L54_is_same-as ?wikidata .
            FILTER(STRSTARTS(STR(?wikidata), "http://www.wikidata.org/entity/"))
        }}
        LIMIT 1
        """

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            if results["results"]["bindings"]:
                wikidata_uri = results["results"]["bindings"][0]["wikidata"]["value"]
                # Extract the Q-ID from the URI
                wikidata_id = wikidata_uri.split('/')[-1]
                return wikidata_id
            return None
        except Exception as e:
            # Log at debug level since this is expected when SPARQL is unavailable
            logger.debug(f"Could not fetch Wikidata ID from SPARQL: {str(e)}")
            return None

    def fetch_wikidata_info(self, wikidata_id):
        """Fetch information from Wikidata for a given Q-ID"""
        
        # Handle rate limits with retries
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Use the Wikidata API to get entity data
                url = f"https://www.wikidata.org/w/api.php"
                params = {
                    "action": "wbgetentities",
                    "ids": wikidata_id,
                    "format": "json",
                    "languages": "en",
                    "props": "labels|descriptions|claims|sitelinks"
                }
                
                # Add proper headers including User-Agent
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; RAG-Bot/1.0; +http://example.com/bot)',
                    'Accept': 'application/json'
                }

                # Use secure session instead of direct requests call
                response = self._http_session.get(url, params=params, headers=headers, timeout=10)
                
                # Check if response is empty
                if not response.text:
                    logger.warning(f"Empty response from Wikidata API for {wikidata_id} (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                
                # Check status code
                if response.status_code != 200:
                    logger.warning(f"Wikidata API returned status {response.status_code} for {wikidata_id} (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                
                try:
                    data = response.json()
                except ValueError as e:
                    logger.warning(f"Failed to parse JSON from Wikidata API for {wikidata_id} (attempt {attempt+1}/{max_retries}): {str(e)}")
                    logger.debug(f"Response text: {response.text[:200]}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                
                if "entities" in data and wikidata_id in data["entities"]:
                    entity = data["entities"][wikidata_id]
                    
                    # Extract useful information
                    result = {
                        "id": wikidata_id,
                        "url": f"https://www.wikidata.org/wiki/{wikidata_id}"
                    }
                    
                    # Get label
                    if "labels" in entity and "en" in entity["labels"]:
                        result["label"] = entity["labels"]["en"]["value"]
                    
                    # Get description
                    if "descriptions" in entity and "en" in entity["descriptions"]:
                        result["description"] = entity["descriptions"]["en"]["value"]
                    
                    # Get Wikipedia link if available
                    if "sitelinks" in entity and "enwiki" in entity["sitelinks"]:
                        result["wikipedia"] = {
                            "title": entity["sitelinks"]["enwiki"]["title"],
                            "url": f"https://en.wikipedia.org/wiki/{entity['sitelinks']['enwiki']['title'].replace(' ', '_')}"
                        }
                    
                    # Get selected property values (customize these as needed)
                    if "claims" in entity:
                        result["properties"] = {}
                        
                        # Map of interesting Wikidata properties and their human-readable names
                        property_map = {
                            "P18": "image",           # image
                            "P571": "inception",      # date created/founded
                            "P17": "country",         # country
                            "P131": "located_in",     # administrative territorial entity
                            "P625": "coordinates",    # coordinate location
                            "P1343": "described_by",  # described by source
                            "P138": "named_after",    # named after
                            "P180": "depicts",        # depicts
                            "P31": "instance_of",     # instance of
                            "P276": "location"        # location
                        }
                        
                        for prop_id, prop_name in property_map.items():
                            if prop_id in entity["claims"]:
                                values = []
                                for claim in entity["claims"][prop_id]:
                                    if "mainsnak" in claim and "datavalue" in claim["mainsnak"]:
                                        datavalue = claim["mainsnak"]["datavalue"]
                                        
                                        if datavalue["type"] == "wikibase-entityid":
                                            # For entity references, we just store the Q-ID
                                            values.append(datavalue["value"]["id"])
                                        elif datavalue["type"] == "string":
                                            # For string values
                                            values.append(datavalue["value"])
                                        elif datavalue["type"] == "time":
                                            # For time values
                                            values.append(datavalue["value"]["time"])
                                        elif datavalue["type"] == "globecoordinate":
                                            # For coordinates
                                            values.append({
                                                "latitude": datavalue["value"]["latitude"],
                                                "longitude": datavalue["value"]["longitude"]
                                            })
                                
                                # Only add property if we found values
                                if values:
                                    result["properties"][prop_name] = values[0] if len(values) == 1 else values
                    
                    return result
                else:
                    logger.warning(f"No entity data found for {wikidata_id} in response")
                    return None
                
            except requests.exceptions.Timeout:
                logger.warning(f"Wikidata API request timeout for {wikidata_id} (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
            except requests.exceptions.RequestException as e:
                logger.warning(f"Wikidata API request failed for {wikidata_id} (attempt {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2
            except Exception as e:
                logger.error(f"Unexpected error fetching Wikidata info for {wikidata_id} (attempt {attempt+1}/{max_retries}): {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2
        
        logger.error(f"Failed to fetch Wikidata info after {max_retries} attempts for {wikidata_id}")
        return None


    def compute_coherent_subgraph(self, candidates, adjacency_matrix, initial_scores, k=RetrievalConfig.DEFAULT_RETRIEVAL_K, alpha=RetrievalConfig.RELEVANCE_CONNECTIVITY_ALPHA, query_analysis=None):
        """
        Extract a coherent subgraph using greedy selection that balances individual relevance and connectivity.

        Args:
            candidates: List of GraphDocument objects
            adjacency_matrix: Weighted adjacency matrix (n x n)
            initial_scores: Initial relevance scores for each candidate (n,)
            k: Number of documents to select
            alpha: Weight for individual relevance vs connectivity (0-1, higher = more emphasis on relevance)
            query_analysis: Optional QueryAnalysis for FC-aware type boosting

        Returns:
            List of selected GraphDocument objects in order of selection
        """
        n = len(candidates)
        selected_indices = []
        selected_mask = np.zeros(n, dtype=bool)

        # Normalize initial scores to [0, 1] using min-max normalization
        normalized_scores = self.normalize_scores(initial_scores)

        # Pre-compute cosine similarity matrix for MMR diversity penalty
        diversity_penalty_weight = RetrievalConfig.DIVERSITY_PENALTY
        embeddings = []
        for c in candidates:
            if c.embedding is not None:
                embeddings.append(c.embedding)
            else:
                embeddings.append(np.zeros(1))
        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        emb_normalized = embeddings / norms
        sim_matrix = emb_normalized @ emb_normalized.T

        # Pre-compute type-based score modifiers for all candidates
        type_modifiers = RetrievalConfig.TYPE_SCORE_MODIFIERS
        candidate_type_mods = np.zeros(n)
        for i, c in enumerate(candidates):
            primary_type = c.metadata.get('type', '')
            all_types = c.metadata.get('all_types', [])
            # Check primary type first, then fall back to all_types
            mod = type_modifiers.get(primary_type, None)
            if mod is None and all_types:
                for t in all_types:
                    mod = type_modifiers.get(t)
                    if mod is not None:
                        break
            candidate_type_mods[i] = mod if mod is not None else 0.0

        # FC-aware boosting: boost candidates whose types match query target categories
        fc_boost_count = 0
        FC_BOOST = 0.10  # +10% score boost for FC-matching candidates
        if query_analysis and query_analysis.categories:
            fc_mapping = self._load_fc_class_mapping()
            target_classes = set()
            for fc_name in query_analysis.categories:
                target_classes.update(fc_mapping.get(fc_name, []))

            if target_classes:
                for i, c in enumerate(candidates):
                    all_types = set(c.metadata.get('all_types', []))
                    if all_types & target_classes:
                        candidate_type_mods[i] += FC_BOOST
                        fc_boost_count += 1
                logger.info(f"FC-aware boosting: {fc_boost_count}/{n} candidates match "
                            f"target categories {query_analysis.categories} (+{FC_BOOST})")

        logger.info(f"\n{'='*80}")
        logger.info(f"COHERENT SUBGRAPH EXTRACTION")
        logger.info(f"{'='*80}")
        logger.info(f"Parameters: k={k}, alpha={alpha} (relevance weight), diversity_penalty={diversity_penalty_weight}")
        logger.info(f"Candidates: {n} documents")
        logger.info(f"\n--- Initial Relevance Scores (normalized to [0,1]) ---")
        logger.info(f"  Min: {np.min(normalized_scores):.3f}")
        logger.info(f"  Max: {np.max(normalized_scores):.3f}")
        logger.info(f"  Mean: {np.mean(normalized_scores):.3f}")
        logger.info(f"  Std: {np.std(normalized_scores):.3f}")

        # Log type modifier distribution
        boosted = np.sum(candidate_type_mods > 0)
        penalized = np.sum(candidate_type_mods < 0)
        neutral = np.sum(candidate_type_mods == 0)
        logger.info(f"\n--- Type-Based Score Modifiers ---")
        logger.info(f"  Boosted: {boosted}, Penalized: {penalized}, Neutral: {neutral}")

        # Show top 5 initial candidates
        logger.info(f"\n--- Top 5 Initial Candidates (by relevance) ---")
        top_indices = np.argsort(normalized_scores)[::-1][:5]
        for rank, idx in enumerate(top_indices, 1):
            label = candidates[idx].metadata.get('label', 'Unknown')
            etype = candidates[idx].metadata.get('type', '')
            mod_str = f" [type_mod={candidate_type_mods[idx]:+.2f}]" if candidate_type_mods[idx] != 0 else ""
            logger.info(f"  {rank}. {label} ({etype}): {normalized_scores[idx]:.3f}{mod_str}")

        # First selection: pick the highest-scoring document (with type modifier applied)
        first_round_scores = normalized_scores * (1.0 + candidate_type_mods)
        first_idx = np.argmax(first_round_scores)
        selected_indices.append(first_idx)
        selected_mask[first_idx] = True
        logger.info(f"\n{'='*80}")
        logger.info(f"SELECTION ROUND 1/{k}")
        logger.info(f"{'='*80}")
        logger.info(f"Strategy: Select highest relevance score (with type modifier)")
        first_mod = candidate_type_mods[first_idx]
        first_mod_str = f", type_mod={first_mod:+.2f}" if first_mod != 0 else ""
        logger.info(f"Selected: {candidates[first_idx].metadata.get('label', 'Unknown')} (score={first_round_scores[first_idx]:.3f}{first_mod_str})")

        # Iteratively select remaining documents
        for iteration in range(1, k):
            if len(selected_indices) >= n:
                break

            # Collect connectivity scores for all unselected candidates
            connectivity_scores = []
            candidate_indices = []

            for idx in range(n):
                if selected_mask[idx]:
                    continue

                # Connectivity component: sum of weighted edges to already-selected documents
                connectivity = 0.0
                for selected_idx in selected_indices:
                    # Check both directions in adjacency matrix
                    edge_weight = max(
                        adjacency_matrix[idx, selected_idx],
                        adjacency_matrix[selected_idx, idx]
                    )
                    connectivity += edge_weight

                # Average by number of selected documents to avoid bias toward later iterations
                if len(selected_indices) > 0:
                    connectivity = connectivity / len(selected_indices)

                connectivity_scores.append(connectivity)
                candidate_indices.append(idx)

            # Normalize connectivity scores to [0, 1] for this iteration
            if len(connectivity_scores) > 0:
                connectivity_array = np.array(connectivity_scores)
                normalized_connectivity = self.normalize_scores(connectivity_array)
            else:
                break

            logger.info(f"\n{'='*80}")
            logger.info(f"SELECTION ROUND {iteration+1}/{k}")
            logger.info(f"{'='*80}")
            logger.info(f"Strategy: Combine relevance ({alpha:.1f}) + connectivity ({1-alpha:.1f})")
            logger.info(f"Connectivity scores (normalized): min={np.min(normalized_connectivity):.3f}, "
                       f"max={np.max(normalized_connectivity):.3f}, mean={np.mean(normalized_connectivity):.3f}")

            # Compute all scores and collect top candidates
            all_scores = []
            for i, idx in enumerate(candidate_indices):
                relevance = normalized_scores[idx]
                connectivity_norm = normalized_connectivity[i]
                # MMR diversity penalty: penalize similarity to most-similar already-selected doc
                max_sim = max(sim_matrix[idx, sel_idx] for sel_idx in selected_indices)
                div_penalty = diversity_penalty_weight * max_sim
                base_score = alpha * relevance + (1 - alpha) * connectivity_norm - div_penalty
                # Apply type-based modifier
                type_mod = candidate_type_mods[idx]
                combined_score = base_score * (1.0 + type_mod)
                all_scores.append((idx, combined_score, relevance, connectivity_norm, div_penalty, type_mod))

            # Sort by combined score
            all_scores.sort(key=lambda x: x[1], reverse=True)

            # Show top 3 candidates for this iteration
            logger.info(f"\n--- Top 3 Candidates for Round {iteration+1} ---")
            for rank, (idx, combined, rel, conn, div_pen, t_mod) in enumerate(all_scores[:3], 1):
                label = candidates[idx].metadata.get('label', 'Unknown')
                etype = candidates[idx].metadata.get('type', '')
                logger.info(f"  {rank}. {label} ({etype})")
                logger.info(f"      Relevance: {rel:.3f} (weight={alpha:.1f}) → contrib={alpha*rel:.3f}")
                logger.info(f"      Connectivity: {conn:.3f} (weight={1-alpha:.1f}) → contrib={(1-alpha)*conn:.3f}")
                logger.info(f"      Diversity penalty: -{div_pen:.3f}")
                type_mod_str = f", type_mod={t_mod:+.2f}" if t_mod != 0 else ""
                logger.info(f"      Combined: {combined:.3f}{type_mod_str}")

            # Select the best
            best_idx, best_score, best_rel, best_connectivity_norm, best_div_penalty, best_type_mod = all_scores[0]

            if best_idx == -1:
                logger.warning(f"Could not find more connected documents after {len(selected_indices)} selections")
                break

            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            best_type_mod_str = f" * (1{best_type_mod:+.2f})" if best_type_mod != 0 else ""
            logger.info(f"\n✓ SELECTED: {candidates[best_idx].metadata.get('label', 'Unknown')}")
            logger.info(f"  Final score: {best_score:.3f} = ({alpha:.1f}×{best_rel:.3f} + {1-alpha:.1f}×{best_connectivity_norm:.3f} - {best_div_penalty:.3f}){best_type_mod_str}")

        # Log final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"SUBGRAPH EXTRACTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Selected {len(selected_indices)}/{k} documents:")
        for i, idx in enumerate(selected_indices, 1):
            label = candidates[idx].metadata.get('label', 'Unknown')
            etype = candidates[idx].metadata.get('type', '')
            mod = candidate_type_mods[idx]
            mod_str = f", type_mod={mod:+.2f}" if mod != 0 else ""
            logger.info(f"  {i}. {label} [{etype}] (relevance={normalized_scores[idx]:.3f}{mod_str})")
        logger.info(f"{'='*80}\n")

        # Return selected documents in order
        return [candidates[idx] for idx in selected_indices]

    def _rrf_fuse(self, faiss_results, bm25_results, pool_size, k_rrf=60):
        """Reciprocal Rank Fusion of FAISS and BM25 ranked lists.

        Args:
            faiss_results: List of (GraphDocument, score) from FAISS.
            bm25_results: List of (GraphDocument, score) from BM25.
            pool_size: Maximum number of fused results to return.
            k_rrf: RRF smoothing constant (standard=60).

        Returns:
            List of (GraphDocument, rrf_score) sorted by fused score descending.
        """
        scores = {}
        doc_map = {}
        for rank, (doc, _) in enumerate(faiss_results):
            scores[doc.id] = scores.get(doc.id, 0) + 1.0 / (k_rrf + rank + 1)
            doc_map[doc.id] = doc
        for rank, (doc, _) in enumerate(bm25_results):
            scores[doc.id] = scores.get(doc.id, 0) + 1.0 / (k_rrf + rank + 1)
            doc_map[doc.id] = doc

        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:pool_size]

        # Log fusion stats
        faiss_ids = {doc.id for doc, _ in faiss_results}
        bm25_ids = {doc.id for doc, _ in bm25_results}
        overlap = faiss_ids & bm25_ids
        bm25_only = bm25_ids - faiss_ids
        logger.info(f"RRF fusion: {len(faiss_ids)} FAISS + {len(bm25_ids)} BM25 → "
                     f"{len(sorted_ids)} fused ({len(overlap)} overlap, {len(bm25_only)} BM25-only)")

        return [(doc_map[did], scores[did]) for did in sorted_ids]

    def retrieve(self, query, k=RetrievalConfig.DEFAULT_RETRIEVAL_K, initial_pool_size=60, alpha=RetrievalConfig.RELEVANCE_CONNECTIVITY_ALPHA, query_analysis=None):
        """
        Retrieve documents using hybrid FAISS+BM25 similarity + coherent subgraph extraction:
        1. FAISS vector retrieval + BM25 sparse retrieval → RRF fusion (initial pool)
        2. Adjacency matrix with virtual 2-hop edges through full graph
        3. Coherent subgraph extraction balancing relevance + connectivity

        Args:
            query: Query string
            k: Number of documents to return
            initial_pool_size: Size of initial candidate pool (should be > k)
            alpha: Balance between relevance (higher) and connectivity (lower)
            query_analysis: Optional QueryAnalysis for FC-aware type boosting in subgraph extraction
        """
        logger.info(f"Retrieving documents for query: '{query}'")

        # FAISS retrieval with actual similarity scores
        faiss_results = self.cidoc_aware_retrieval(query, k=initial_pool_size)

        if not faiss_results:
            logger.warning("No documents found in FAISS retrieval")
            return []

        # BM25 retrieval and RRF fusion
        bm25_results = self.document_store.retrieve_bm25(query, k=initial_pool_size)
        if bm25_results:
            results_with_scores = self._rrf_fuse(faiss_results, bm25_results, pool_size=initial_pool_size)
        else:
            results_with_scores = faiss_results

        initial_docs = [doc for doc, _ in results_with_scores]
        faiss_scores = np.array([score for _, score in results_with_scores])

        # If we got fewer documents than requested, just return them
        if len(initial_docs) <= k:
            logger.info(f"Retrieved {len(initial_docs)} documents (less than k={k})")
            return initial_docs

        # Create a subgraph of the retrieved documents
        doc_ids = [doc.id for doc in initial_docs]

        # Create weighted adjacency matrix with virtual 2-hop edges through full graph
        adjacency_matrix = self.document_store.create_adjacency_matrix(doc_ids, max_hops=RetrievalConfig.MAX_ADJACENCY_HOPS)

        # Extract coherent subgraph using actual FAISS similarity scores
        logger.info(f"Extracting coherent subgraph of size {k} from {len(initial_docs)} candidates")
        selected_docs = self.compute_coherent_subgraph(
            candidates=initial_docs,
            adjacency_matrix=adjacency_matrix,
            initial_scores=faiss_scores,
            k=k,
            alpha=alpha,
            query_analysis=query_analysis
        )

        logger.info(f"Retrieved and selected {len(selected_docs)} coherent documents")
        return selected_docs

    def _get_entity_label_from_triples(self, uri):
        """Find a human-readable label for a URI from the triples index."""
        triples = getattr(self, '_triples_index', {}).get(uri, [])
        for t in triples:
            if t.get('subject') == uri and t.get('subject_label'):
                return t['subject_label']
            if t.get('object') == uri and t.get('object_label'):
                return t['object_label']
        return uri.split('/')[-1]

    def _build_triples_enrichment(self, retrieved_docs):
        """Build structured triple enrichment text from _triples_index for retrieved documents.

        For each retrieved document, pulls its raw triples and formats them as
        human-readable relationship lines. Prioritizes inter-document triples and
        key CIDOC-CRM predicates (temporal, creator, location, exhibition, depiction).
        Does 1-hop enrichment for time-span entities to inline date values.

        Args:
            retrieved_docs: List of GraphDocument objects from retrieval.

        Returns:
            Formatted string with structured relationships, or empty string.
        """
        triples_index = getattr(self, '_triples_index', None)
        if not triples_index:
            return ""

        retrieved_uris = {doc.id for doc in retrieved_docs}

        # Predicates to skip: labels and technical metadata already in doc.text
        SKIP_PREDICATES = {
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            "http://www.w3.org/2000/01/rdf-schema#subClassOf",
            "http://www.w3.org/2000/01/rdf-schema#subPropertyOf",
            "http://www.w3.org/2002/07/owl#equivalentClass",
            "http://www.w3.org/2002/07/owl#inverseOf",
            "http://www.w3.org/2000/01/rdf-schema#comment",
            "http://www.w3.org/2000/01/rdf-schema#domain",
            "http://www.w3.org/2000/01/rdf-schema#range",
        }

        # Predicate label fragments to skip (case-insensitive check)
        SKIP_LABEL_FRAGMENTS = {"label", "type", "same as", "see also"}

        # High-priority predicate URI fragments (partial match on local name)
        HIGH_PRIORITY_FRAGMENTS = {
            "P4_has_time-span", "P82a_begin_of_the_begin", "P82b_end_of_the_end",
            "P14_carried_out_by", "P14i_performed", "P108_has_produced",
            "P108i_was_produced_by", "P7_took_place_at", "P55_has_current_location",
            "P53_has_former_or_current_location", "P89_falls_within",
            "P16_used_specific_object", "P16i_was_used_for",
            "P12_occurred_in_the_presence_of", "P12i_was_present_at",
            "P62_depicts", "P62i_is_depicted_by",
            "P46_is_composed_of", "P46i_forms_part_of",
            "P56_bears_feature", "P56i_is_found_on",
            "K24_portray", "K24i_portrayed_by",
            "P9_consists_of", "P9i_forms_part_of",
        }

        # Time-span predicates for 1-hop enrichment
        TIME_SPAN_PREDICATES = {
            "http://www.cidoc-crm.org/cidoc-crm/P4_has_time-span",
            "http://www.cidoc-crm.org/cidoc-crm/P4i_is_time-span_of",
        }
        DATE_PREDICATES = {
            "P82a_begin_of_the_begin",
            "P82b_end_of_the_end",
            "P82_at_some_time_within",
        }

        MAX_TRIPLES_PER_ENTITY = 15
        MAX_TOTAL_CHARS = 5000

        def _is_skip_label(pred_label):
            """Check if predicate label is too generic to be useful."""
            if not pred_label:
                return True
            lower = pred_label.strip().lower()
            if lower in SKIP_LABEL_FRAGMENTS:
                return True
            return False

        def _is_blank_or_hash(value):
            """Check if a value looks like a blank node or non-informative hash URI."""
            if not value:
                return True
            if value.startswith("_:"):
                return True
            # Hash URIs with no meaningful label
            if "#" in value and "/" not in value.split("#")[-1]:
                return True
            return False

        def _predicate_priority(triple, entity_uri):
            """Return priority score: 0 = highest (inter-doc), 1 = high, 2 = medium, 3 = low."""
            other_uri = triple["object"] if triple["subject"] == entity_uri else triple["subject"]
            # Highest: connects to another retrieved document
            if other_uri in retrieved_uris:
                return 0

            pred = triple.get("predicate", "")
            local_name = pred.split("/")[-1].split("#")[-1]

            # High: key CIDOC-CRM predicates
            for frag in HIGH_PRIORITY_FRAGMENTS:
                if frag in local_name:
                    return 1

            # Medium: other named predicates
            if triple.get("predicate_label"):
                return 2

            return 3

        def _resolve_time_span(time_span_uri):
            """Follow a time-span URI to extract begin/end date values."""
            ts_triples = triples_index.get(time_span_uri, [])
            dates = {}
            for t in ts_triples:
                if t["subject"] != time_span_uri:
                    continue
                pred_local = t["predicate"].split("/")[-1].split("#")[-1]
                for dp in DATE_PREDICATES:
                    if dp in pred_local:
                        obj_val = t.get("object_label") or t.get("object", "")
                        if obj_val and not obj_val.startswith("http"):
                            # Use a short key for output
                            if "begin" in dp:
                                dates["began"] = obj_val
                            elif "end" in dp:
                                dates["ended"] = obj_val
                            else:
                                dates["date"] = obj_val
            return dates

        # Build per-entity enrichment
        entity_sections = []
        total_chars = 0

        for doc in retrieved_docs:
            entity_uri = doc.id
            entity_label = doc.metadata.get("label", entity_uri.split("/")[-1])
            raw_triples = triples_index.get(entity_uri, [])
            if not raw_triples:
                continue

            # Filter and sort triples
            scored_triples = []
            for t in raw_triples:
                pred = t.get("predicate", "")
                pred_label = t.get("predicate_label", "")

                # Skip blacklisted predicates
                if pred in SKIP_PREDICATES:
                    continue
                if _is_skip_label(pred_label):
                    continue

                # Determine the "other" side of the triple relative to this entity
                if t["subject"] == entity_uri:
                    other_label = t.get("object_label", "")
                    other_uri = t.get("object", "")
                    direction = "outgoing"
                else:
                    other_label = t.get("subject_label", "")
                    other_uri = t.get("subject", "")
                    direction = "incoming"

                # Skip blank nodes / empty labels
                if _is_blank_or_hash(other_label) and _is_blank_or_hash(other_uri):
                    continue

                priority = _predicate_priority(t, entity_uri)
                scored_triples.append((priority, t, other_label, other_uri, direction))

            # Sort by priority (lower = better)
            scored_triples.sort(key=lambda x: x[0])

            # Format lines
            lines = []
            seen_lines = set()
            for priority, t, other_label, other_uri, direction in scored_triples[:MAX_TRIPLES_PER_ENTITY * 2]:
                if len(lines) >= MAX_TRIPLES_PER_ENTITY:
                    break

                pred_label = t.get("predicate_label", "")
                pred = t.get("predicate", "")

                # Use label if available, otherwise extract local name from URI
                display_label = other_label
                if not display_label or display_label.startswith("http"):
                    # Try to extract a readable local name
                    display_label = other_uri.split("/")[-1].split("#")[-1]
                    if not display_label or display_label == other_uri:
                        continue

                # Format the line depending on direction
                if direction == "outgoing":
                    line = f"  - {pred_label}: {display_label}"
                else:
                    # Incoming: show who/what points to this entity
                    line = f"  - [{display_label}] {pred_label}"

                # Deduplicate
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                lines.append(line)

                # 1-hop enrichment for time-span targets
                if direction == "outgoing" and t.get("predicate") in TIME_SPAN_PREDICATES:
                    dates = _resolve_time_span(other_uri)
                    for date_key, date_val in dates.items():
                        date_line = f"  - {date_key}: {date_val}"
                        if date_line not in seen_lines:
                            seen_lines.add(date_line)
                            lines.append(date_line)

            if not lines:
                continue

            section = f"{entity_label}:\n" + "\n".join(lines)

            # Check total character budget
            if total_chars + len(section) + 2 > MAX_TOTAL_CHARS:
                # Try to fit a truncated version
                remaining = MAX_TOTAL_CHARS - total_chars - 2
                if remaining > 100:  # Only add if we can fit something meaningful
                    section = section[:remaining] + "..."
                    entity_sections.append(section)
                break

            entity_sections.append(section)
            total_chars += len(section) + 2  # +2 for the \n\n separator

        if not entity_sections:
            return ""

        result = "\n\n".join(entity_sections)
        logger.info(f"Triples enrichment: {len(entity_sections)} entities, "
                    f"{sum(s.count(chr(10)) + 1 for s in entity_sections)} lines, "
                    f"{len(result)} chars")
        return result

    def _build_graph_context(self, selected_docs):
        """Walk graph edges from selected docs and build structured context about neighboring entities.

        Returns a text block describing relationships to entities NOT in the selected set,
        giving the LLM visibility into the broader knowledge graph without including full documents.
        """
        selected_uris = {doc.id for doc in selected_docs}
        selected_labels = {doc.id: doc.metadata.get('label', doc.id.split('/')[-1]) for doc in selected_docs}

        # Collect neighbor info: {neighbor_uri: {label, type, connections, max_weight}}
        neighbor_info = {}

        for doc in selected_docs:
            doc_label = selected_labels[doc.id]
            graph_doc = self.document_store.docs.get(doc.id)
            if not graph_doc:
                continue

            sorted_neighbors = sorted(graph_doc.neighbors, key=lambda n: n.get('weight', 0), reverse=True)

            for neighbor in sorted_neighbors:
                nb_id = neighbor.get('doc_id', '')
                if nb_id in selected_uris or not nb_id:
                    continue

                edge_type = neighbor.get('edge_type', 'related_to')
                weight = neighbor.get('weight', 0.5)

                nb_doc = self.document_store.docs.get(nb_id)
                if nb_doc:
                    nb_label = nb_doc.metadata.get('label', nb_id.split('/')[-1])
                    nb_type = nb_doc.metadata.get('type', '')
                else:
                    nb_label = self._get_entity_label_from_triples(nb_id)
                    nb_type = ''

                if nb_id not in neighbor_info:
                    neighbor_info[nb_id] = {
                        'label': nb_label,
                        'type': nb_type,
                        'connections': [],
                        'max_weight': 0
                    }
                neighbor_info[nb_id]['connections'].append((doc_label, edge_type))
                neighbor_info[nb_id]['max_weight'] = max(neighbor_info[nb_id]['max_weight'], weight)

        if not neighbor_info:
            return ""

        # Sort: most connections first (bridge entities), then by max weight
        sorted_neighbors = sorted(
            neighbor_info.items(),
            key=lambda x: (len(x[1]['connections']), x[1]['max_weight']),
            reverse=True
        )

        lines = []
        max_neighbors = RetrievalConfig.GRAPH_CONTEXT_MAX_NEIGHBORS
        max_lines = RetrievalConfig.GRAPH_CONTEXT_MAX_LINES

        for nb_uri, info in sorted_neighbors:
            if len(lines) >= max_lines:
                break

            type_suffix = f" [{info['type']}]" if info['type'] else ""

            # Show how this neighbor connects to selected entities
            for from_label, edge_type in info['connections'][:max_neighbors]:
                lines.append(f"- {from_label} -> {edge_type} -> {info['label']}{type_suffix}")
                if len(lines) >= max_lines:
                    break

            if len(lines) >= max_lines:
                break

            # 2nd hop: show this neighbor's own key relationships (from graph)
            nb_graph_doc = self.document_store.docs.get(nb_uri)
            if nb_graph_doc:
                hop2_neighbors = sorted(nb_graph_doc.neighbors, key=lambda n: n.get('weight', 0), reverse=True)
                for hop2 in hop2_neighbors[:3]:
                    hop2_id = hop2.get('doc_id', '')
                    if not hop2_id or hop2_id == nb_uri:
                        continue
                    hop2_doc = self.document_store.docs.get(hop2_id)
                    if hop2_doc:
                        hop2_label = hop2_doc.metadata.get('label', '')
                        if hop2_label:
                            lines.append(f"  -> {info['label']} -> {hop2['edge_type']} -> {hop2_label}")
                            if len(lines) >= max_lines:
                                break

        if not lines:
            return ""

        logger.info(f"Graph context: {len(lines)} lines from {len(neighbor_info)} neighbor entities")
        return "\n".join(lines[:max_lines])

    def answer_question(self, question, include_wikidata=True, chat_history=None):
        """Answer a question using the universal RAG system with CIDOC-CRM knowledge and optional Wikidata context.

        Args:
            question: The user's question
            include_wikidata: Whether to fetch Wikidata context
            chat_history: Optional list of {"role": "user"|"assistant", "content": str} dicts
                         for conversation context (most recent last)
        """
        # Validate input
        if not question or not question.strip():
            return {
                "answer": "Please provide a question.",
                "sources": []
            }

        logger.info(f"Answering question directly: '{question}'")

        # LLM-based query analysis for dynamic k and FC-aware boosting
        query_analysis = self._analyze_query(question)

        # Dynamic k based on query type
        k_map = {
            "SPECIFIC": RetrievalConfig.SPECIFIC_K,
            "ENUMERATION": RetrievalConfig.ENUMERATION_K,
            "AGGREGATION": RetrievalConfig.AGGREGATION_K,
        }
        k = k_map.get(query_analysis.query_type, RetrievalConfig.DEFAULT_RETRIEVAL_K)
        initial_pool_size = k * RetrievalConfig.POOL_MULTIPLIER
        logger.info(f"Dynamic retrieval: type={query_analysis.query_type}, k={k}, pool={initial_pool_size}")

        if chat_history:
            # Dual retrieval: run contextualized query AND raw question separately,
            # then merge.  The contextualized query uses only the immediately
            # previous user message (not the current one, which is already the raw
            # query).  This avoids two bugs:
            #   1. Current question was duplicated (already in chat_history + appended)
            #   2. Multiple previous topics polluted the embedding
            prev_user_msgs = [
                m["content"] for m in chat_history[:-1] if m["role"] == "user"
            ]
            if prev_user_msgs:
                contextualized_query = f"{prev_user_msgs[-1]} {question}"
            else:
                contextualized_query = question
            logger.info(f"Dual retrieval — contextualized: '{contextualized_query[:200]}'")
            logger.info(f"Dual retrieval — raw: '{question}'")

            ctx_docs = self.retrieve(contextualized_query, k=k, initial_pool_size=initial_pool_size, query_analysis=query_analysis)

            # Detect vague follow-ups: short questions dominated by pronouns/stopwords
            # For these, the raw FAISS query retrieves irrelevant results, so skip it
            q_words = set(re.findall(r'\b[a-z]+\b', question.lower()))
            stopwords = {'it', 'they', 'them', 'this', 'that', 'these', 'those',
                         'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have',
                         'had', 'do', 'did', 'does', 'when', 'where', 'what', 'how',
                         'and', 'or', 'with', 'which', 'other', 'more', 'its', 'of',
                         'in', 'to', 'for', 'about', 'me', 'tell', 'happened', 'took',
                         'place', 'there', 'any'}
            content_words = q_words - stopwords
            is_vague = len(content_words) == 0 and len(question.split()) <= 10
            if is_vague:
                logger.info(f"Vague follow-up detected, skipping raw retrieval: '{question}'")

            if is_vague:
                retrieved_docs = ctx_docs[:k]
            else:
                raw_docs = self.retrieve(question, k=k, initial_pool_size=initial_pool_size, query_analysis=query_analysis)

                # Interleaved merge: alternate ctx and raw results so raw query
                # gets fair representation instead of being pushed to the tail
                seen_uris = set()
                retrieved_docs = []
                for i in range(max(len(ctx_docs), len(raw_docs))):
                    if i < len(ctx_docs) and ctx_docs[i].id not in seen_uris:
                        seen_uris.add(ctx_docs[i].id)
                        retrieved_docs.append(ctx_docs[i])
                    if i < len(raw_docs) and raw_docs[i].id not in seen_uris:
                        seen_uris.add(raw_docs[i].id)
                        retrieved_docs.append(raw_docs[i])
                retrieved_docs = retrieved_docs[:k]
                logger.info(f"Dual retrieval merged: {len(retrieved_docs)} unique docs "
                            f"(ctx={len(ctx_docs)}, raw={len(raw_docs)})")
        else:
            retrieved_docs = self.retrieve(question, k=k, initial_pool_size=initial_pool_size, query_analysis=query_analysis)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }

        # Create context from retrieved documents with better references
        context = ""

        # Track Wikidata IDs for retrieved entities
        wikidata_context = ""
        entities_with_wikidata = []

        # Truncate each doc if needed to prevent rate limit errors
        # Roughly 4 chars per token, limit ~500 tokens per doc
        MAX_DOC_CHARS = 2000

        for i, doc in enumerate(retrieved_docs):
            entity_uri = doc.id
            entity_label = doc.metadata.get("label", entity_uri.split('/')[-1])

            # Build context without technical type information
            context += f"Entity: {entity_label}\n"
            doc_text = doc.text
            if len(doc_text) > MAX_DOC_CHARS:
                doc_text = doc_text[:MAX_DOC_CHARS] + "...[truncated]"
            context += doc_text + "\n"

            context += "\n"

            # Get Wikidata info if available and requested
            if include_wikidata:
                wikidata_id = self.get_wikidata_for_entity(entity_uri)
                if wikidata_id:
                    entities_with_wikidata.append({
                        "entity_uri": entity_uri,
                        "entity_label": entity_label,
                        "wikidata_id": wikidata_id
                    })

        # Add graph context: relationships to entities beyond the selected documents
        graph_context = self._build_graph_context(retrieved_docs)
        if graph_context:
            context += "\n## Graph Context (relationships beyond retrieved documents)\n"
            context += graph_context + "\n"

        # Add structured triples enrichment from edges.parquet
        triples_enrichment = self._build_triples_enrichment(retrieved_docs)
        if triples_enrichment:
            context += "\n## Structured Relationships\n\n"
            context += triples_enrichment + "\n"

        # If requested, fetch Wikidata information for top 2 most relevant entities
        if include_wikidata and entities_with_wikidata:
            wikidata_context += "\nWikidata Context:\n"
            for entity_info in entities_with_wikidata[:2]:  # Limit to top 2 entities
                wikidata_data = self.fetch_wikidata_info(entity_info["wikidata_id"])
                if wikidata_data:
                    wikidata_context += f"\nWikidata information for {entity_info['entity_label']} ({entity_info['wikidata_id']}):\n"

                    if "label" in wikidata_data:
                        wikidata_context += f"- Label: {wikidata_data['label']}\n"

                    if "description" in wikidata_data:
                        wikidata_context += f"- Description: {wikidata_data['description']}\n"

                    if "properties" in wikidata_data:
                        for prop_name, prop_value in wikidata_data["properties"].items():
                            if isinstance(prop_value, dict) and "latitude" in prop_value:
                                wikidata_context += f"- {prop_name.replace('_', ' ').title()}: Latitude {prop_value['latitude']}, Longitude {prop_value['longitude']}\n"
                            elif isinstance(prop_value, list):
                                wikidata_context += f"- {prop_name.replace('_', ' ').title()}: {', '.join(str(v) for v in prop_value)}\n"
                            else:
                                wikidata_context += f"- {prop_name.replace('_', ' ').title()}: {prop_value}\n"

                    if "wikipedia" in wikidata_data:
                        wikidata_context += f"- Wikipedia: {wikidata_data['wikipedia']['title']}\n"

        # Get CIDOC-CRM system prompt
        system_prompt = self.get_cidoc_system_prompt()

        # Query-type-aware prompt tuning
        if query_analysis.query_type == "ENUMERATION":
            system_prompt += "\nList ALL matching entities from the retrieved information. Be comprehensive."
        elif query_analysis.query_type == "AGGREGATION":
            system_prompt += "\nCount or rank entities based on the retrieved information. State if the count may be incomplete."

        # Add Wikidata instructions to system prompt
        if include_wikidata and wikidata_context:
            system_prompt += "\n\nI have also provided Wikidata information for some entities. When appropriate, incorporate this Wikidata information to enhance your answer with additional context, especially for factual details not present in the RDF data."

        # Create enhanced prompt
        prompt = ""

        # Include conversation history for follow-up context
        if chat_history:
            prompt += "Conversation so far:\n"
            for msg in chat_history[-6:]:  # Last 3 exchanges
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "\n"

        prompt += f"""Retrieved information:
{context}
"""

        # Add Wikidata context if available
        if include_wikidata and wikidata_context:
            prompt += f"{wikidata_context}\n"

        prompt += f"""
Question: {question}

Answer directly using only the retrieved information above. Use entity names, not codes.
"""

        # Generate answer using the provider
        answer = self.llm_provider.generate(system_prompt, prompt)

        # Prepare sources
        sources = []
        entities_with_local_images = set()  # Track entities that have local images

        for i, doc in enumerate(retrieved_docs):
            entity_uri = doc.id
            entity_label = doc.metadata.get("label", entity_uri.split('/')[-1])
            raw_triples = getattr(self, '_triples_index', {}).get(entity_uri, [])
            local_images = doc.metadata.get("images", [])

            source_entry = {
                "id": i,
                "entity_uri": entity_uri,
                "entity_label": entity_label,
                "type": "graph",
                "entity_type": doc.metadata.get("type", "unknown"),
                "raw_triples": raw_triples
            }

            # Add local images if present (priority over Wikidata)
            if local_images:
                entities_with_local_images.add(entity_uri)
                source_entry["images"] = [
                    {
                        "url": img_url,
                        "source": "dataset"
                    }
                    for img_url in local_images
                ]
                logger.info(f"Using {len(local_images)} local image(s) for {entity_label}")

            sources.append(source_entry)

        # Enrich existing graph sources with Wikidata info (images, IDs)
        # Build URI→source index for fast lookup
        source_by_uri = {s["entity_uri"]: s for s in sources}

        for entity_info in entities_with_wikidata:
            entity_uri = entity_info["entity_uri"]
            existing = source_by_uri.get(entity_uri)
            if not existing:
                continue

            existing["wikidata_id"] = entity_info["wikidata_id"]
            existing["wikidata_url"] = f"https://www.wikidata.org/wiki/{entity_info['wikidata_id']}"

            has_local_images = entity_uri in entities_with_local_images

            # Only fetch Wikidata image if no local images exist
            if has_local_images:
                logger.debug(f"Skipping Wikidata image for {entity_info['entity_label']} - local images available")
            else:
                # Fetch Wikidata info to get image (P18)
                wikidata_data = self.fetch_wikidata_info(entity_info["wikidata_id"])
                if wikidata_data and "properties" in wikidata_data:
                    image_value = wikidata_data["properties"].get("image")
                    if image_value:
                        try:
                            # Handle both single image and array of images
                            if isinstance(image_value, list):
                                image_filename = image_value[0] if image_value else None
                            else:
                                image_filename = image_value

                            if image_filename:
                                if isinstance(image_filename, bytes):
                                    image_filename_str = image_filename.decode('utf-8')
                                else:
                                    image_filename_str = str(image_filename)

                                encoded_filename = quote(image_filename_str, safe='')

                                existing["image"] = {
                                    "url": f"https://commons.wikimedia.org/wiki/File:{encoded_filename}",
                                    "thumbnail_url": f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded_filename}?width=300",
                                    "full_url": f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded_filename}",
                                    "filename": image_filename_str,
                                    "source": "wikidata_p18"
                                }
                                logger.info(f"Found Wikidata image for {entity_info['entity_label']}: {image_filename_str}")
                        except Exception as e:
                            logger.warning(f"Error processing image filename for {entity_info['entity_label']}: {e}")

        return {
            "answer": answer,
            "sources": sources
        }