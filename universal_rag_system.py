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
import pickle
import re
import shutil
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
import numpy as np
import networkx as nx
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON

# Langchain imports
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# Third-party data fetching
import requests

# Local imports
from graph_document_store import GraphDocumentStore
from llm_providers import get_llm_provider, get_embedding_provider, BaseLLMProvider
from embedding_cache import EmbeddingCache
from scripts.extract_ontology_labels import run_extraction

logger = logging.getLogger(__name__)


class RetrievalConfig:
    """Configuration constants for the RAG retrieval system"""

    # Score combination weights
    VECTOR_PAGERANK_ALPHA = 0.6  # Weight for combining vector similarity and PageRank scores
    RELEVANCE_CONNECTIVITY_ALPHA = 0.7  # Weight for combining relevance and connectivity scores

    # PageRank parameters
    PAGERANK_DAMPING = 0.85  # Damping factor for PageRank algorithm
    PAGERANK_ITERATIONS = 20  # Number of iterations for PageRank computation

    # Rate limiting
    TOKENS_PER_MINUTE_LIMIT = 950_000  # Token limit for rate limiting (TPM)

    # Retrieval parameters
    DEFAULT_RETRIEVAL_K = 10  # Default number of documents to retrieve
    INITIAL_POOL_MULTIPLIER = 2  # Multiplier for initial candidate pool size

    # Processing parameters
    DEFAULT_BATCH_SIZE = 50  # Default batch size for processing entities
    ENTITY_CONTEXT_DEPTH = 2  # Depth for entity context traversal
    MAX_ADJACENCY_HOPS = 2  # Maximum hops for adjacency matrix construction

    # Event classes are loaded from config/event_classes.json at runtime
    # This allows users to modify event classes without changing code.
    # See UniversalRagSystem._load_event_classes() for loading logic.


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

    def __init__(self, endpoint_url, config=None, dataset_id=None):
        """
        Initialize the universal RAG system.

        Args:
            endpoint_url: SPARQL endpoint URL
            config: Configuration dictionary for LLM provider
            dataset_id: Optional dataset identifier for multi-dataset support.
                        Used to create dataset-specific cache directories.
        """
        self.endpoint_url = endpoint_url
        self.dataset_id = dataset_id or "default"
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)

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

    # ==================== Path Helper Methods ====================
    # These methods return dataset-specific paths for multi-dataset support

    def _get_cache_dir(self) -> str:
        """Return the cache directory for this dataset."""
        return f'data/cache/{self.dataset_id}'

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

    def _get_documents_dir(self) -> str:
        """Return the entity documents directory for this dataset."""
        return f'data/documents/{self.dataset_id}/entity_documents'

    # ==================== End Path Helper Methods ====================

    def _load_inverse_properties(self):
        """
        Load inverse property mappings from JSON file generated from ontologies.

        Returns:
            dict: Inverse property mappings (property URI -> inverse URI)
        """
        inverse_file = 'data/labels/inverse_properties.json'

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

    def _load_property_labels(self, force_extract=False):
        """
        Load property labels from JSON file generated from ontologies.
        Automatically extracts labels from ontology files if JSON doesn't exist.

        Args:
            force_extract: If True, force re-extraction even if JSON exists

        Returns:
            dict: Property labels mapping
        """
        labels_file = 'data/labels/property_labels.json'
        ontology_dir = 'data/ontologies'

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
        classes_file = 'data/labels/ontology_classes.json'
        ontology_dir = 'data/ontologies'

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
                success = run_extraction(ontology_dir, 'data/labels/property_labels.json', classes_file)
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
        labels_file = 'data/labels/class_labels.json'
        ontology_dir = 'data/ontologies'

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
                success = run_extraction(ontology_dir, 'data/labels/property_labels.json', 'data/labels/ontology_classes.json', labels_file)
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
        config_file = os.path.join(os.path.dirname(__file__), 'config', 'event_classes.json')

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

        # Check for special modes that don't need full initialization
        if self.config.get('generate_docs_only'):
            return self.generate_documents_only()

        if self.config.get('embed_from_docs'):
            return self.embed_from_documents()

        # Test connection (not needed for embed_from_docs mode)
        if not self.test_connection():
            logger.error("Failed to connect to SPARQL endpoint")
            return False

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
                return True
            else:
                logger.warning("Failed to load saved data completely, rebuilding...")
        else:
            logger.info("No saved data found, building from scratch...")
        
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
        return True

    def generate_documents_only(self):
        """
        Generate entity documents from SPARQL without computing embeddings.
        Use this locally when you have SPARQL access but want to compute
        embeddings on a remote GPU cluster.

        Documents are saved to: data/documents/<dataset_id>/entity_documents/
        Metadata is saved to: data/documents/<dataset_id>/documents_metadata.json

        Transfer these files to the cluster, then run with --embed-from-docs.
        """
        logger.info("=" * 60)
        logger.info("GENERATE DOCUMENTS ONLY MODE")
        logger.info("=" * 60)
        logger.info("Will create documents from SPARQL without computing embeddings.")
        logger.info("Transfer documents to cluster, then run with --embed-from-docs")
        logger.info("=" * 60)

        # Test SPARQL connection
        if not self.test_connection():
            logger.error("Failed to connect to SPARQL endpoint")
            return False

        # Get all entities
        entities = self.get_all_entities()
        total_entities = len(entities)
        logger.info(f"Found {total_entities} entities")

        # Prepare output directory
        output_dir = self._get_documents_dir()
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving documents to: {output_dir}")

        # Track document metadata for later embedding
        documents_metadata = []

        # Process entities and create documents
        for entity in tqdm(entities, desc="Generating documents", unit="entity"):
            entity_uri = entity["entity"]

            try:
                doc_text, entity_label, entity_types, raw_triples = self.create_enhanced_document(entity_uri)

                # Save document to disk
                filepath = self.save_entity_document(entity_uri, doc_text, entity_label)

                # Determine primary type
                primary_type = "Unknown"
                if entity_types:
                    human_readable_types = [
                        t for t in entity_types
                        if not self.is_technical_class_name(t)
                    ]
                    primary_type = human_readable_types[0] if human_readable_types else "Entity"

                # Store metadata for embedding phase
                documents_metadata.append({
                    "uri": entity_uri,
                    "label": entity_label,
                    "type": primary_type,
                    "all_types": entity_types,
                    "filepath": os.path.basename(filepath) if filepath else None
                })

            except Exception as e:
                logger.error(f"Error processing entity {entity_uri}: {str(e)}")
                continue

        # Save metadata file
        metadata_path = os.path.join(os.path.dirname(output_dir), "documents_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset_id": self.dataset_id,
                "total_documents": len(documents_metadata),
                "generated_at": datetime.now().isoformat(),
                "documents": documents_metadata
            }, f, indent=2)

        logger.info("=" * 60)
        logger.info("DOCUMENT GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Generated {len(documents_metadata)} documents")
        logger.info(f"Documents saved to: {output_dir}")
        logger.info(f"Metadata saved to: {metadata_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Transfer to cluster:")
        logger.info(f"     scp -r data/documents/{self.dataset_id}/ user@cluster:CRM_RAG/data/documents/{self.dataset_id}/")
        logger.info(f"  2. On cluster, run:")
        logger.info(f"     python main.py --env .env.cluster --dataset {self.dataset_id} --embed-from-docs --process-only")
        logger.info("=" * 60)

        return True

    def embed_from_documents(self):
        """
        Generate embeddings from existing document files (no SPARQL needed).
        Use this on a GPU cluster after transferring documents from local machine.

        Reads documents from: data/documents/<dataset_id>/entity_documents/
        Reads metadata from: data/documents/<dataset_id>/documents_metadata.json

        Creates: data/cache/<dataset_id>/document_graph.pkl
                 data/cache/<dataset_id>/vector_index/
        """
        logger.info("=" * 60)
        logger.info("EMBED FROM DOCUMENTS MODE")
        logger.info("=" * 60)
        logger.info("Will generate embeddings from existing document files.")
        logger.info("No SPARQL connection needed.")
        logger.info("=" * 60)

        # Check for documents directory
        docs_dir = self._get_documents_dir()
        metadata_path = os.path.join(os.path.dirname(docs_dir), "documents_metadata.json")

        if not os.path.exists(metadata_path):
            logger.error(f"Metadata file not found: {metadata_path}")
            logger.error("Run --generate-docs-only first, then transfer documents to this machine.")
            return False

        if not os.path.exists(docs_dir):
            logger.error(f"Documents directory not found: {docs_dir}")
            return False

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        documents_metadata = metadata.get("documents", [])
        logger.info(f"Found metadata for {len(documents_metadata)} documents")

        # Initialize document store
        self.document_store = GraphDocumentStore(self.embeddings)

        # Read documents and prepare for batch embedding
        batch_docs = []  # List of (uri, text, metadata, cached_embedding)

        for doc_meta in tqdm(documents_metadata, desc="Reading documents", unit="doc"):
            filepath = doc_meta.get("filepath")
            if not filepath:
                continue

            full_path = os.path.join(docs_dir, filepath)
            if not os.path.exists(full_path):
                logger.warning(f"Document file not found: {full_path}")
                continue

            # Read document content
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Skip YAML frontmatter to get actual document text
                if content.startswith('---'):
                    parts = content.split('---', 2)
                    if len(parts) >= 3:
                        doc_text = parts[2].strip()
                    else:
                        doc_text = content
                else:
                    doc_text = content

                # Check embedding cache
                cached_embedding = None
                if self.embedding_cache:
                    cached_embedding = self.embedding_cache.get(doc_meta["uri"])

                batch_docs.append((
                    doc_meta["uri"],
                    doc_text,
                    {
                        "uri": doc_meta["uri"],
                        "label": doc_meta["label"],
                        "type": doc_meta["type"],
                        "all_types": doc_meta.get("all_types", []),
                        "wikidata_id": doc_meta.get("wikidata_id")
                    },
                    cached_embedding
                ))

            except Exception as e:
                logger.error(f"Error reading document {filepath}: {str(e)}")
                continue

        logger.info(f"Loaded {len(batch_docs)} documents")

        # Process embeddings in batches
        batch_size = int(self.config.get("embedding_batch_size", 64))

        for i in range(0, len(batch_docs), batch_size):
            batch = batch_docs[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(batch_docs) + batch_size - 1) // batch_size
            logger.info(f"Processing embedding batch {batch_num}/{total_batches}")

            if self.use_batch_embedding:
                self._process_batch_embeddings(batch)
            else:
                self._process_sequential_embeddings(batch, 0, time.time(), float('inf'))

        # Build edges from document relationships
        # Note: Without SPARQL, we can't query relationships directly
        # Edges will be built from the raw_triples stored in documents if available
        logger.info("Building document graph edges from stored relationships...")
        self._build_edges_from_documents(documents_metadata, docs_dir)

        # Build vector store
        logger.info("Building vector store...")
        self.document_store.rebuild_vector_store()

        # Save document graph
        doc_graph_path = self._get_document_graph_path()
        vector_index_dir = self._get_vector_index_dir()

        os.makedirs(os.path.dirname(doc_graph_path), exist_ok=True)
        self.document_store.save_document_graph(doc_graph_path)

        os.makedirs(vector_index_dir, exist_ok=True)
        if self.document_store.vector_store:
            self.document_store.vector_store.save_local(vector_index_dir)
            logger.info(f"Vector store saved to {vector_index_dir}")

        logger.info("=" * 60)
        logger.info("EMBEDDING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed {len(self.document_store.docs)} documents")
        logger.info(f"Document graph saved to: {doc_graph_path}")
        logger.info(f"Vector index saved to: {vector_index_dir}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Transfer cache to local machine:")
        logger.info(f"     scp -r data/cache/{self.dataset_id}/ user@local:CRM_RAG/data/cache/{self.dataset_id}/")
        logger.info(f"  2. On local machine, run:")
        logger.info(f"     python main.py --env .env.local")
        logger.info("=" * 60)

        return True

    def _build_edges_from_documents(self, documents_metadata, docs_dir):
        """
        Build edges between documents by extracting relationships from document content.
        This is used when SPARQL is not available (embed-from-docs mode).
        """
        # Create a mapping of URIs to doc_ids for quick lookup
        uri_to_doc = {meta["uri"]: meta["uri"] for meta in documents_metadata}

        # For each document, try to find relationships in the content
        edges_added = 0
        for doc_meta in documents_metadata:
            uri = doc_meta["uri"]
            if uri not in self.document_store.docs:
                continue

            # Check if other document URIs are mentioned in this document's text
            doc = self.document_store.docs[uri]
            doc_text = doc.text.lower()

            for other_meta in documents_metadata:
                other_uri = other_meta["uri"]
                if other_uri == uri:
                    continue
                if other_uri not in self.document_store.docs:
                    continue

                # Check if other entity's label appears in this document
                other_label = other_meta.get("label", "").lower()
                if other_label and len(other_label) > 3 and other_label in doc_text:
                    # Add edge with default weight
                    self.document_store.add_edge(uri, other_uri, "mentioned", weight=0.5)
                    edges_added += 1

        logger.info(f"Added {edges_added} edges based on document content relationships")

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
        # Schema-level predicates to exclude
        schema_patterns = [
            'rdf-syntax-ns#type',
            'rdf-schema#subClassOf',
            'rdf-schema#domain',
            'rdf-schema#range',
            'rdf-schema#Class',
            'rdf-schema#subPropertyOf',
            'rdf-schema#label',
            'rdf-schema#comment',
            'owl#',
            '/type',  # Catch various type predicates
            '/subClassOf',
            '/domain',
            '/range'
        ]

        # Check if predicate contains any schema pattern
        for pattern in schema_patterns:
            if pattern in predicate:
                return True

        return False

    def is_technical_class_name(self, class_name):
        """
        Check if a class name is a technical ontology class that should be filtered
        from natural language output.

        This method uses the ontology classes extracted from CIDOC-CRM, VIR, CRMdig, etc.
        ontology files rather than hard-coded regex patterns.

        Examples of technical classes that will be filtered:
        - E22_Human-Made_Object
        - E53_Place
        - D1_Digital_Object (CRMdig)
        - IC9_Representation (VIR)
        - FXX_Name (FRBRoo)

        Args:
            class_name: The class name to check (can be full URI or local name)

        Returns:
            bool: True if it's a technical ontology class, False if it's human-readable
        """
        # Use ontology classes loaded from ontology files
        if UniversalRagSystem._ontology_classes is None:
            # Fallback to regex pattern if classes haven't been loaded
            logger.warning("Ontology classes not loaded, falling back to regex pattern matching")
            technical_pattern = r'^[A-Z]+\d+[a-z]?_'
            return bool(re.match(technical_pattern, class_name))

        # Check if the class name (or its local name) is in the ontology classes
        # First check direct match
        if class_name in UniversalRagSystem._ontology_classes:
            return True

        # Also check if the local name (after last / or #) is a technical class
        if '/' in class_name or '#' in class_name:
            local_name = class_name.split('/')[-1].split('#')[-1]
            if local_name in UniversalRagSystem._ontology_classes:
                return True

        return False

    def get_entity_types_cached(self, entity_uri, cache=None):
        """
        Get entity types with optional caching to avoid repeated SPARQL queries.

        Args:
            entity_uri: The URI of the entity
            cache: Optional dict to cache results (entity_uri -> set of type URIs)

        Returns:
            set: Set of type URIs for this entity
        """
        if cache is not None and entity_uri in cache:
            return cache[entity_uri]

        type_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?type WHERE {{
            <{entity_uri}> rdf:type ?type .
            FILTER(STRSTARTS(STR(?type), "http://"))
        }}
        """

        types = set()
        try:
            self.sparql.setQuery(type_query)
            results = self.sparql.query().convert()
            for result in results["results"]["bindings"]:
                types.add(result["type"]["value"])
        except Exception as e:
            logger.warning(f"Error getting types for {entity_uri}: {str(e)}")

        if cache is not None:
            cache[entity_uri] = types

        return types

    def get_entity_label(self, entity_uri):
        """
        Get a human-readable label for an entity.
        Tries multiple strategies to find a good label instead of falling back to UUIDs.
        """
        # Try to get any literal that could serve as a label
        try:
            literals = self.get_entity_literals(entity_uri)
            if literals:
                # Try common label properties in order of preference
                for label_prop in ['label', 'prefLabel', 'name', 'title', 'skos:prefLabel']:
                    if label_prop in literals and literals[label_prop]:
                        return literals[label_prop][0]

                # If no standard label found, try to find any literal that looks like a label
                # Prefer shorter strings that don't look like UUIDs or descriptions
                for prop, values in literals.items():
                    if values and len(values) > 0:
                        first_value = str(values[0])
                        # Skip if it looks like a UUID or is very long (likely a description)
                        if len(first_value) < 100 and not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-', first_value):
                            return first_value
        except Exception as e:
            logger.debug(f"Error getting literals for {entity_uri}: {str(e)}")

        # Last resort: try to extract a meaningful part from the URI
        # Avoid UUIDs and prefer human-readable segments
        uri_parts = entity_uri.rstrip('/').split('/')
        for part in reversed(uri_parts):
            # Skip empty parts, UUIDs, and generic terms
            if part and not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-', part) and part not in ['semantics', 'icon', 'appellation', 'data']:
                # Clean up the part (remove underscores, etc.)
                cleaned = part.replace('_', ' ').replace('-', ' ')
                if cleaned and len(cleaned) < 50:
                    return cleaned

        # Absolute fallback: return the last part of URI
        return entity_uri.rstrip('/').split('/')[-1] if entity_uri else "Unknown"

    def get_entity_context(self, entity_uri, depth=RetrievalConfig.ENTITY_CONTEXT_DEPTH, return_triples=False):
        """Get entity context by traversing the graph bidirectionally with event-aware logic.

        In CIDOC-CRM, events are the "glue" connecting things, actors, places, and times.
        This method uses event-aware traversal:
        - For EVENT entities: full multi-hop traversal (events connect everything)
        - For NON-EVENT entities: only traverse deeper through events

        This prevents the "global pollution" problem where unrelated entities get
        included in documents just because they share a distant connection.

        Args:
            entity_uri: The URI of the entity to get context for
            depth: How many hops to traverse
            return_triples: If True, also return raw RDF triples

        Returns:
            If return_triples is False: list of natural language statements
            If return_triples is True: tuple of (statements list, triples list)
        """

        context_statements = []
        raw_triples = []
        visited = set()

        # Cache for entity types to avoid repeated SPARQL queries
        type_cache = {}

        # Load event classes once for this traversal
        event_classes = UniversalRagSystem.get_event_classes()

        # Debug flag - set to True to trace filtering decisions
        DEBUG_TRAVERSAL = logger.isEnabledFor(logging.DEBUG)

        def is_event(uri):
            """Check if URI is an event class instance (using cache)"""
            types = self.get_entity_types_cached(uri, type_cache)
            is_evt = bool(types & event_classes)
            if DEBUG_TRAVERSAL:
                logger.debug(f"  is_event({uri.split('/')[-1]}): types={[t.split('/')[-1].split('#')[-1] for t in types]}, is_event={is_evt}")
            return is_evt

        # Check if the starting entity is an event
        start_is_event = is_event(entity_uri)
        if DEBUG_TRAVERSAL:
            logger.debug(f"Starting entity {entity_uri.split('/')[-1]}: is_event={start_is_event}")

        def traverse(uri, current_depth=0, direction="both", came_from_event=False):
            if uri in visited or current_depth > depth:
                return

            visited.add(uri)

            # Get entity label using the improved label retrieval method
            entity_label = self.get_entity_label(uri)

            if DEBUG_TRAVERSAL:
                logger.debug(f"==> Traversing {entity_label} (depth={current_depth}, uri={uri.split('/')[-1]})")

            # Get outgoing relationships if direction is "both" or "outgoing"
            if direction in ["both", "outgoing"]:
                outgoing_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?pred ?obj ?objLabel WHERE {{
                    <{uri}> ?pred ?obj .
                    OPTIONAL {{ ?obj rdfs:label ?objLabel }}
                    FILTER(isURI(?obj))
                }}
                """

                try:
                    self.sparql.setQuery(outgoing_query)
                    outgoing_results = self.sparql.query().convert()

                    for result in outgoing_results["results"]["bindings"]:
                        pred = result["pred"]["value"]
                        obj = result["obj"]["value"]

                        # Filter out schema-level predicates
                        if self.is_schema_predicate(pred):
                            continue

                        # Get labels if available, with improved fallback
                        # Use property_labels.json for English predicate labels
                        pred_label = UniversalRagSystem._property_labels.get(pred)
                        if not pred_label:
                            # Fallback to local name if not in property_labels
                            pred_label = pred.split('/')[-1].split('#')[-1]

                        obj_label = result.get("objLabel", {}).get("value")
                        if not obj_label:
                            obj_label = self.get_entity_label(obj)

                        # Filter out self-referential relationships
                        # Skip if same URI or same label (redundant statements)
                        if uri == obj or (entity_label and obj_label and entity_label.lower() == obj_label.lower()):
                            continue

                        # Event-aware filtering:
                        # For non-event starting entities at depth > 0, only include:
                        # - Paths TO events (target is event)
                        # - Connections FROM events to non-events (completing the chain)
                        # This prevents pollution from unrelated event connections.
                        current_is_event = is_event(uri)
                        target_is_event = is_event(obj)

                        if start_is_event:
                            # Starting entity is an event - include all connections
                            should_include = True
                            reason = "start_is_event"
                        elif current_depth == 0:
                            # Always include direct connections from starting entity
                            should_include = True
                            reason = "depth_0"
                        elif current_is_event:
                            # At depth > 0: only include if traversing FROM an event
                            # Events are the "glue" - we traverse THROUGH them, not TO them
                            # A non-event's production/creation events are about that entity,
                            # not relevant to the starting entity's document
                            should_include = True
                            reason = "from_event"
                        else:
                            # Non-event at depth > 0: don't pull in connected events
                            should_include = False
                            reason = "non_event_skip"

                        if DEBUG_TRAVERSAL:
                            logger.debug(f"  [OUT d={current_depth}] {entity_label} --{pred_label}--> {obj_label}: {reason} -> include={should_include}")

                        if not should_include:
                            continue  # Skip this statement - not relevant to starting entity

                        # Store raw triple if requested
                        if return_triples:
                            raw_triples.append({
                                "subject": uri,
                                "subject_label": entity_label,
                                "predicate": pred,
                                "predicate_label": pred_label,
                                "object": obj,
                                "object_label": obj_label
                            })

                        # Create natural language statement
                        statement = self.process_cidoc_relationship(
                            uri, pred, obj, entity_label, obj_label
                        )

                        context_statements.append(statement)

                        # Event-aware recursive traversal:
                        # Only continue deeper if this connection involves events
                        if current_depth < depth and should_include:
                            traverse(obj, current_depth + 1, "both", came_from_event=current_is_event)
                except Exception as e:
                    logger.error(f"Error traversing outgoing relationships: {str(e)}")

            # Get incoming relationships if direction is "both" or "incoming"
            if direction in ["both", "incoming"]:
                incoming_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?subj ?subjLabel ?pred WHERE {{
                    ?subj ?pred <{uri}> .
                    OPTIONAL {{ ?subj rdfs:label ?subjLabel }}
                    FILTER(isURI(?subj))
                }}
                """

                try:
                    self.sparql.setQuery(incoming_query)
                    incoming_results = self.sparql.query().convert()

                    for result in incoming_results["results"]["bindings"]:
                        subj = result["subj"]["value"]
                        pred = result["pred"]["value"]

                        # Filter out schema-level predicates
                        if self.is_schema_predicate(pred):
                            continue

                        # Get labels if available, with improved fallback
                        subj_label = result.get("subjLabel", {}).get("value")
                        if not subj_label:
                            subj_label = self.get_entity_label(subj)

                        # Use property_labels.json for English predicate labels
                        pred_label = UniversalRagSystem._property_labels.get(pred)
                        if not pred_label:
                            # Fallback to local name if not in property_labels
                            pred_label = pred.split('/')[-1].split('#')[-1]

                        # Filter out self-referential relationships
                        # Skip if same URI or same label (redundant statements)
                        if subj == uri or (subj_label and entity_label and subj_label.lower() == entity_label.lower()):
                            continue

                        # Event-aware filtering (same logic as outgoing)
                        current_is_event = is_event(uri)
                        target_is_event = is_event(subj)

                        if start_is_event:
                            should_include = True
                            reason = "start_is_event"
                        elif current_depth == 0:
                            should_include = True
                            reason = "depth_0"
                        elif current_is_event:
                            # At depth > 0: only include if traversing FROM an event
                            # Events are the "glue" - we traverse THROUGH them, not TO them
                            should_include = True
                            reason = "from_event"
                        else:
                            # Non-event at depth > 0: don't pull in connected events
                            should_include = False
                            reason = "non_event_skip"

                        if DEBUG_TRAVERSAL:
                            logger.debug(f"  [IN d={current_depth}] {subj_label} --{pred_label}--> {entity_label}: {reason} -> include={should_include}")

                        if not should_include:
                            continue  # Skip this statement - not relevant to starting entity

                        # Store raw triple if requested
                        if return_triples:
                            raw_triples.append({
                                "subject": subj,
                                "subject_label": subj_label,
                                "predicate": pred,
                                "predicate_label": pred_label,
                                "object": uri,
                                "object_label": entity_label
                            })

                        # Create natural language statement using INVERSE predicate
                        # For incoming: Production P108 Painting → Painting P108i Production
                        # This expresses the relationship from the current entity's perspective
                        inverse_pred = self._get_inverse_predicate(pred)
                        if DEBUG_TRAVERSAL:
                            logger.debug(f"    Inverse: {pred.split('#')[-1]} -> {inverse_pred.split('#')[-1]}")
                        statement = self.process_cidoc_relationship(
                            uri, inverse_pred, subj, entity_label, subj_label
                        )
                        if DEBUG_TRAVERSAL:
                            logger.debug(f"    Statement: {statement}")

                        context_statements.append(statement)

                        # NOTE: We do NOT traverse from incoming relationships.
                        # Incoming relationships are "about" this entity from another entity's perspective.
                        # Information flows FROM subject TO object, so we only traverse outgoing.
                        # Example: "Atom carries Inscription" is the Atom's property, not the Inscription's.
                        # The Inscription's document should not inherit the Atom's production info.

                except Exception as e:
                    logger.error(f"Error traversing incoming relationships: {str(e)}")

        # Start traversal
        traverse(entity_uri)

        # Return unique statements
        unique_statements = list(set(context_statements))

        if return_triples:
            # Deduplicate triples based on subject-predicate-object URIs
            seen_triples = set()
            unique_triples = []
            for triple in raw_triples:
                triple_key = (triple["subject"], triple["predicate"], triple["object"])
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    unique_triples.append(triple)
            return unique_statements, unique_triples
        else:
            return unique_statements

    def create_enhanced_document(self, entity_uri):
        """Create an enhanced document with natural language interpretation of CIDOC-CRM relationships"""

        try:
            # Get all literal properties for this entity
            literals = self.get_entity_literals(entity_uri)

            # Extract label from literals
            entity_label = entity_uri.split('/')[-1]  # Default to URI fragment
            for label_prop in ['label', 'prefLabel', 'name', 'title']:
                if label_prop in literals and literals[label_prop]:
                    entity_label = literals[label_prop][0]
                    break

            # Get entity types
            type_query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?type ?typeLabel WHERE {{
                <{entity_uri}> rdf:type ?type .
                OPTIONAL {{ ?type rdfs:label ?typeLabel FILTER(LANG(?typeLabel) = "en" || LANG(?typeLabel) = "") }}
                FILTER(STRSTARTS(STR(?type), "http://"))
            }}
            """

            entity_types = []
            try:
                self.sparql.setQuery(type_query)
                type_results = self.sparql.query().convert()

                for result in type_results["results"]["bindings"]:
                    type_uri = result["type"]["value"]

                    # Get English label from class_labels.json
                    type_label = None
                    if UniversalRagSystem._class_labels:
                        type_label = UniversalRagSystem._class_labels.get(type_uri)

                    # If not in ontology files, try to get from triplestore
                    if not type_label:
                        triplestore_label = result.get("typeLabel", {}).get("value")
                        if triplestore_label:
                            type_label = triplestore_label
                            # Track this as a missing class from ontology files
                            if type_uri not in UniversalRagSystem._missing_classes:
                                UniversalRagSystem._missing_classes.add(type_uri)
                                logger.warning(f"Class not found in ontology files: {type_uri}")
                                logger.info(f"  Using label from triplestore: {type_label}")

                    # Final fallback to local name if still no label found
                    if not type_label:
                        type_label = type_uri.split('/')[-1].split('#')[-1]
                        # Track this as a missing class with no label anywhere
                        if type_uri not in UniversalRagSystem._missing_classes:
                            UniversalRagSystem._missing_classes.add(type_uri)
                            logger.warning(f"Class not found in ontology files or triplestore: {type_uri}")
                            logger.info(f"  Using fallback label from URI: {type_label}")

                    entity_types.append(type_label)
            except Exception as e:
                logger.warning(f"Error getting entity types for {entity_uri}: {str(e)}")

            # Get relationships and convert to natural language (also get raw triples)
            try:
                context_statements, raw_triples = self.get_entity_context(
                    entity_uri,
                    depth=RetrievalConfig.ENTITY_CONTEXT_DEPTH,
                    return_triples=True
                )
            except Exception as e:
                logger.warning(f"Error getting entity context for {entity_uri}: {str(e)}")
                context_statements = []
                raw_triples = []

            # Create document text
            text = f"# {entity_label}\n\n"

            # Add entity identifier
            text += f"URI: {entity_uri}\n\n"

            # Add entity types (filter out technical CIDOC-CRM class names)
            if entity_types:
                # Filter to keep only human-readable type labels
                human_readable_types = [
                    t for t in entity_types
                    if not self.is_technical_class_name(t)
                ]

                # Only include Types section if there are human-readable types
                if human_readable_types:
                    text += "## Types\n\n"
                    for type_label in human_readable_types:
                        text += f"- {type_label}\n"
                    text += "\n"

            # Add all literal properties (labels, descriptions, WKT, dates, etc.)
            if literals:
                text += "## Properties\n\n"
                for prop_name, values in sorted(literals.items()):
                    # Format property name for display
                    display_name = prop_name.replace('_', ' ').title()

                    # Handle single vs multiple values
                    if len(values) == 1:
                        # Truncate very long values (like WKT) for readability
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

            # Add natural language descriptions of relationships
            if context_statements:
                text += "## Relationships\n\n"
                for statement in context_statements:
                    text += f"- {statement}\n"

            return text, entity_label, entity_types, raw_triples
        except Exception as e:
            logger.error(f"Error creating enhanced document for {entity_uri}: {str(e)}")
            # Return minimal document to prevent complete failure
            return f"Entity: {entity_uri}", entity_uri, [], []

    def save_entity_document(self, entity_uri, document_text, entity_label, output_dir=None):
        """Save entity document to disk for transparency and reuse"""

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

            # Add metadata header to document
            metadata = f"""---
URI: {entity_uri}
Label: {entity_label}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
---

"""

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
            logger.warning("NOT present in the 'ontology/' directory (CIDOC-CRM/VIR/CRMdig):")

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
            logger.error("\n  STEP 3: Add ontology files to 'ontology/' directory")
            logger.error("    $ cp /path/to/ontology.ttl ontology/")
            logger.error("\n  STEP 4: Extract labels from new ontology files")
            logger.error("    $ python extract_ontology_labels.py")
            logger.error("\n  STEP 5: Rebuild RAG system (delete caches and re-run)")
            logger.error("    $ rm -rf data/cache/document_graph.pkl data/cache/vector_index/ data/documents/entity_documents/")
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
            logger.warning("NOT present in the 'ontology/' directory (CIDOC-CRM/VIR/CRMdig):")

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
            logger.error("\n  STEP 3: Add ontology files to 'ontology/' directory")
            logger.error("    $ cp /path/to/ontology.ttl ontology/")
            logger.error("\n  STEP 4: Extract labels from new ontology files")
            logger.error("    $ python extract_ontology_labels.py")
            logger.error("\n  STEP 5: Rebuild RAG system (delete caches and re-run)")
            logger.error("    $ rm -rf data/cache/document_graph.pkl data/cache/vector_index/ data/documents/entity_documents/")
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
                    f.write("'ontology/' directory. To proceed with optimal use of the RAG system,\n")
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
                    f.write("  • http://www.cidoc-crm.org/cidoc-crm/...  → CIDOC-CRM (already in ontology/)\n")
                    f.write("  • http://w3id.org/vir#...                  → VIR (already in ontology/)\n")
                    f.write("  • http://www.ics.forth.gr/isl/CRMdig/...  → CRMdig (already in ontology/)\n")
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

                    f.write("STEP 3: Add ontology files to the 'ontology/' directory\n")
                    f.write("-" * 80 + "\n")
                    f.write("  $ cp /path/to/downloaded/ontology.ttl ontology/\n")
                    f.write("  $ cp /path/to/custom/ontology.rdf ontology/\n\n")
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
                    f.write("Delete cached data:\n")
                    f.write("  $ rm -f data/cache/document_graph.pkl data/cache/document_graph_temp.pkl\n")
                    f.write("  $ rm -rf data/cache/vector_index/\n")
                    f.write("  $ rm -rf data/documents/entity_documents/\n\n")
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
        readme_content = """# Entity Documents

This directory contains individual markdown files for each entity processed from the RDF data.

## Purpose
- **Transparency**: View exactly what the system extracts and processes for each entity
- **Debugging**: Identify issues with relationship extraction or label processing
- **Reuse**: These documents can be reused for other purposes or analyses

## File Naming Convention
Files are named: `{label}_{hash}.md`
- `label`: Cleaned entity label (special chars removed, spaces replaced with underscores)
- `hash`: 8-character MD5 hash of the entity URI (ensures uniqueness)

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
        
        # First pass: create document nodes with enhanced content
        logger.info("Creating enhanced document nodes...")

        # Determine batch size based on embedding provider
        if self.use_batch_embedding:
            # For local embeddings, use larger batches (no rate limits)
            batch_size = int(self.config.get("embedding_batch_size", 64))
            logger.info(f"Using batch embedding with batch_size={batch_size}")
        else:
            # For API-based embeddings, use smaller batches with rate limiting
            batch_size = RetrievalConfig.DEFAULT_BATCH_SIZE
            logger.info(f"Using sequential embedding with batch_size={batch_size}")

        # Global rate limit tracking (only used for API-based embeddings)
        global_token_count = 0
        tokens_per_min_limit = RetrievalConfig.TOKENS_PER_MINUTE_LIMIT
        last_reset_time = time.time()

        # Check embedding cache for already processed entities
        cached_count = 0
        if self.embedding_cache:
            cache_stats = self.embedding_cache.get_stats()
            logger.info(f"Embedding cache: {cache_stats['count']} cached embeddings ({cache_stats['size_mb']} MB)")

        # Process entities in batches
        for i in range(0, total_entities, batch_size):
            batch = entities[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_entities + batch_size - 1) // batch_size
            logger.info(f"Processing batch {batch_num}/{total_batches}")

            # Collect document data for batch processing
            batch_docs = []  # List of (entity_uri, doc_text, metadata, cached_embedding)

            for entity in tqdm(batch, desc=f"Batch {batch_num}", unit="entity"):
                entity_uri = entity["entity"]

                try:
                    # Create enhanced document with CIDOC-CRM aware natural language
                    doc_text, entity_label, entity_types, raw_triples = self.create_enhanced_document(entity_uri)

                    # Save document to disk for transparency and reuse
                    self.save_entity_document(entity_uri, doc_text, entity_label)

                    # Determine primary entity type
                    primary_type = "Unknown"
                    if entity_types:
                        human_readable_types = [
                            t for t in entity_types
                            if not self.is_technical_class_name(t)
                        ]
                        primary_type = human_readable_types[0] if human_readable_types else "Entity"

                    # Get Wikidata ID if available (cache it in metadata)
                    wikidata_id = self._fetch_wikidata_id_from_sparql(entity_uri)

                    metadata = {
                        "label": entity_label,
                        "type": primary_type,
                        "uri": entity_uri,
                        "all_types": entity_types,
                        "raw_triples": raw_triples,
                        "wikidata_id": wikidata_id  # May be None
                    }

                    # Check embedding cache
                    cached_embedding = None
                    if self.embedding_cache:
                        cached_embedding = self.embedding_cache.get(entity_uri)
                        if cached_embedding:
                            cached_count += 1

                    batch_docs.append((entity_uri, doc_text, metadata, cached_embedding))

                except Exception as e:
                    logger.error(f"Error processing entity {entity_uri}: {str(e)}")
                    continue

            # Process embeddings for the batch
            if self.use_batch_embedding:
                # Batch embedding (local embeddings - fast)
                self._process_batch_embeddings(batch_docs)
            else:
                # Sequential embedding with rate limiting (API-based)
                global_token_count, last_reset_time = self._process_sequential_embeddings(
                    batch_docs, global_token_count, last_reset_time, tokens_per_min_limit
                )

            # Save progress after each batch
            self.document_store.save_document_graph(self._get_document_graph_temp_path())

            # For API-based embeddings, pause between batches
            if not self.use_batch_embedding:
                logger.info(f"Completed batch of {len(batch)} documents, pausing for 2 seconds...")
                time.sleep(2)

        if cached_count > 0:
            logger.info(f"Used {cached_count} cached embeddings")
        
        # Second pass: create edges between documents
        logger.info("Creating document graph edges...")
        
        for i, entity in tqdm(enumerate(entities), total=total_entities, desc="Creating edges", unit="entity"):
            entity_uri = entity["entity"]
            
            # Get relationships (both incoming and outgoing)
            outgoing_rels = self.get_outgoing_relationships(entity_uri)
            incoming_rels = self.get_incoming_relationships(entity_uri)
            
            # Add edges for outgoing relationships with weights based on relationship type
            for rel in outgoing_rels:
                target_uri = rel["object"]
                predicate = rel["predicate"]
                
                # Determine weight based on relationship type
                weight = 1.0  # Default weight
                
                # Important CIDOC-CRM relationships get higher weights
                if "P89_falls_within" in predicate:
                    weight = 1.5  # Higher weight for spatial containment
                elif "P55_has_current_location" in predicate:
                    weight = 1.5  # Higher weight for location
                elif "P46_is_composed_of" in predicate or "P56_bears_feature" in predicate:
                    weight = 1.3  # Higher weight for physical relationships
                elif "P108i_was_produced_by" in predicate:
                    weight = 1.2  # Higher weight for production
                
                # Only add edge if both entities exist as documents
                if entity_uri in self.document_store.docs and target_uri in self.document_store.docs:
                    self.document_store.add_edge(
                        entity_uri, 
                        target_uri, 
                        predicate.split('/')[-1],
                        weight=weight
                    )
            
            # Add edges for incoming relationships
            for rel in incoming_rels:
                source_uri = rel["subject"]
                predicate = rel["predicate"]
                
                # Determine weight based on relationship type
                weight = 1.0  # Default weight
                
                # Important CIDOC-CRM relationships get higher weights
                if "P89_falls_within" in predicate:
                    weight = 1.5  # Higher weight for spatial containment
                elif "P55_has_current_location" in predicate:
                    weight = 1.5  # Higher weight for location
                elif "P46_is_composed_of" in predicate or "P56_bears_feature" in predicate:
                    weight = 1.3  # Higher weight for physical relationships
                elif "P108i_was_produced_by" in predicate:
                    weight = 1.2  # Higher weight for production
                
                # Only add edge if both entities exist as documents
                if entity_uri in self.document_store.docs and source_uri in self.document_store.docs:
                    self.document_store.add_edge(
                        entity_uri, 
                        source_uri, 
                        predicate.split('/')[-1],
                        weight=weight
                    )
        
        # Rename temp file to final
        temp_path = self._get_document_graph_temp_path()
        final_path = self._get_document_graph_path()
        if os.path.exists(temp_path):
            os.replace(temp_path, final_path)
        
        # Build vector store with batched embedding requests
        logger.info("Building vector store...")
        self.build_vector_store_batched()

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

    def build_vector_store_batched(self, batch_size=RetrievalConfig.DEFAULT_BATCH_SIZE):
        """
        Build vector store using pre-computed embeddings from GraphDocument objects.
        This avoids redundant API calls since embeddings were already generated
        during document graph building.
        """
        from langchain_community.vectorstores import FAISS

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
        """Enhanced retrieval using CIDOC-CRM aware scoring.

        Uses pre-computed graph edges from document_store instead of
        querying SPARQL at runtime. Edges are built during initialization
        with relationship weights based on CIDOC-CRM semantics.
        """

        # Initial vector search
        vector_results = self.document_store.retrieve(query, k=k*2)

        if not vector_results:
            return []

        # Get entity URIs from results for filtering
        entity_uris = set(doc.id for doc in vector_results)

        # Create a graph representation from pre-computed edges (no SPARQL needed)
        G = nx.DiGraph()

        # Add nodes and edges from document store
        for doc in vector_results:
            G.add_node(doc.id, label=doc.metadata.get("label", ""))

            # Add edges from pre-computed neighbors
            for neighbor in doc.neighbors:
                neighbor_id = neighbor["doc_id"]
                # Only add edges to other retrieved documents
                if neighbor_id in entity_uris:
                    weight = neighbor.get("weight", 0.5)
                    G.add_edge(doc.id, neighbor_id, weight=weight, predicate=neighbor.get("edge_type", ""))

        # Compute PageRank if graph has edges
        pagerank_scores = {}
        if G.number_of_edges() > 0:
            try:
                pagerank_scores = nx.pagerank(
                    G,
                    alpha=RetrievalConfig.PAGERANK_DAMPING,
                    max_iter=RetrievalConfig.PAGERANK_ITERATIONS
                )
            except Exception as e:
                logger.warning(f"PageRank computation failed: {str(e)}")

        # Re-rank documents by combined vector similarity and graph centrality
        ranked_docs = []

        for i, doc in enumerate(vector_results):
            # Vector score (inversely proportional to rank)
            vector_score = (len(vector_results) - i) / len(vector_results)

            # Graph score (from PageRank)
            graph_score = pagerank_scores.get(doc.id, 0.0)

            # Combined score
            combined_score = RetrievalConfig.VECTOR_PAGERANK_ALPHA * vector_score + (1 - RetrievalConfig.VECTOR_PAGERANK_ALPHA) * graph_score

            ranked_docs.append((doc, combined_score))

        # Sort by combined score
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top k documents
        return [doc for doc, _ in ranked_docs[:k]]

    def get_cidoc_system_prompt(self):
        """Get a system prompt with CIDOC-CRM knowledge"""

        return """You are a cultural heritage expert who provides clear, accessible answers about cultural heritage.

IMPORTANT - Natural Language Output:
- Write in clear, natural language suitable for general audiences
- NEVER use technical ontology identifiers (like E22_Man-Made_Object, E53_Place, IC9_Representation, D1_Digital_Object)
- Use everyday terms: instead of "E22_Man-Made_Object" say "church", "building", "artwork", "object"
- Instead of "E53_Place" say "place" or "location"
- Instead of "IC9_Representation" say "depiction", "image", or "representation"
- Focus on the actual entities and their meaningful relationships

Understanding the Data:
The data comes from a structured cultural heritage database (CIDOC-CRM) which organizes information about:
- Physical objects (churches, artworks, buildings)
- Places and locations
- Visual representations (paintings, icons, frescoes)
- People, events, and historical contexts
- Relationships between these entities

When answering questions:
1. Interpret the structured relationships to extract meaningful information
2. Present information in natural, flowing language
3. Focus on what matters to the user, not technical classifications
4. Use specific entity names (like "Panagia Phorbiottisa") rather than generic terms
5. If information is missing or unclear, say so plainly

Remember: Your audience wants to learn about cultural heritage, not database schemas. Make your answers informative and accessible.
"""

    def get_entity_literals(self, entity_uri):
        """Get all literal values for an entity (labels, WKT, dates, descriptions, etc.)"""
        query = f"""
        SELECT DISTINCT ?property ?value
        WHERE {{
            <{entity_uri}> ?property ?value .
            FILTER(isLiteral(?value))
        }}
        """

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            literals = {}
            for result in results["results"]["bindings"]:
                prop = result["property"]["value"]
                value = result["value"]["value"]

                # Store literals by property, handling multiple values
                prop_name = prop.split('/')[-1].split('#')[-1]
                if prop_name not in literals:
                    literals[prop_name] = []
                literals[prop_name].append(value)

            return literals
        except Exception as e:
            logger.error(f"Error fetching entity literals for {entity_uri}: {str(e)}")
            return {}

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
        """
        Get weight for a CIDOC-CRM relationship predicate.
        Higher weights indicate more semantically important relationships.

        Args:
            predicate_uri: Full URI of the predicate

        Returns:
            Float weight between 0 and 1
        """
        # Extract local name from URI
        local_name = predicate_uri.split('/')[-1].split('#')[-1]

        # CIDOC-CRM relationship weights
        weights = {
            # Spatial relationships (high weight - location is key context)
            "P89_falls_within": 0.9,
            "P89i_contains": 0.9,
            "P55_has_current_location": 0.9,
            "P55i_currently_holds": 0.9,

            # Physical composition
            "P56_bears_feature": 0.8,
            "P56i_is_found_on": 0.8,
            "P46_is_composed_of": 0.8,
            "P46i_forms_part_of": 0.8,

            # Creation/Production (important for authorship)
            "P108_has_produced": 0.85,
            "P108i_was_produced_by": 0.85,
            "P14_carried_out_by": 0.85,
            "P14i_performed": 0.85,
            "P94_has_created": 0.85,
            "P94i_was_created_by": 0.85,

            # Visual representation (VIR)
            "K24_portray": 0.7,
            "K24i_is_portrayed_in": 0.7,
            "K34_illustrates": 0.7,
            "K34i_is_illustrated_by": 0.7,

            # Type classification
            "P2_has_type": 0.6,
            "P2i_is_type_of": 0.6,

            # Documentation/Reference
            "P67_refers_to": 0.5,
            "P67i_is_referred_to_by": 0.5,
            "P70_documents": 0.5,
            "P70i_is_documented_in": 0.5,

            # Temporal
            "P4_has_time-span": 0.6,
            "P4i_is_time-span_of": 0.6,

            # Identification
            "P1_is_identified_by": 0.4,
            "P1i_identifies": 0.4,
        }

        # Try full URI first, then local name
        weight = weights.get(predicate_uri)
        if weight is None:
            weight = weights.get(local_name, 0.5)  # Default weight

        return weight

    def get_outgoing_relationships(self, entity_uri):
        """Get outgoing relationships from an entity (domain-level only, no schema predicates)"""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?predicate ?object WHERE {{
            <{entity_uri}> ?predicate ?object .

            # Filter for meaningful relationships
            FILTER(STRSTARTS(STR(?predicate), "http://"))
            FILTER(isURI(?object))
        }}
        """

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            relationships = []
            for result in results["results"]["bindings"]:
                predicate = result["predicate"]["value"]
                object_uri = result["object"]["value"]

                # Filter out schema-level predicates
                if self.is_schema_predicate(predicate):
                    continue

                relationships.append({
                    "predicate": predicate,
                    "object": object_uri
                })

            return relationships
        except Exception as e:
            logger.error(f"Error fetching outgoing relationships: {str(e)}")
            return []
    
    def get_incoming_relationships(self, entity_uri):
        """Get incoming relationships to an entity (domain-level only, no schema predicates)"""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?subject ?predicate WHERE {{
            ?subject ?predicate <{entity_uri}> .

            # Filter for meaningful relationships
            FILTER(STRSTARTS(STR(?predicate), "http://"))
            FILTER(isURI(?subject))
        }}
        """

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            relationships = []
            for result in results["results"]["bindings"]:
                subject_uri = result["subject"]["value"]
                predicate = result["predicate"]["value"]

                # Filter out schema-level predicates
                if self.is_schema_predicate(predicate):
                    continue

                relationships.append({
                    "subject": subject_uri,
                    "predicate": predicate
                })

            return relationships
        except Exception as e:
            logger.error(f"Error fetching incoming relationships: {str(e)}")
            return []


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


    def compute_coherent_subgraph(self, candidates, adjacency_matrix, initial_scores, k=RetrievalConfig.DEFAULT_RETRIEVAL_K, alpha=RetrievalConfig.RELEVANCE_CONNECTIVITY_ALPHA):
        """
        Extract a coherent subgraph using greedy selection that balances individual relevance and connectivity.

        Args:
            candidates: List of GraphDocument objects
            adjacency_matrix: Weighted adjacency matrix (n x n)
            initial_scores: Initial relevance scores for each candidate (n,)
            k: Number of documents to select
            alpha: Weight for individual relevance vs connectivity (0-1, higher = more emphasis on relevance)

        Returns:
            List of selected GraphDocument objects in order of selection
        """
        n = len(candidates)
        selected_indices = []
        selected_mask = np.zeros(n, dtype=bool)

        # Normalize initial scores to [0, 1] using min-max normalization
        normalized_scores = self.normalize_scores(initial_scores)

        logger.info(f"\n{'='*80}")
        logger.info(f"COHERENT SUBGRAPH EXTRACTION")
        logger.info(f"{'='*80}")
        logger.info(f"Parameters: k={k}, alpha={alpha} (relevance weight)")
        logger.info(f"Candidates: {n} documents")
        logger.info(f"\n--- Initial Relevance Scores (normalized to [0,1]) ---")
        logger.info(f"  Min: {np.min(normalized_scores):.3f}")
        logger.info(f"  Max: {np.max(normalized_scores):.3f}")
        logger.info(f"  Mean: {np.mean(normalized_scores):.3f}")
        logger.info(f"  Std: {np.std(normalized_scores):.3f}")

        # Show top 5 initial candidates
        logger.info(f"\n--- Top 5 Initial Candidates (by relevance) ---")
        top_indices = np.argsort(normalized_scores)[::-1][:5]
        for rank, idx in enumerate(top_indices, 1):
            label = candidates[idx].metadata.get('label', 'Unknown')
            logger.info(f"  {rank}. {label}: {normalized_scores[idx]:.3f}")

        # First selection: pick the highest-scoring document
        first_idx = np.argmax(normalized_scores)
        selected_indices.append(first_idx)
        selected_mask[first_idx] = True
        logger.info(f"\n{'='*80}")
        logger.info(f"SELECTION ROUND 1/{k}")
        logger.info(f"{'='*80}")
        logger.info(f"Strategy: Select highest relevance score")
        logger.info(f"Selected: {candidates[first_idx].metadata.get('label', 'Unknown')} (score={normalized_scores[first_idx]:.3f})")

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
                combined_score = alpha * relevance + (1 - alpha) * connectivity_norm
                all_scores.append((idx, combined_score, relevance, connectivity_norm))

            # Sort by combined score
            all_scores.sort(key=lambda x: x[1], reverse=True)

            # Show top 3 candidates for this iteration
            logger.info(f"\n--- Top 3 Candidates for Round {iteration+1} ---")
            for rank, (idx, combined, rel, conn) in enumerate(all_scores[:3], 1):
                label = candidates[idx].metadata.get('label', 'Unknown')
                logger.info(f"  {rank}. {label}")
                logger.info(f"      Relevance: {rel:.3f} (weight={alpha:.1f}) → contrib={alpha*rel:.3f}")
                logger.info(f"      Connectivity: {conn:.3f} (weight={1-alpha:.1f}) → contrib={(1-alpha)*conn:.3f}")
                logger.info(f"      Combined: {combined:.3f}")

            # Select the best
            best_idx, best_score, best_rel, best_connectivity_norm = all_scores[0]

            if best_idx == -1:
                logger.warning(f"Could not find more connected documents after {len(selected_indices)} selections")
                break

            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            logger.info(f"\n✓ SELECTED: {candidates[best_idx].metadata.get('label', 'Unknown')}")
            logger.info(f"  Final score: {best_score:.3f} = {alpha:.1f}×{best_rel:.3f} + {1-alpha:.1f}×{best_connectivity_norm:.3f}")

        # Log final summary
        logger.info(f"\n{'='*80}")
        logger.info(f"SUBGRAPH EXTRACTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Selected {len(selected_indices)}/{k} documents:")
        for i, idx in enumerate(selected_indices, 1):
            label = candidates[idx].metadata.get('label', 'Unknown')
            logger.info(f"  {i}. {label} (relevance={normalized_scores[idx]:.3f})")
        logger.info(f"{'='*80}\n")

        # Return selected documents in order
        return [candidates[idx] for idx in selected_indices]

    def retrieve(self, query, k=RetrievalConfig.DEFAULT_RETRIEVAL_K, initial_pool_size=30, alpha=RetrievalConfig.RELEVANCE_CONNECTIVITY_ALPHA):
        """
        Retrieve documents for a query using coherent subgraph extraction:
        1. CIDOC-CRM aware retrieval with relationship weights (initial pool)
        2. Coherent subgraph extraction using greedy selection based on relevance + connectivity
        
        Args:
            query: Query string
            k: Number of documents to return
            initial_pool_size: Size of initial candidate pool (should be > k)
            alpha: Balance between relevance (higher) and connectivity (lower)
        """
        logger.info(f"Retrieving documents for query: '{query}'")
        
        # First-stage retrieval with CIDOC-CRM aware retrieval
        # Get more candidates than needed for subgraph extraction
        initial_docs = self.cidoc_aware_retrieval(query, k=initial_pool_size)
        
        if not initial_docs:
            logger.warning("No documents found in first-stage retrieval")
            return []
        
        # If we got fewer documents than requested, just return them
        if len(initial_docs) <= k:
            logger.info(f"Retrieved {len(initial_docs)} documents (less than k={k})")
            return initial_docs
        
        # Create a subgraph of the retrieved documents
        doc_ids = [doc.id for doc in initial_docs]
        
        # Create weighted adjacency matrix with multi-hop connections
        adjacency_matrix = self.document_store.create_adjacency_matrix(doc_ids, max_hops=RetrievalConfig.MAX_ADJACENCY_HOPS)
        
        # Compute initial relevance scores based on ranking position
        # Higher rank = higher score (inverse of position)
        initial_scores = np.array([
            (len(initial_docs) - i) / len(initial_docs) 
            for i in range(len(initial_docs))
        ])
        
        # Extract coherent subgraph
        logger.info(f"Extracting coherent subgraph of size {k} from {len(initial_docs)} candidates")
        selected_docs = self.compute_coherent_subgraph(
            candidates=initial_docs,
            adjacency_matrix=adjacency_matrix,
            initial_scores=initial_scores,
            k=k,
            alpha=alpha
        )
        
        logger.info(f"Retrieved and selected {len(selected_docs)} coherent documents")
        return selected_docs

    def answer_question(self, question, include_wikidata=True):
        """Answer a question using the universal RAG system with CIDOC-CRM knowledge and optional Wikidata context"""
        # Validate input
        if not question or not question.strip():
            return {
                "answer": "Please provide a question.",
                "sources": []
            }

        logger.info(f"Answering question directly: '{question}'")

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, k=RetrievalConfig.DEFAULT_RETRIEVAL_K)

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
            context += doc_text + "\n\n"

            # Get Wikidata info if available and requested
            if include_wikidata:
                wikidata_id = self.get_wikidata_for_entity(entity_uri)
                if wikidata_id:
                    entities_with_wikidata.append({
                        "entity_uri": entity_uri,
                        "entity_label": entity_label,
                        "wikidata_id": wikidata_id
                    })

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

        # Add Wikidata instructions to system prompt
        if include_wikidata and wikidata_context:
            system_prompt += "\n\nI have also provided Wikidata information for some entities. When appropriate, incorporate this Wikidata information to enhance your answer with additional context, especially for factual details not present in the RDF data."

        # Create enhanced prompt
        prompt = f"""Answer the following question based on the retrieved information about cultural heritage entities:

Retrieved information:
{context}
"""

        # Add Wikidata context if available
        if include_wikidata and wikidata_context:
            prompt += f"{wikidata_context}\n"

        prompt += f"""
Question: {question}

Provide a clear, comprehensive answer in natural language:
- Use the entities' actual names (like "Panagia Phorbiottisa", "Nikitari") not document numbers
- Write in accessible language - avoid technical ontology codes (E22_, IC9_, D1_, etc.)
- Focus on meaningful information that answers the question
- Present relationships and context in natural, flowing prose
"""

        # Generate answer using the provider
        answer = self.llm_provider.generate(system_prompt, prompt)

        # Prepare sources
        sources = []
        for i, doc in enumerate(retrieved_docs):
            entity_uri = doc.id
            entity_label = doc.metadata.get("label", entity_uri.split('/')[-1])
            raw_triples = doc.metadata.get("raw_triples", [])

            sources.append({
                "id": i,
                "entity_uri": entity_uri,
                "entity_label": entity_label,
                "type": "graph",
                "entity_type": doc.metadata.get("type", "unknown"),
                "raw_triples": raw_triples
            })

        # Add Wikidata sources
        for entity_info in entities_with_wikidata:
            sources.append({
                "id": f"wikidata_{entity_info['wikidata_id']}",
                "entity_uri": entity_info["entity_uri"],
                "entity_label": entity_info["entity_label"],
                "type": "wikidata",
                "wikidata_id": entity_info["wikidata_id"],
                "wikidata_url": f"https://www.wikidata.org/wiki/{entity_info['wikidata_id']}"
            })

        return {
            "answer": answer,
            "sources": sources
        }