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
    categories: List[str]  # Primary FC categories (what the user wants returned)
    context_categories: List[str] = None  # Contextual FCs (mentioned but not the answer type)

QUERY_ANALYSIS_PROMPT = """Classify this question about cultural heritage data.

Query type (pick one):
- SPECIFIC: asks about a particular entity ("tell me about X", "where is X")
- ENUMERATION: asks to list entities ("which paintings…", "list all…", "what artists…")
- AGGREGATION: asks to count or rank ("how many…", "top 10…", "most…")

Categories — pick from: Thing, Actor, Place, Event, Concept, Time.
- "categories": the entity types the user wants as the ANSWER (1-2). What should the result list contain?
- "context_categories": entity types mentioned as FILTERS or CONTEXT but not the answer (0-2).

Examples:
- "Which pieces from Swiss Artists are in the museum?" → categories=["Thing"], context=["Actor","Place"]
- "List artists who exhibited in Paris" → categories=["Actor"], context=["Place","Event"]
- "How many churches are in Cyprus?" → categories=["Thing"], context=["Place"]
- "Tell me about Hodler" → categories=["Actor"]
- "What events took place in 1905?" → categories=["Event"], context=["Time"]
- "Which saints are depicted in the church?" → categories=["Actor","Thing"], context=["Place"]

Category definitions:
- Thing: objects, artworks, buildings, features, visual representations, iconographic subjects
- Actor: people, artists, groups, organizations, donors, saints, historical figures
- Place: locations, cities, regions
- Event: activities, creation, production, exhibitions
- Concept: types, materials, techniques
- Time: dates, periods

Return ONLY valid JSON: {{"query_type": "...", "categories": ["..."], "context_categories": ["..."]}}

Question: "{question}"
"""


def _fetch_wikidata_info(wikidata_id, session):
    """Fetch information from Wikidata for a given Q-ID.

    Pure function — uses the provided HTTP session for requests.
    Returns a dict with id, url, label, description, properties, wikipedia,
    or None on failure.
    """
    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            url = "https://www.wikidata.org/w/api.php"
            params = {
                "action": "wbgetentities",
                "ids": wikidata_id,
                "format": "json",
                "languages": "en",
                "props": "labels|descriptions|claims|sitelinks"
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; RAG-Bot/1.0; +http://example.com/bot)',
                'Accept': 'application/json'
            }
            response = session.get(url, params=params, headers=headers, timeout=10)

            if not response.text or response.status_code != 200:
                logger.warning(f"Wikidata API: status {response.status_code} for {wikidata_id} "
                              f"(attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue

            try:
                data = response.json()
            except ValueError as e:
                logger.warning(f"Wikidata JSON parse failed for {wikidata_id}: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue

            if "entities" not in data or wikidata_id not in data["entities"]:
                logger.warning(f"No entity data found for {wikidata_id}")
                return None

            entity = data["entities"][wikidata_id]
            result = {
                "id": wikidata_id,
                "url": f"https://www.wikidata.org/wiki/{wikidata_id}"
            }

            if "labels" in entity and "en" in entity["labels"]:
                result["label"] = entity["labels"]["en"]["value"]
            if "descriptions" in entity and "en" in entity["descriptions"]:
                result["description"] = entity["descriptions"]["en"]["value"]
            if "sitelinks" in entity and "enwiki" in entity["sitelinks"]:
                result["wikipedia"] = {
                    "title": entity["sitelinks"]["enwiki"]["title"],
                    "url": f"https://en.wikipedia.org/wiki/{entity['sitelinks']['enwiki']['title'].replace(' ', '_')}"
                }

            if "claims" in entity:
                result["properties"] = {}
                property_map = {
                    "P18": "image", "P571": "inception", "P17": "country",
                    "P131": "located_in", "P625": "coordinates",
                    "P1343": "described_by", "P138": "named_after",
                    "P180": "depicts", "P31": "instance_of", "P276": "location"
                }
                for prop_id, prop_name in property_map.items():
                    if prop_id in entity["claims"]:
                        values = []
                        for claim in entity["claims"][prop_id]:
                            if "mainsnak" not in claim or "datavalue" not in claim["mainsnak"]:
                                continue
                            dv = claim["mainsnak"]["datavalue"]
                            if dv["type"] == "wikibase-entityid":
                                values.append(dv["value"]["id"])
                            elif dv["type"] == "string":
                                values.append(dv["value"])
                            elif dv["type"] == "time":
                                values.append(dv["value"]["time"])
                            elif dv["type"] == "globecoordinate":
                                values.append({"latitude": dv["value"]["latitude"],
                                               "longitude": dv["value"]["longitude"]})
                        if values:
                            result["properties"][prop_name] = values[0] if len(values) == 1 else values

            return result

        except requests.exceptions.Timeout:
            logger.warning(f"Wikidata timeout for {wikidata_id} (attempt {attempt+1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Wikidata request failed for {wikidata_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected Wikidata error for {wikidata_id}: {e}")
        time.sleep(retry_delay)
        retry_delay *= 2

    logger.error(f"Failed to fetch Wikidata info after {max_retries} attempts for {wikidata_id}")
    return None



# ---------------------------------------------------------------------------
# Helper functions for _build_triples_enrichment (module-level to flatten
# the method body and reduce nesting)
# ---------------------------------------------------------------------------

# Predicate label fragments to skip (case-insensitive check)
_SKIP_LABEL_FRAGMENTS = {"label", "type", "same as", "see also"}

# High-priority predicate URI fragments (partial match on local name)
_HIGH_PRIORITY_FRAGMENTS = {
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

_DATE_PREDICATES = {
    "P82a_begin_of_the_begin", "P82b_end_of_the_end",
    "P82_at_some_time_within", "P81a_end_of_the_begin", "P81b_begin_of_the_end",
}

_TRIPLES_PRIORITY_TYPES = frozenset({
    "Actor", "E39_Actor", "E21_Person", "Person", "Group",
    "Human-Made Object", "E22_Human-Made_Object", "E22_Man-Made_Object",
    "Man-Made Object", "Physical Human-Made Thing",
    "Activity", "E7_Activity", "Event", "E5_Event",
    "Place", "E53_Place",
})


def _is_skip_label(pred_label):
    """Check if predicate label is too generic to be useful."""
    if not pred_label:
        return True
    return pred_label.strip().lower() in _SKIP_LABEL_FRAGMENTS


def _is_blank_or_hash(value):
    """Check if a value looks like a blank node or non-informative hash URI."""
    if not value:
        return True
    if value.startswith("_:"):
        return True
    if "#" in value and "/" not in value.split("#")[-1]:
        return True
    return False


def _predicate_priority(triple, entity_uri, retrieved_uris):
    """Return priority score: 0 = highest (inter-doc), 1 = high, 2 = medium, 3 = low."""
    other_uri = triple["object"] if triple["subject"] == entity_uri else triple["subject"]
    if other_uri in retrieved_uris:
        return 0
    pred = triple.get("predicate", "")
    local_name = pred.split("/")[-1].split("#")[-1]
    for frag in _HIGH_PRIORITY_FRAGMENTS:
        if frag in local_name:
            return 1
    if triple.get("predicate_label"):
        return 2
    return 3


def _resolve_time_span(time_span_uri, triples_index):
    """Follow a time-span URI to extract begin/end date values."""
    ts_triples = triples_index.get(time_span_uri, [])
    dates = {}
    for t in ts_triples:
        if t["subject"] != time_span_uri:
            continue
        pred_local = t["predicate"].split("/")[-1].split("#")[-1]
        for dp in _DATE_PREDICATES:
            if dp in pred_local:
                obj_val = t.get("object_label") or t.get("object", "")
                if obj_val and not obj_val.startswith("http"):
                    if "begin" in dp:
                        dates["began"] = obj_val
                    elif "end" in dp:
                        dates["ended"] = obj_val
                    else:
                        dates["date"] = obj_val
    return dates


def _triples_type_priority(doc):
    """Sort key: informative entity types first for budget allocation."""
    return 0 if doc.metadata.get('type', '') in _TRIPLES_PRIORITY_TYPES else 1


class RetrievalConfig:
    """Configuration constants for the RAG retrieval system"""

    # Score combination weights
    RELEVANCE_CONNECTIVITY_ALPHA = 0.7  # Weight for combining relevance and connectivity scores

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

    # Type-filtered retrieval channel
    TYPE_FILTERED_FETCH_K = 10000   # Raw FAISS results scanned before type filtering
    TYPE_CHANNEL_POOL_FRACTION = 0.5  # Fraction of pool reserved for type-matching docs (ENUM/AGG)
    TYPE_CHANNEL_POOL_FRACTION_SPECIFIC = 0.3  # Smaller fraction for SPECIFIC queries (safety net)

    # Thin document chaining: absorb thin docs into their richest neighbor
    THIN_DOC_THRESHOLD = 400  # Docs with fewer chars are candidates for chaining
    MAX_CHAINED_DOC_SIZE = 5000  # Stop absorbing into a target that would exceed this

    # Context assembly: per-document truncation when building the LLM prompt
    MAX_DOC_CHARS = 5000  # Aligned with MAX_CHAINED_DOC_SIZE

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
        # Non-informative types - stronger penalty
        "Linguistic Object": -0.35,
        "E33_Linguistic_Object": -0.35,
        "E33_E41_Linguistic_Appellation": -0.35,
        "E41_E33_Linguistic_Appellation": -0.35,
        "Linguistic Appellation": -0.35,
        "Inscription": -0.40,
        "E34_Inscription": -0.40,
        "E31_Document": -0.20,
        "Document": -0.20,
        "Appellation": -0.25,
        "E41_Appellation": -0.25,
        # Reference types - moderate penalty
        "PC67_refers_to": -0.25,
    }

    # Mega-entity penalty: entities with very high triple counts provide diluted context
    MEGA_ENTITY_TRIPLES_THRESHOLD = 2000   # Entities with more triples than this are penalized
    MEGA_ENTITY_PENALTY = 0.15             # Score penalty (subtracted after type modifier)

    # Pool pre-filtering: cap the ratio of non-informative entity types
    NON_INFORMATIVE_TYPES = frozenset({
        "Linguistic Object", "E33_Linguistic_Object",
        "E33_E41_Linguistic_Appellation", "E41_E33_Linguistic_Appellation",
        "Linguistic Appellation", "Inscription", "E34_Inscription",
        "Appellation", "E41_Appellation", "PC67_refers_to",
    })
    MAX_NON_INFORMATIVE_RATIO = 0.25


class UniversalRagSystem:
    """Universal RAG system with graph-based document retrieval"""

    # Class-level cache for property labels and ontology classes
    _property_labels = None
    _ontology_classes = None
    _class_labels = None  # Cache for class URI -> English label mapping
    _inverse_properties = None  # Cache for property URI -> inverse property URI mapping
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
            cache_dir = os.path.join(self._path('cache'), "embeddings")
            self.embedding_cache = EmbeddingCache(cache_dir)
            logger.info(f"Embedding cache enabled at {cache_dir}")
        else:
            self.embedding_cache = None
            logger.info("Embedding cache disabled")

        # Initialize document store
        self.document_store = None
        self._aggregation_index = {}
        self._parquet_writer = None
        self._parquet_triple_count = 0

        # Create a secure session for HTTP requests (e.g., Wikidata API)
        # Disable trust_env to prevent .netrc credential leaks (CVE fix)
        self._http_session = requests.Session()
        self._http_session.trust_env = False

        # Load property labels and ontology classes from ontology extraction (cached at class level)
        if UniversalRagSystem._property_labels is None:
            UniversalRagSystem._property_labels = self._load_ontology_json('property_labels.json')

        if UniversalRagSystem._ontology_classes is None:
            UniversalRagSystem._ontology_classes = self._load_ontology_json('ontology_classes.json', as_set=True)

        if UniversalRagSystem._class_labels is None:
            UniversalRagSystem._class_labels = self._load_ontology_json('class_labels.json')

        if UniversalRagSystem._inverse_properties is None:
            UniversalRagSystem._inverse_properties = self._load_inverse_properties()

        # Initialize FR traversal for FR-based document generation
        self.fr_traversal = self._init_fr_traversal()

    # ==================== Path Helper Methods ====================
    # These methods return dataset-specific paths for multi-dataset support

    def _path(self, key: str) -> str:
        """Return a dataset-specific path by key.

        Keys: cache, graph, graph_temp, vector_dir, vector_index,
              bm25_dir, aggregation, documents
        """
        base = self.data_dir if self.data_dir else str(PROJECT_ROOT / 'data')
        cache = f'{base}/cache/{self.dataset_id}'
        paths = {
            'cache':        cache,
            'graph':        f'{cache}/document_graph.pkl',
            'graph_temp':   f'{cache}/document_graph_temp.pkl',
            'vector_dir':   f'{cache}/vector_index',
            'vector_index': f'{cache}/vector_index/index.faiss',
            'bm25_dir':     f'{cache}/bm25_index',
            'aggregation':  f'{cache}/aggregation_index.json',
            'documents':    f'{base}/documents/{self.dataset_id}/entity_documents',
        }
        return paths[key]

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

    def _init_fr_traversal(self) -> FRTraversal:
        """Initialize FR traversal module with required config files.

        Raises FileNotFoundError if any required config file is missing.
        """
        fr_json = str(PROJECT_ROOT / 'config' / 'fundamental_relationships_cidoc_crm.json')
        inverse_props = str(PROJECT_ROOT / 'data' / 'labels' / 'inverse_properties.json')
        fc_mapping = str(PROJECT_ROOT / 'config' / 'fc_class_mapping.json')

        for path, label in [
            (fr_json, "FR JSON"),
            (inverse_props, "Inverse properties"),
            (fc_mapping, "FC class mapping"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{label} not found: {path}. "
                    "Run: python scripts/extract_ontology_labels.py to generate it."
                )

        traversal = FRTraversal(
            fr_json_path=fr_json,
            inverse_properties_path=inverse_props,
            fc_mapping_path=fc_mapping,
            property_labels=UniversalRagSystem._property_labels
        )
        logger.info("FR traversal initialized for document generation")
        return traversal

    def _ensure_ontology_extraction(self):
        """Run ontology label extraction if any label files are missing. Returns True on success."""
        ontology_dir = str(PROJECT_ROOT / 'data' / 'ontologies')
        if not os.path.exists(ontology_dir):
            logger.error(f"Ontology directory not found at '{ontology_dir}'")
            return False
        ontology_files = [f for f in os.listdir(ontology_dir) if f.endswith(('.ttl', '.rdf', '.owl', '.n3'))]
        if not ontology_files:
            logger.error(f"No ontology files found in '{ontology_dir}'")
            return False
        logger.info(f"Extracting ontology labels from {len(ontology_files)} files...")
        try:
            labels_dir = str(PROJECT_ROOT / 'data' / 'labels')
            success = run_extraction(
                ontology_dir,
                f'{labels_dir}/property_labels.json',
                f'{labels_dir}/ontology_classes.json',
                f'{labels_dir}/class_labels.json',
            )
            if success:
                UniversalRagSystem._extraction_attempted = True
            return success
        except Exception as e:
            logger.error(f"Ontology extraction failed: {e}")
            return False

    def _load_ontology_json(self, filename, as_set=False):
        """Load a JSON resource from data/labels/, extracting from ontologies if missing.

        Args:
            filename: JSON filename in data/labels/ (e.g. 'property_labels.json')
            as_set: If True, convert loaded list to set

        Returns:
            dict, set, or empty default on failure
        """
        json_path = str(PROJECT_ROOT / 'data' / 'labels' / filename)
        empty = set() if as_set else {}

        if not os.path.exists(json_path):
            if not self._ensure_ontology_extraction():
                return empty

        if not os.path.exists(json_path):
            logger.error(f"File not found after extraction: {json_path}")
            return empty

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            result = set(data) if as_set else data
            logger.info(f"Loaded {len(result)} entries from {filename}")
            return result
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return empty

    @property
    def embeddings(self):
        """
        Return an Embeddings object compatible with FAISS and the rest of the code.
        Uses the embedding_provider (which may be different from llm_provider).
        """
        from langchain_core.embeddings import Embeddings

        class EmbeddingFunction(Embeddings):
            def __init__(self, provider):
                self.provider = provider

            def embed_query(self, text: str) -> list[float]:
                return self.provider.get_embeddings(text)

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
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
        doc_graph_path = self._path('graph')
        vector_index_path = self._path('vector_index')
        vector_index_dir = self._path('vector_dir')

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
                self._load_aggregation_index()
                # Load or build BM25 index
                bm25_dir = self._path('bm25_dir')
                if not self.document_store.load_bm25_index(bm25_dir):
                    logger.info("BM25 index not cached, building from loaded documents...")
                    if self.document_store.build_bm25_index():
                        self.document_store.save_bm25_index(bm25_dir)
                # Build FC type index for type-filtered retrieval
                self._build_fc_type_index()
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
            bm25_dir = self._path('bm25_dir')
            self.document_store.save_bm25_index(bm25_dir)

        # Build FC type index for type-filtered retrieval
        self._build_fc_type_index()

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
        # P82a/P82b = outer bounds, P81a/P81b = inner bounds (used by MAH)
        # If both exist, first match wins (P82a before P81a)
        _TIME_PROPS = {
            "P82a_begin_of_the_begin": "begin",
            "P82b_end_of_the_end": "end",
            "P82_at_some_time_within": "within",
            "P81a_end_of_the_begin": "begin",
            "P81b_begin_of_the_end": "end",
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

    def _chain_thin_documents(self) -> Dict[str, str]:
        """Chain thin documents into ALL graph neighbors.

        After edges are built, scans for documents below THIN_DOC_THRESHOLD
        characters.  For each, appends the thin doc's body as a "Related
        entity" section to **every** neighbor that won't exceed
        MAX_CHAINED_DOC_SIZE.  The thin doc is then removed from the store
        so it won't be embedded or indexed.

        Replicating into all neighbors ensures that queries in any
        neighborhood of the thin entity find its information, regardless
        of which neighbor is retrieved.

        Processing order is ascending by text length so that the smallest docs
        are absorbed first (cascading: a thin doc absorbed into another thin
        doc may push the target above threshold).

        Returns:
            Dict mapping chained doc_id → first target doc_id it was merged into.
        """
        store = self.document_store
        threshold = RetrievalConfig.THIN_DOC_THRESHOLD
        max_target_size = RetrievalConfig.MAX_CHAINED_DOC_SIZE

        # FC categories exempt from chaining (carry irreplaceable temporal/spatial info)
        _EXEMPT_FC_PREFIXES = ("[Event]", "[Actor]")

        # Collect candidates: thin docs that have at least one neighbor
        candidates = []
        skipped_exempt = 0
        for doc_id, doc in store.docs.items():
            if len(doc.text) < threshold and doc.neighbors:
                # Exempt entities whose FC category makes them independently important
                if any(doc.text.lstrip().startswith(pfx) for pfx in _EXEMPT_FC_PREFIXES):
                    skipped_exempt += 1
                    continue
                candidates.append((doc_id, len(doc.text)))

        if not candidates:
            logger.info("Thin-doc chaining: no candidates found")
            return {}

        # Process smallest first for cascading
        candidates.sort(key=lambda x: x[1])

        chained_map: Dict[str, str] = {}  # thin_id → first target_id

        for doc_id, _ in candidates:
            if doc_id not in store.docs:
                continue  # already removed by earlier iteration

            doc = store.docs[doc_id]

            # Build compact section to append (once per thin doc)
            label = doc.metadata.get("label", doc_id.split("/")[-1])
            primary_type = doc.metadata.get("type", "Entity")

            # Strip YAML frontmatter from thin doc text
            body = doc.text
            if body.startswith("---"):
                end_idx = body.find("---", 3)
                if end_idx != -1:
                    body = body[end_idx + 3:].strip()

            chain_section = f"\n\nRelated entity — {label} ({primary_type}):\n{body}"
            chain_len = len(chain_section)

            # Absorb into ALL unique neighbors that won't exceed size cap
            absorbed_into: list[str] = []
            seen_neighbors: set[str] = set()
            for neighbor_info in doc.neighbors:
                n_id = neighbor_info["doc_id"]
                if n_id in seen_neighbors:
                    continue  # skip duplicate neighbor entries (different edge types)
                seen_neighbors.add(n_id)
                if n_id not in store.docs or n_id == doc_id:
                    continue
                if len(store.docs[n_id].text) + chain_len > max_target_size:
                    continue  # would exceed size cap

                store.docs[n_id].text += chain_section

                # Propagate metadata for source tracking
                target_meta = store.docs[n_id].metadata
                chained_entities = target_meta.get("chained_entities", [])
                chained_entities.append({
                    "uri": doc.metadata.get("uri", doc_id),
                    "label": label,
                    "type": primary_type,
                })
                target_meta["chained_entities"] = chained_entities

                absorbed_into.append(n_id)

            if not absorbed_into:
                continue  # no neighbor could accept it

            # Remove thin doc from store
            del store.docs[doc_id]

            # Clean up neighbor lists pointing to the removed doc
            for neighbor_info in doc.neighbors:
                n_id = neighbor_info["doc_id"]
                if n_id in store.docs:
                    store.docs[n_id].neighbors = [
                        n for n in store.docs[n_id].neighbors
                        if n["doc_id"] != doc_id
                    ]

            chained_map[doc_id] = absorbed_into[0]

        logger.info(
            f"Thin-doc chaining: {len(chained_map)} docs absorbed into neighbors, "
            f"{skipped_exempt} exempt (Event/Actor) "
            f"({len(store.docs)} docs remaining)"
        )

        return chained_map

    def _get_edges_parquet_path(self) -> str:
        """Return the path to the edges Parquet file."""
        return os.path.join(os.path.dirname(self._path('documents')), "edges.parquet")

    def _append_edges_parquet(self, triples: List[Dict]) -> None:
        """Append triples to the Parquet edges file incrementally.

        Creates the file on first call, appends on subsequent calls.
        Uses a persistent ParquetWriter for efficient streaming writes.
        """
        import pyarrow as pa

        if not triples:
            return

        table = pa.table({
            "s": [t["subject"] for t in triples],
            "s_label": [t.get("subject_label", "") for t in triples],
            "p": [t["predicate"] for t in triples],
            "p_label": [t.get("predicate_label", "") for t in triples],
            "o": [t["object"] for t in triples],
            "o_label": [t.get("object_label", "") for t in triples],
        })

        if self._parquet_writer is None:
            import pyarrow.parquet as pq
            edges_path = self._get_edges_parquet_path()
            os.makedirs(os.path.dirname(edges_path), exist_ok=True)
            self._parquet_writer = pq.ParquetWriter(edges_path, table.schema)
            self._parquet_triple_count = 0

        self._parquet_writer.write_table(table)
        self._parquet_triple_count += len(triples)

    def _close_edges_parquet(self) -> None:
        """Close the incremental Parquet writer and log final stats."""
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
            edges_path = self._get_edges_parquet_path()
            size_mb = os.path.getsize(edges_path) / (1024 * 1024)
            logger.info(f"Saved {self._parquet_triple_count:,} triples to {edges_path} ({size_mb:.1f} MB)")
            self._parquet_triple_count = 0

    def _load_triples_index(self):
        """Load Parquet edges file and build entity -> triples index.

        Reads the enriched edges.parquet (6 columns: s, s_label, p, p_label, o, o_label)
        and creates a dict mapping each entity URI to its list of triple dicts.
        Each triple appears under both its subject and object URI.

        Falls back gracefully if the file is missing or uses the old 3-column format.
        """
        edges_path = os.path.join(os.path.dirname(self._path('documents')), "edges.parquet")
        if not os.path.exists(edges_path):
            logger.warning(f"No edges file at {edges_path} — raw_triples will be empty in responses")
            self._triples_index = {}
            self._actor_work_counts = {}
            return

        import pyarrow.parquet as pq

        table = pq.read_table(edges_path)
        columns = set(table.column_names)

        # Check for enriched format (6 columns with labels)
        has_labels = {"s_label", "p_label", "o_label"}.issubset(columns)
        if not has_labels:
            logger.warning("edges.parquet uses old 3-column format — re-generate docs to get labels")
            self._triples_index = {}
            self._actor_work_counts = {}
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

        # Build actor work-count index by tracing production chains:
        # Work →[P108i]→ creation/impression →[P14]→ actor
        # Work →[P16i]→ edition/creation/impression →[P14]→ actor
        creation_to_actors = {}   # event_uri → [actor_uri, ...]
        work_to_events = {}       # work_uri → [event_uri, ...]
        for s, p, o in zip(table.column("s"), table.column("p"), table.column("o")):
            s_py, p_py, o_py = s.as_py(), p.as_py(), o.as_py()
            local = p_py.split("/")[-1].split("#")[-1]
            if "P14_carried_out_by" in local or "P14i_performed" in local:
                if local.endswith("_carried_out_by"):
                    creation_to_actors.setdefault(s_py, []).append(o_py)
                else:  # P14i: actor → event
                    creation_to_actors.setdefault(o_py, []).append(s_py)
            elif "P108i_was_produced_by" in local:
                work_to_events.setdefault(s_py, []).append(o_py)
            elif "P16i_was_used_for" in local:
                # Only count production-type events (creation/edition/impression)
                if any(pat in o_py for pat in ("/creation", "/edition", "/impression")):
                    work_to_events.setdefault(s_py, []).append(o_py)

        actor_work_counts = {}
        for work_uri, event_uris in work_to_events.items():
            counted_actors = set()
            for event_uri in event_uris:
                for actor_uri in creation_to_actors.get(event_uri, []):
                    counted_actors.add(actor_uri)
            for actor_uri in counted_actors:
                actor_work_counts[actor_uri] = actor_work_counts.get(actor_uri, 0) + 1

        self._actor_work_counts = actor_work_counts
        if actor_work_counts:
            top_3 = sorted(actor_work_counts.items(), key=lambda x: -x[1])[:3]
            top_labels = []
            for uri, count in top_3:
                label = uri.split("/")[-1]
                for t in index.get(uri, [])[:5]:
                    if t.get("subject") == uri and t.get("subject_label"):
                        label = t["subject_label"]
                        break
                    if t.get("object") == uri and t.get("object_label"):
                        label = t["object_label"]
                        break
                top_labels.append(f"{label}({count})")
            logger.info(f"Actor work-count index: {len(actor_work_counts)} actors, "
                        f"top: {', '.join(top_labels)}")

        logger.info(f"Loaded triples index from {edges_path}: "
                    f"{len(table)} triples, {len(index)} entities indexed")

    def _build_aggregation_index(self, all_fr_stats=None):
        """Build pre-computed aggregation metrics from FR traversal stats.

        Uses the FR traversal results accumulated during document generation
        to count multi-hop CIDOC-CRM relationships (e.g. Actor→Event→Thing).
        Falls back to empty index if no FR stats provided and no cached index.

        Args:
            all_fr_stats: List of (entity_uri, entity_label, fr_stats_dict)
                accumulated during process_rdf_data(). Each fr_stats_dict has
                "fc" (str) and "fr_results" (list of FR match dicts).
        """
        if not all_fr_stats:
            logger.info("No FR stats provided — skipping aggregation index build")
            self._aggregation_index = {}
            return

        logger.info(f"Building FR-based aggregation index from {len(all_fr_stats)} entities...")

        # --- 1. Entity type counts from document store ---
        type_counts: Dict[str, int] = {}
        if self.document_store:
            for doc in self.document_store.docs.values():
                doc_type = doc.metadata.get("type", "Unknown")
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        # --- 2. FC entity counts ---
        fc_counts: Dict[str, int] = {}
        for _uri, _label, stats in all_fr_stats:
            fc = stats.get("fc")
            if fc:
                fc_counts[fc] = fc_counts.get(fc, 0) + 1

        # --- 3. FR summaries — single pass over accumulated FR results ---
        # Per-FR accumulators
        source_counts: Dict[str, Dict[str, int]] = {}   # fr_id → {entity_uri → target_count}
        source_labels: Dict[str, Dict[str, str]] = {}   # fr_id → {entity_uri → label}
        target_counts: Dict[str, Dict[str, int]] = {}   # fr_id → {target_uri → appearance_count}
        target_labels: Dict[str, Dict[str, str]] = {}   # fr_id → {target_uri → label}

        for entity_uri, entity_label, stats in all_fr_stats:
            for fr in stats.get("fr_results", []):
                fr_id = fr["fr_id"]

                if fr_id not in source_counts:
                    source_counts[fr_id] = {}
                    source_labels[fr_id] = {}
                    target_counts[fr_id] = {}
                    target_labels[fr_id] = {}

                source_counts[fr_id][entity_uri] = \
                    source_counts[fr_id].get(entity_uri, 0) + fr["total_count"]
                source_labels[fr_id][entity_uri] = entity_label

                for t_uri, t_label in fr["targets"]:
                    target_counts[fr_id][t_uri] = target_counts[fr_id].get(t_uri, 0) + 1
                    target_labels[fr_id][t_uri] = t_label

        # Build FR metadata lookup from fr_traversal
        fr_meta = {}
        if self.fr_traversal:
            for fr in self.fr_traversal.fr_list:
                fr_meta[fr["id"]] = {
                    "label": fr["label"],
                    "domain_fc": fr["domain_fc"],
                    "range_fc": fr["range_fc"],
                }

        def _top_n(counts_dict, labels_dict, n=50):
            sorted_items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)[:n]
            return [
                {"uri": uri, "label": labels_dict.get(uri, uri.rsplit("/", 1)[-1]), "count": count}
                for uri, count in sorted_items
            ]

        # Build per-FR summary
        fr_summaries = {}
        for fr_id in source_counts:
            meta = fr_meta.get(fr_id, {})
            total_connections = sum(source_counts[fr_id].values())
            fr_summaries[fr_id] = {
                "label": meta.get("label", fr_id),
                "domain_fc": meta.get("domain_fc", ""),
                "range_fc": meta.get("range_fc", ""),
                "total_connections": total_connections,
                "unique_sources": len(source_counts[fr_id]),
                "unique_targets": len(target_counts[fr_id]),
                "top_sources": _top_n(source_counts[fr_id], source_labels[fr_id]),
                "top_targets": _top_n(target_counts[fr_id], target_labels[fr_id]),
            }

        agg_index = {
            "entity_type_counts": type_counts,
            "fc_counts": fc_counts,
            "fr_summaries": fr_summaries,
            "total_entities": len(self.document_store.docs) if self.document_store else 0,
        }

        # Compute PageRank on the FR graph
        pagerank_data = self._compute_pagerank(all_fr_stats, top_n=500)
        if pagerank_data:
            agg_index["pagerank"] = pagerank_data

        # Save to JSON
        out_path = self._path('aggregation')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(agg_index, f, ensure_ascii=False, indent=2)

        self._aggregation_index = agg_index
        active_frs = [fid for fid, s in fr_summaries.items() if s["total_connections"] > 0]
        logger.info(
            f"Built FR-based aggregation index: {len(type_counts)} types, "
            f"{len(fc_counts)} FCs, {len(active_frs)} active FRs — saved to {out_path}"
        )

    def _compute_pagerank(self, all_fr_stats, top_n=500):
        """Compute PageRank on the FR graph for centrality-based retrieval.

        Builds a directed graph from FR traversal results (multi-hop semantic
        connections) and runs PageRank. Returns top entities per FC for use
        in the type-filtered retrieval channel.

        Args:
            all_fr_stats: List of (entity_uri, entity_label, fr_stats_dict).
            top_n: Number of top entities to keep per FC.

        Returns:
            Dict with "metadata", "by_fc", and "global_top" keys, or None on failure.
        """
        import networkx as nx

        G = nx.DiGraph()

        # Track entity metadata (label, fc)
        entity_meta = {}  # uri → {"label": str, "fc": str}

        for entity_uri, entity_label, stats in all_fr_stats:
            fc = stats.get("fc", "")
            entity_meta[entity_uri] = {"label": entity_label, "fc": fc}

            for fr in stats.get("fr_results", []):
                for target_uri, target_label in fr["targets"]:
                    if G.has_edge(entity_uri, target_uri):
                        G[entity_uri][target_uri]["weight"] += 1
                    else:
                        G.add_edge(entity_uri, target_uri, weight=1)
                    # Store target label if not already tracked
                    if target_uri not in entity_meta:
                        entity_meta[target_uri] = {"label": target_label, "fc": ""}

        if G.number_of_nodes() == 0:
            logger.info("PageRank: empty graph, skipping")
            return None

        # Run PageRank
        pr_scores = nx.pagerank(G, alpha=0.85, weight="weight")

        # Filter to entities that have documents in the store
        doc_ids = set(self.document_store.docs.keys()) if self.document_store else set()

        # Partition by FC and build ranked lists
        by_fc = {}  # fc → [(uri, label, score), ...]
        global_list = []

        for uri, score in pr_scores.items():
            if uri not in doc_ids:
                continue
            meta = entity_meta.get(uri, {})
            label = meta.get("label", uri.rsplit("/", 1)[-1])
            fc = meta.get("fc", "")

            entry = {"uri": uri, "label": label, "score": score}
            global_list.append({**entry, "fc": fc})

            if fc:
                by_fc.setdefault(fc, []).append(entry)

        # Sort and truncate
        for fc in by_fc:
            by_fc[fc] = sorted(by_fc[fc], key=lambda x: x["score"], reverse=True)[:top_n]
        global_top = sorted(global_list, key=lambda x: x["score"], reverse=True)[:top_n]

        scored_count = len(global_list)
        logger.info(
            f"PageRank computed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
            f"{scored_count} entities with docs, {len(by_fc)} FCs"
        )

        return {
            "metadata": {
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "alpha": 0.85,
            },
            "by_fc": {fc: entries for fc, entries in by_fc.items()},
            "global_top": global_top,
        }

    def _load_aggregation_index(self):
        """Load pre-computed aggregation index from JSON.

        Graceful fallback: if file missing or corrupt, sets empty dict.
        """
        path = self._path('aggregation')
        if not os.path.exists(path):
            logger.info("No aggregation index found — AGGREGATION queries will work without pre-computed stats")
            self._aggregation_index = {}
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                self._aggregation_index = json.load(f)
            fr_count = len(self._aggregation_index.get('fr_summaries', {}))
            logger.info(
                f"Loaded aggregation index from {path}: "
                f"{self._aggregation_index.get('total_entities', '?')} entities, "
                f"{fr_count} FRs"
            )
        except Exception as e:
            logger.warning(f"Failed to load aggregation index from {path}: {e}")
            self._aggregation_index = {}

    def _build_edges_from_parquet(self):
        """Build edges from Parquet edges file. O(n*r) where r is number of triples.

        Reads edges.parquet (saved during --generate-docs) and adds weighted edges
        to the document store using proper CIDOC-CRM semantic weights.
        """
        edges_path = os.path.join(os.path.dirname(self._path('documents')), "edges.parquet")
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
                weight = _get_relationship_weight(p_str)
                pred_name = p_str.split('/')[-1].split('#')[-1]
                self.document_store.add_edge(s_str, o_str, pred_name, weight=weight)
                edges_added += 1
            else:
                skipped += 1

        logger.info(f"Added {edges_added} edges from Parquet file "
                    f"({skipped} triples skipped — endpoints not in document store)")


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
        raw_outgoing = self.batch_sparql.batch_query_outgoing(chunk_uris)
        raw_incoming = self.batch_sparql.batch_query_incoming(chunk_uris)

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
                if _is_schema_predicate(pred):
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
                if _is_schema_predicate(pred):
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

            inter_outgoing = self.batch_sparql.batch_query_outgoing(intermediate_list)
            inter_incoming = self.batch_sparql.batch_query_incoming(intermediate_list)

            for uri, rels in inter_outgoing.items():
                for pred, obj, obj_label in rels:
                    if _is_schema_predicate(pred):
                        continue
                    fr_outgoing[uri].append((pred, obj))
                    if obj_label:
                        entity_labels[obj] = obj_label
                    elif obj not in entity_labels:
                        entity_labels[obj] = obj.split('/')[-1].split('#')[-1]
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

            for uri, rels in inter_incoming.items():
                for subj, pred, subj_label in rels:
                    if _is_schema_predicate(pred):
                        continue
                    fr_incoming[uri].append((pred, subj))
                    if subj_label:
                        entity_labels[subj] = subj_label
                    elif subj not in entity_labels:
                        entity_labels[subj] = subj.split('/')[-1].split('#')[-1]
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

            # Fetch types for intermediates (needed for FR range-FC filtering)
            inter_types = self.batch_sparql.batch_fetch_types(intermediate_list)
            chunk_types.update(inter_types)

        # entity_types_map = chunk_types (already updated with intermediates)
        entity_types_map = chunk_types

        # Add date literals for E52_Time-Span entities to raw_triples.
        # batch_query_outgoing uses FILTER(isURI(?o)) so P82a/P82b literal values
        # are excluded.  We fetch them here so _resolve_time_span can find them
        # in the triples index at query time.
        _E52_URI = "http://www.cidoc-crm.org/cidoc-crm/E52_Time-Span"
        _CRM_NS = "http://www.cidoc-crm.org/cidoc-crm/"
        _TS_DATE_PROPS = {
            "P82a_begin_of_the_begin", "P82b_end_of_the_end",
            "P82_at_some_time_within", "P81a_end_of_the_begin",
            "P81b_begin_of_the_end",
        }

        ts_from_chunk = [u for u in chunk_uris if _E52_URI in chunk_types.get(u, set())]
        ts_from_inter = [u for u in intermediate_uris if _E52_URI in chunk_types.get(u, set())]

        # Chunk time-spans already have literals in chunk_literals;
        # intermediate time-spans need a fresh fetch.
        inter_ts_lits = (
            self.batch_sparql.batch_fetch_literals(ts_from_inter) if ts_from_inter else {}
        )
        ts_date_count = 0
        for ts_uri, ts_lits in (
            [(u, chunk_literals.get(u, {})) for u in ts_from_chunk] +
            [(u, inter_ts_lits.get(u, {})) for u in ts_from_inter]
        ):
            for prop_local, vals in ts_lits.items():
                if prop_local not in _TS_DATE_PROPS or not vals:
                    continue
                prop_label = ""
                if UniversalRagSystem._property_labels:
                    prop_label = (
                        UniversalRagSystem._property_labels.get(f"{_CRM_NS}{prop_local}") or
                        UniversalRagSystem._property_labels.get(prop_local) or ""
                    )
                raw_triples.append({
                    "subject": ts_uri,
                    "subject_label": entity_labels.get(ts_uri, ts_uri.split('/')[-1]),
                    "predicate": f"{_CRM_NS}{prop_local}",
                    "predicate_label": prop_label,
                    "object": vals[0],
                    "object_label": vals[0],
                })
                ts_date_count += 1

        logger.info(f"    FR graph: {len(fr_outgoing)} outgoing, {len(fr_incoming)} incoming, "
                    f"{len(entity_labels)} labels, {len(raw_triples)} raw triples"
                    + (f" ({ts_date_count} date literals)" if ts_date_count else ""))

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
    ) -> Tuple[str, str, List[str], dict]:
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
            (text, label, type_labels, fr_stats) where fr_stats contains
            the entity's FC and FR traversal results for aggregation indexing.
        """
        types_display = [t for t in entity_type_labels if not _is_technical_class_name(t, UniversalRagSystem._ontology_classes)]

        # Minimal doc for vocabulary entities
        if self.fr_traversal.is_minimal_doc_entity(raw_types):
            text = self.fr_traversal.format_minimal_document(
                entity_uri, entity_label, types_display, literals
            )
            return text, entity_label, entity_type_labels, {}

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
            schema_filter=_is_schema_predicate,
        )

        # Build target enrichments (type tags + attributes for FR targets)
        all_target_uris = set()
        for fr in fr_results:
            for uri, _lbl in fr["targets"]:
                all_target_uris.add(uri)
        for dp in (direct_preds or []):
            for uri, _lbl in dp["targets"]:
                all_target_uris.add(uri)

        from crm_rag.fr_traversal import build_target_enrichments
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

        # FR stats for aggregation index (piggyback on existing traversal)
        fr_stats = {
            "fc": fc,
            "fr_results": fr_results,
        }

        return text, entity_label, entity_type_labels, fr_stats

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
                output_dir = self._path('documents')

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
        output_dir = self._path('documents')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            logger.info(f"Cleared existing {output_dir} directory")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Entity documents will be saved to: {output_dir}/")

        # Create README for entity_documents directory
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
        tokens_per_min_limit = int(self.config.get("tokens_per_minute"))
        last_reset_time = time.time()

        # Check embedding cache for already processed entities
        cached_count = 0
        if self.embedding_cache:
            cache_stats = self.embedding_cache.get_stats()
            logger.info(f"Embedding cache: {cache_stats['count']} cached embeddings ({cache_stats['size_mb']} MB)")

        # Accumulate satellite URIs across all chunks
        all_satellite_uris = set()
        # Accumulate FR stats across all chunks for aggregation index
        all_fr_stats = []  # [(entity_uri, entity_label, fr_stats_dict), ...]

        # Pre-fetch image index (single SPARQL query, shared across all chunks)
        image_index = self.batch_sparql.build_image_index(self.dataset_config)

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

            chunk_literals = self.batch_sparql.batch_fetch_literals(chunk_uris)
            logger.info(f"    Literals: {len(chunk_literals)} entities")

            chunk_types = self.batch_sparql.batch_fetch_types(chunk_uris)
            logger.info(f"    Types: {len(chunk_types)} entities")

            # Collect all type URIs for batch label fetching
            chunk_type_uris = set()
            for types in chunk_types.values():
                chunk_type_uris.update(types)
            chunk_type_labels = self.batch_sparql.batch_fetch_type_labels(chunk_type_uris)
            logger.info(f"    Type labels: {len(chunk_type_labels)} types")

            # Build graph indexes for FR traversal
            fr_outgoing, fr_incoming, entity_labels_map, entity_types_map, chunk_raw_triples = \
                self._build_fr_graph_for_chunk(chunk_uris, chunk_types, chunk_literals)
            self._append_edges_parquet(chunk_raw_triples)
            logger.info(f"    FR graph built: {len(chunk_raw_triples)} raw triples")
            del chunk_raw_triples  # free immediately after writing to Parquet

            # Identify satellites for this chunk
            chunk_satellite_uris = set()
            chunk_parent_satellites = {}
            chunk_satellite_uris, chunk_parent_satellites = self._identify_satellites_from_prefetched(
                chunk_types, fr_incoming, entity_labels_map,
                all_literals=chunk_literals,
            )
            all_satellite_uris.update(chunk_satellite_uris)

            chunk_wikidata = self.batch_sparql.batch_fetch_wikidata_ids(chunk_uris)
            logger.info(f"    Wikidata IDs: {len(chunk_wikidata)} entities")

            # Phase B: Generate docs from pre-fetched data (zero SPARQL queries)
            # Filter out satellite entities
            doc_entities = [e for e in chunk_entities if e["entity"] not in chunk_satellite_uris]
            logger.info(f"  Phase B: Generating {len(doc_entities)} documents "
                        f"(skipping {len(chunk_satellite_uris)} satellites) (FR)...")
            chunk_docs = []  # List of (entity_uri, doc_text, metadata, cached_embedding)

            for entity in tqdm(doc_entities, desc=f"Chunk {chunk_num}", unit="entity"):
                entity_uri = entity["entity"]

                try:
                    literals = chunk_literals.get(entity_uri, {})
                    types = chunk_types.get(entity_uri, set())

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

                    doc_text, entity_label, entity_types, fr_stats = self._create_fr_document_from_prefetched(
                        entity_uri, entity_label, types, entity_type_labels,
                        literals, fr_outgoing, fr_incoming,
                        entity_labels_map, entity_types_map,
                        absorbed_lines=absorbed_lines,
                    )
                    if fr_stats:
                        all_fr_stats.append((entity_uri, entity_label, fr_stats))

                    # Determine primary entity type
                    primary_type = "Unknown"
                    human_readable_types = []
                    if entity_types:
                        human_readable_types = [
                            t for t in entity_types
                            if not _is_technical_class_name(t, UniversalRagSystem._ontology_classes)
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
            del fr_outgoing, fr_incoming, entity_labels_map, entity_types_map

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
            self.document_store.save_document_graph(self._path('graph_temp'))
            logger.info(f"  Chunk {chunk_num}/{total_chunks} complete, progress saved")

        if cached_count > 0:
            logger.info(f"Used {cached_count} cached embeddings")

        # Finalize incremental Parquet writes and build edges
        self._close_edges_parquet()
        logger.info("Creating document graph edges...")
        self._build_edges_from_parquet()

        # Chain thin documents into their richest neighbors (before FAISS/BM25 indexing)
        self._chain_thin_documents()

        # Rename temp file to final
        temp_path = self._path('graph_temp')
        final_path = self._path('graph')
        if os.path.exists(temp_path):
            os.replace(temp_path, final_path)

        # Build vector store with batched embedding requests
        logger.info("Building vector store...")
        self.build_vector_store_batched()

        # Build triples index from Parquet for query-time lookup
        self._load_triples_index()

        # Build aggregation index for pre-computed counts/rankings
        self._build_aggregation_index(all_fr_stats)

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

        vector_index_path = self._path('vector_dir')
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
        filtered = {k: v for k, v in raw.items() if not k.startswith('_')}

        # Expand each FC category to include both E-coded and human-readable names
        # so that matching works against doc.metadata["all_types"] which stores labels
        class_labels = cls._class_labels or {}
        local_to_label = {}
        for uri, label in class_labels.items():
            local_name = uri.split('/')[-1].split('#')[-1]
            local_to_label[local_name] = label

        expanded = {}
        for fc_name, class_list in filtered.items():
            expanded_set = set(class_list)
            for cls_name in class_list:
                label = local_to_label.get(cls_name)
                if label:
                    expanded_set.add(label)
            expanded[fc_name] = list(expanded_set)

        cls._fc_class_mapping = expanded
        total = sum(len(v) for v in cls._fc_class_mapping.values())
        logger.info(f"Loaded FC class mapping: {len(cls._fc_class_mapping)} categories, {total} classes "
                    f"(expanded with human-readable labels)")
        return cls._fc_class_mapping

    def _build_fc_type_index(self):
        """Load FC class mapping and build the FC type index on the document store."""
        fc_mapping = self._load_fc_class_mapping()
        if fc_mapping and self.document_store:
            self.document_store.build_fc_type_index(fc_mapping)

    def _analyze_query(self, question: str) -> 'QueryAnalysis':
        """Classify a user question using the LLM for query-type-aware retrieval.

        Returns a QueryAnalysis with query_type, primary categories (what the
        user wants returned), and context_categories (mentioned but not the
        answer type).  Falls back to SPECIFIC with no categories on failure.
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
            ctx_cats = parsed.get("context_categories", [])

            if qtype not in valid_types:
                qtype = "SPECIFIC"
            cats = [c for c in cats if c in valid_categories]
            ctx_cats = [c for c in ctx_cats if c in valid_categories and c not in cats]

            result = QueryAnalysis(query_type=qtype, categories=cats,
                                   context_categories=ctx_cats)
            logger.info(f"Query analysis: type={result.query_type}, "
                        f"categories={result.categories}, context={result.context_categories}")
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

    def normalize_scores(self, scores):
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            scores: numpy array of scores

        Returns:
            numpy array with normalized scores in [0, 1] range
        """
        values = np.array(scores)
        min_val = np.min(values)
        max_val = np.max(values)

        if max_val - min_val < 1e-10:
            return np.ones_like(values)

        return (values - min_val) / (max_val - min_val)

    def get_wikidata_for_entity(self, entity_uri):
        """Get Wikidata ID for an entity if available.

        Returns the cached wikidata_id from document metadata (indexed at build
        time).  If the entity is in the document store, its metadata is
        authoritative — a missing or None value means "no Wikidata link" and
        no SPARQL fallback is attempted.
        """
        if entity_uri in self.document_store.docs:
            return self.document_store.docs[entity_uri].metadata.get("wikidata_id")
        return None

    def fetch_wikidata_info(self, wikidata_id):
        """Fetch information from Wikidata for a given Q-ID."""
        return _fetch_wikidata_info(wikidata_id, self._http_session)


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
                # Mega-entity penalty: high-triple-count entities provide diluted context
                triples_count = len(self._triples_index.get(candidates[idx].id, []))
                if triples_count > RetrievalConfig.MEGA_ENTITY_TRIPLES_THRESHOLD:
                    combined_score -= RetrievalConfig.MEGA_ENTITY_PENALTY
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
                tc = len(self._triples_index.get(candidates[idx].id, []))
                mega_str = f", MEGA(-{RetrievalConfig.MEGA_ENTITY_PENALTY:.2f}, {tc} triples)" if tc > RetrievalConfig.MEGA_ENTITY_TRIPLES_THRESHOLD else ""
                logger.info(f"      Combined: {combined:.3f}{type_mod_str}{mega_str}")

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

    def _get_pagerank_candidates(self, categories, k):
        """Get top PageRank entities for target FC categories.

        Looks up pre-computed PageRank scores from the aggregation index
        and returns GraphDocument objects ranked by centrality.

        Args:
            categories: List of FC names (e.g. ["Actor", "Thing"]).
            k: Maximum number of candidates to return.

        Returns:
            List of (GraphDocument, pagerank_score) tuples.
        """
        pagerank = self._aggregation_index.get("pagerank", {})
        by_fc = pagerank.get("by_fc", {})
        if not by_fc:
            return []

        candidates = []
        seen = set()
        for fc in categories:
            for entry in by_fc.get(fc, []):
                uri = entry["uri"]
                if uri in seen:
                    continue
                seen.add(uri)
                doc = self.document_store.docs.get(uri)
                if doc:
                    candidates.append((doc, entry["score"]))

        # Sort by PageRank score descending and truncate
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def _type_filtered_channel(self, query, results_with_scores, initial_pool_size, query_analysis):
        """Run type-filtered FAISS+BM25 retrieval and merge into the main pool.

        Returns an updated results_with_scores list with type-matching documents
        injected, or the original list if no type filtering applies.
        """
        fc_doc_ids = getattr(self.document_store, '_fc_doc_ids', None)
        if not (query_analysis and query_analysis.categories and fc_doc_ids):
            return results_with_scores

        # Collect allowed doc IDs from target FC categories
        allowed_doc_ids = set()
        for fc_name in query_analysis.categories:
            allowed_doc_ids |= fc_doc_ids.get(fc_name, set())
        if not allowed_doc_ids:
            return results_with_scores

        logger.info(f"Type-filtered channel: {len(allowed_doc_ids)} docs in target FCs "
                    f"{query_analysis.categories}")

        # Run type-filtered FAISS + BM25
        typed_faiss = self.document_store.retrieve_faiss_typed(
            query, k=initial_pool_size,
            allowed_doc_ids=allowed_doc_ids,
            fetch_k=RetrievalConfig.TYPE_FILTERED_FETCH_K,
        )
        typed_bm25 = self.document_store.retrieve_bm25_typed(
            query, k=initial_pool_size,
            allowed_doc_ids=allowed_doc_ids,
        )

        # RRF-fuse typed results
        if typed_faiss and typed_bm25:
            typed_fused = self._rrf_fuse(typed_faiss, typed_bm25, pool_size=initial_pool_size)
        elif typed_faiss:
            typed_fused = typed_faiss
        else:
            typed_fused = typed_bm25

        # Fold PageRank into typed channel as a third signal (ENUM/AGG only)
        if (typed_fused
                and query_analysis.query_type in ("ENUMERATION", "AGGREGATION")
                and self._aggregation_index
                and "pagerank" in self._aggregation_index):
            pagerank_candidates = self._get_pagerank_candidates(
                query_analysis.categories, initial_pool_size
            )
            if pagerank_candidates:
                typed_fused = self._rrf_fuse(
                    typed_fused, pagerank_candidates,
                    pool_size=initial_pool_size,
                )
                logger.info(f"PageRank channel: {len(pagerank_candidates)} candidates "
                            f"fused into typed pool")

        if not typed_fused:
            logger.info("Type-filtered channel: no typed results found")
            return results_with_scores

        # Merge: reserve slots for type-matching docs
        fraction = (RetrievalConfig.TYPE_CHANNEL_POOL_FRACTION_SPECIFIC
                    if query_analysis.query_type == "SPECIFIC"
                    else RetrievalConfig.TYPE_CHANNEL_POOL_FRACTION)
        typed_slots = int(initial_pool_size * fraction)
        main_slots = initial_pool_size - typed_slots

        main_ids = {doc.id for doc, _ in results_with_scores}
        main_pool = results_with_scores[:main_slots]
        typed_new = [(doc, score) for doc, score in typed_fused if doc.id not in main_ids]
        typed_pool = typed_fused[:typed_slots]

        # Combine and deduplicate
        merged = {}
        for doc, score in main_pool:
            merged[doc.id] = (doc, score)
        for doc, score in typed_pool:
            if doc.id not in merged:
                merged[doc.id] = (doc, score)

        result = sorted(merged.values(), key=lambda x: x[1], reverse=True)[:initial_pool_size]
        logger.info(f"Type-filtered channel injected {len(typed_new)} new typed docs "
                    f"into pool (pool now {len(result)})")
        return result

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
        faiss_results = self.document_store.retrieve(query, k=initial_pool_size)

        if not faiss_results:
            logger.warning("No documents found in FAISS retrieval")
            return []

        # BM25 retrieval and RRF fusion
        bm25_results = self.document_store.retrieve_bm25(query, k=initial_pool_size)
        if bm25_results:
            results_with_scores = self._rrf_fuse(faiss_results, bm25_results, pool_size=initial_pool_size)
        else:
            results_with_scores = faiss_results

        # Type-filtered retrieval channel (FC-aware FAISS+BM25+PageRank)
        results_with_scores = self._type_filtered_channel(
            query, results_with_scores, initial_pool_size, query_analysis
        )

        # Pre-filter non-informative types from candidate pool
        informative = []
        non_informative = []
        for doc, score in results_with_scores:
            doc_type = doc.metadata.get('type', '')
            all_types = set(doc.metadata.get('all_types', []))
            is_non_informative = doc_type in RetrievalConfig.NON_INFORMATIVE_TYPES or bool(all_types & RetrievalConfig.NON_INFORMATIVE_TYPES)
            if is_non_informative:
                non_informative.append((doc, score))
            else:
                informative.append((doc, score))

        max_non_informative = max(1, int(len(results_with_scores) * RetrievalConfig.MAX_NON_INFORMATIVE_RATIO))
        if len(non_informative) > max_non_informative:
            non_informative.sort(key=lambda x: x[1], reverse=True)
            removed = len(non_informative) - max_non_informative
            non_informative = non_informative[:max_non_informative]
            logger.info(f"Pool filtering: removed {removed} non-informative entities, keeping {max_non_informative}")
        else:
            logger.info(f"Pool filtering: {len(non_informative)} non-informative entities within limit "
                        f"({max_non_informative} max of {len(results_with_scores)} total)")

        filtered_candidates = informative + non_informative
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)

        initial_docs = [doc for doc, _ in filtered_candidates]
        faiss_scores = np.array([score for _, score in filtered_candidates])

        # If we got fewer documents than requested, just return them
        if len(initial_docs) <= k:
            logger.info(f"Retrieved {len(initial_docs)} documents (less than k={k})")
            return initial_docs

        # Create a subgraph of the retrieved documents
        doc_ids = [doc.id for doc in initial_docs]

        # Create weighted adjacency matrix with virtual 2-hop edges through full RDF graph
        adjacency_matrix = self.document_store.create_adjacency_matrix(
            doc_ids,
            triples_index=getattr(self, '_triples_index', {}),
            weight_fn=_get_relationship_weight,
            max_hops=RetrievalConfig.MAX_ADJACENCY_HOPS,
        )

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

    def _build_aggregation_context(self, query_analysis, retrieved_docs):
        """Build pre-computed aggregation statistics context for the LLM prompt.

        Uses FR-based aggregation index with multi-hop relationship summaries.
        Returns a formatted string (capped at ~3000 chars) with entity type counts,
        FC counts, and per-category FR summaries (top sources and targets).
        Returns "" if no aggregation index is available.
        """
        if not self._aggregation_index:
            return ""

        # Primary categories drive PageRank labels; union with context for FR filtering
        primary = set(query_analysis.categories) if query_analysis.categories else set()
        ctx = set(query_analysis.context_categories) if query_analysis.context_categories else set()
        all_categories = primary | ctx

        lines = []

        # --- Compact entity type counts ---
        type_counts = self._aggregation_index.get("entity_type_counts", {})
        total_entities = self._aggregation_index.get("total_entities", 0)
        fc_counts = self._aggregation_index.get("fc_counts", {})
        if type_counts:
            lines.append(f"**Dataset overview**: {total_entities} entities")
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            type_parts = [f"{t}: {c}" for t, c in sorted_types]
            lines.append(f"Entity types: {', '.join(type_parts)}")

        # FC counts summary
        if fc_counts:
            fc_parts = [f"{fc}: {c}" for fc, c in sorted(fc_counts.items(), key=lambda x: x[1], reverse=True)]
            lines.append(f"Fundamental Categories: {', '.join(fc_parts)}")

        # PageRank top entities per PRIMARY FC only
        pagerank = self._aggregation_index.get("pagerank", {})
        by_fc_pr = pagerank.get("by_fc", {})
        if by_fc_pr and primary:
            for fc in sorted(primary):
                fc_entries = by_fc_pr.get(fc, [])
                if fc_entries:
                    top_labels = [e["label"] for e in fc_entries[:20]]
                    lines.append(f"Top {fc}s by graph centrality: {', '.join(top_labels)}")

        # Work-count ranking for Actor queries (traced from P108i → P14 chain)
        actor_work_counts = getattr(self, '_actor_work_counts', {})
        if actor_work_counts and "Actor" in primary:
            # Resolve actor URIs to labels via triples index
            triples_index = getattr(self, '_triples_index', {})
            sorted_actors = sorted(actor_work_counts.items(), key=lambda x: -x[1])[:30]
            labeled_actors = []
            for uri, count in sorted_actors:
                label = uri.split("/")[-1]
                for t in triples_index.get(uri, [])[:5]:
                    if t.get("subject") == uri and t.get("subject_label"):
                        label = t["subject_label"]
                        break
                    if t.get("object") == uri and t.get("object_label"):
                        label = t["object_label"]
                        break
                labeled_actors.append(f"{label} ({count})")
            lines.append(f"Top Actors by work count: {', '.join(labeled_actors)}")

        lines.append("")

        # --- FR summaries filtered by primary + context categories ---
        fr_summaries = self._aggregation_index.get("fr_summaries", {})
        if not fr_summaries:
            result = "\n".join(lines)
            if len(result) > 3000:
                result = result[:3000] + "\n...[truncated]"
            return result

        # Sort FRs by total_connections descending for consistent output
        sorted_frs = sorted(
            fr_summaries.items(),
            key=lambda x: x[1].get("total_connections", 0),
            reverse=True,
        )

        shown_frs = 0
        max_frs = 8  # Limit to avoid flooding the prompt

        for fr_id, summary in sorted_frs:
            if summary.get("total_connections", 0) == 0:
                continue

            domain_fc = summary.get("domain_fc", "")
            range_fc = summary.get("range_fc", "")

            # Show FR if its domain or range matches any category (primary or context)
            if all_categories and domain_fc not in all_categories and range_fc not in all_categories:
                continue

            if shown_frs >= max_frs:
                break
            shown_frs += 1

            fr_label = summary.get("label", fr_id)
            total = summary["total_connections"]
            lines.append(
                f"**{fr_label}** ({domain_fc} -> {range_fc}, "
                f"{total} connections, "
                f"{summary.get('unique_sources', 0)} sources, "
                f"{summary.get('unique_targets', 0)} targets):"
            )

            top_sources = summary.get("top_sources", [])
            if top_sources:
                source_parts = [f"{e['label']} ({e['count']})" for e in top_sources[:15]]
                lines.append(f"  Top {domain_fc}s: {', '.join(source_parts)}")

            top_targets = summary.get("top_targets", [])
            if top_targets:
                target_parts = [f"{e['label']} ({e['count']})" for e in top_targets[:15]]
                lines.append(f"  Top {range_fc}s: {', '.join(target_parts)}")

            lines.append("")

        # Cap total output
        result = "\n".join(lines)
        if len(result) > 3000:
            result = result[:3000] + "\n...[truncated]"
        return result

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

        # Time-span predicates for 1-hop enrichment
        TIME_SPAN_PREDICATES = {
            "http://www.cidoc-crm.org/cidoc-crm/P4_has_time-span",
            "http://www.cidoc-crm.org/cidoc-crm/P4i_is_time-span_of",
        }

        MAX_TRIPLES_PER_ENTITY = 15
        MAX_TOTAL_CHARS = 5000

        sorted_docs = sorted(retrieved_docs, key=_triples_type_priority)

        # Fair budget allocation: cap per-entity chars so mega-entities can't starve others
        entities_with_data = sum(1 for doc in sorted_docs if triples_index.get(doc.id))
        max_chars_per_entity = max(200, MAX_TOTAL_CHARS // max(entities_with_data, 1))

        # Build per-entity enrichment
        entity_sections = []
        total_chars = 0

        for doc in sorted_docs:
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

                priority = _predicate_priority(t, entity_uri, retrieved_uris)
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

                # 1-hop enrichment: replace time-span UUID with resolved dates
                if direction == "outgoing" and t.get("predicate") in TIME_SPAN_PREDICATES:
                    dates = _resolve_time_span(other_uri, triples_index)
                    if dates:
                        for date_key, date_val in dates.items():
                            date_line = f"  - {date_key}: {date_val}"
                            if date_line not in seen_lines:
                                seen_lines.add(date_line)
                                lines.append(date_line)
                        continue  # skip the raw time-span UUID line

                lines.append(line)

            if not lines:
                continue

            section = f"{entity_label}:\n" + "\n".join(lines)

            # Cap per-entity section to fair budget allocation
            if len(section) > max_chars_per_entity:
                section = section[:max_chars_per_entity] + "..."

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

    # ==================== Query Preparation ====================

    _EXCLUSION_PATTERNS = [
        r'\b(?:aside\s+from|other\s+than|besides|apart\s+from|excluding|except(?:\s+for)?)\s+([^,?.!]+)',
    ]

    _VAGUE_STOPWORDS = frozenset({
        'it', 'they', 'them', 'this', 'that', 'these', 'those',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have',
        'had', 'do', 'did', 'does', 'when', 'where', 'what', 'how',
        'and', 'or', 'with', 'which', 'other', 'more', 'its', 'of',
        'in', 'to', 'for', 'about', 'me', 'tell', 'happened', 'took',
        'place', 'there', 'any',
    })

    def _prepare_retrieval(self, question, chat_history, k, initial_pool_size, query_analysis):
        """Determine retrieval strategy and run retrieval.

        Handles topic-pivot detection, dual retrieval with chat context,
        and vague follow-up detection.

        Returns:
            List of retrieved GraphDocument objects.
        """
        # Detect topic-pivot / exclusion patterns (e.g. "aside from X", "other than X")
        excluded_entities = []
        is_pivot_query = False
        for pat in self._EXCLUSION_PATTERNS:
            match = re.search(pat, question, re.IGNORECASE)
            if match:
                is_pivot_query = True
                excluded_name = match.group(1).strip()
                for part in re.split(r'\s+(?:and|or)\s+', excluded_name):
                    part = part.strip().rstrip(',')
                    if part:
                        excluded_entities.append(part)
                break

        clean_query = question
        if is_pivot_query:
            logger.info(f"Topic pivot detected — excluded entities: {excluded_entities}")
            for pat in self._EXCLUSION_PATTERNS:
                clean_query = re.sub(pat, '', clean_query, flags=re.IGNORECASE)
            for entity in excluded_entities:
                clean_query = re.sub(r'\b' + re.escape(entity) + r'\b', '', clean_query, flags=re.IGNORECASE)
            clean_query = re.sub(r'\s+', ' ', clean_query).strip().rstrip(',').strip()
            if not clean_query:
                clean_query = question
            logger.info(f"Cleaned pivot query: '{clean_query}'")

        retrieval_kwargs = dict(k=k, initial_pool_size=initial_pool_size, query_analysis=query_analysis)

        if not chat_history:
            return self.retrieve(question, **retrieval_kwargs)

        if is_pivot_query:
            logger.info(f"Pivot query: using cleaned raw retrieval only ('{clean_query}')")
            return self.retrieve(clean_query, **retrieval_kwargs)

        # Dual retrieval: contextualized + raw, then interleaved merge
        prev_user_msgs = [m["content"] for m in chat_history[:-1] if m["role"] == "user"]
        contextualized_query = f"{prev_user_msgs[-1]} {question}" if prev_user_msgs else question
        logger.info(f"Dual retrieval — contextualized: '{contextualized_query[:200]}'")
        logger.info(f"Dual retrieval — raw: '{question}'")

        ctx_docs = self.retrieve(contextualized_query, **retrieval_kwargs)

        # Detect vague follow-ups: short questions dominated by pronouns/stopwords
        q_words = set(re.findall(r'\b[a-z]+\b', question.lower()))
        content_words = q_words - self._VAGUE_STOPWORDS
        is_vague = len(content_words) == 0 and len(question.split()) <= 10
        if is_vague:
            logger.info(f"Vague follow-up detected, skipping raw retrieval: '{question}'")
            return ctx_docs[:k]

        raw_docs = self.retrieve(question, **retrieval_kwargs)

        # Interleaved merge: alternate ctx and raw for fair representation
        seen_uris = set()
        merged = []
        for i in range(max(len(ctx_docs), len(raw_docs))):
            if i < len(ctx_docs) and ctx_docs[i].id not in seen_uris:
                seen_uris.add(ctx_docs[i].id)
                merged.append(ctx_docs[i])
            if i < len(raw_docs) and raw_docs[i].id not in seen_uris:
                seen_uris.add(raw_docs[i].id)
                merged.append(raw_docs[i])
        merged = merged[:k]
        logger.info(f"Dual retrieval merged: {len(merged)} unique docs "
                    f"(ctx={len(ctx_docs)}, raw={len(raw_docs)})")
        return merged

    # ==================== Answer Generation ====================

    def _generate_answer(self, question, retrieved_docs, query_analysis,
                         include_wikidata, chat_history):
        """Build context, call LLM, assemble sources.

        Returns:
            Dict with "answer" and "sources" keys.
        """
        # Build document context and collect Wikidata candidates
        context = ""
        entities_with_wikidata = []

        for doc in retrieved_docs:
            entity_uri = doc.id
            entity_label = doc.metadata.get("label", entity_uri.split('/')[-1])

            context += f"Entity: {entity_label}\n"
            doc_text = doc.text
            if len(doc_text) > RetrievalConfig.MAX_DOC_CHARS:
                doc_text = doc_text[:RetrievalConfig.MAX_DOC_CHARS] + "...[truncated]"
            context += doc_text + "\n\n"

            if include_wikidata:
                wikidata_id = self.get_wikidata_for_entity(entity_uri)
                if wikidata_id:
                    entities_with_wikidata.append({
                        "entity_uri": entity_uri,
                        "entity_label": entity_label,
                        "wikidata_id": wikidata_id
                    })

        # Add structured triples enrichment
        triples_enrichment = self._build_triples_enrichment(retrieved_docs)
        if triples_enrichment:
            context += "\n## Structured Relationships\n\n" + triples_enrichment + "\n"

        # Add aggregation statistics for ENUMERATION/AGGREGATION queries
        if query_analysis.query_type in ("AGGREGATION", "ENUMERATION"):
            agg_context = self._build_aggregation_context(query_analysis, retrieved_docs)
            if agg_context:
                context += "\n## Dataset Statistics (pre-computed from full knowledge graph)\n\n" + agg_context + "\n"

        # Fetch Wikidata context for top 2 entities
        wikidata_context = ""
        if include_wikidata and entities_with_wikidata:
            wikidata_context += "\nWikidata Context:\n"
            for entity_info in entities_with_wikidata[:2]:
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

        # Build system prompt
        system_prompt = self.get_cidoc_system_prompt()
        if query_analysis.query_type == "ENUMERATION":
            system_prompt += "\nList ALL matching entities from the retrieved information. Be comprehensive."
        elif query_analysis.query_type == "AGGREGATION":
            system_prompt += (
                "\nCount or rank entities based on the retrieved information. "
                "A 'Dataset Statistics' section with pre-computed counts and rankings "
                "from the full knowledge graph is included when available — use these "
                "statistics for accurate counts. State if the count may be incomplete."
            )
        if include_wikidata and wikidata_context:
            system_prompt += "\n\nI have also provided Wikidata information for some entities. When appropriate, incorporate this Wikidata information to enhance your answer with additional context, especially for factual details not present in the RDF data."

        # Build user prompt
        prompt = ""
        if chat_history:
            prompt += "Conversation so far:\n"
            for msg in chat_history[-6:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "\n"

        prompt += f"Retrieved information:\n{context}\n"
        if include_wikidata and wikidata_context:
            prompt += f"{wikidata_context}\n"
        prompt += f"\nQuestion: {question}\n\nAnswer directly using only the retrieved information above. Use entity names, not codes.\n"

        # Generate answer
        answer = self.llm_provider.generate(system_prompt, prompt)

        # Build sources
        sources = self._build_sources(retrieved_docs, entities_with_wikidata)

        return {"answer": answer, "sources": sources}

    def _build_sources(self, retrieved_docs, entities_with_wikidata):
        """Build source entries from retrieved docs, enriched with images and Wikidata."""
        sources = []
        entities_with_local_images = set()

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

            if local_images:
                entities_with_local_images.add(entity_uri)
                source_entry["images"] = [{"url": img_url, "source": "dataset"} for img_url in local_images]
                logger.info(f"Using {len(local_images)} local image(s) for {entity_label}")

            sources.append(source_entry)

        # Enrich with Wikidata IDs and images
        source_by_uri = {s["entity_uri"]: s for s in sources}
        for entity_info in entities_with_wikidata:
            entity_uri = entity_info["entity_uri"]
            existing = source_by_uri.get(entity_uri)
            if not existing:
                continue

            existing["wikidata_id"] = entity_info["wikidata_id"]
            existing["wikidata_url"] = f"https://www.wikidata.org/wiki/{entity_info['wikidata_id']}"

            if entity_uri in entities_with_local_images:
                logger.debug(f"Skipping Wikidata image for {entity_info['entity_label']} - local images available")
                continue

            wikidata_data = self.fetch_wikidata_info(entity_info["wikidata_id"])
            if wikidata_data and "properties" in wikidata_data:
                image_value = wikidata_data["properties"].get("image")
                if image_value:
                    try:
                        image_filename = image_value[0] if isinstance(image_value, list) else image_value
                        if image_filename:
                            image_filename_str = image_filename.decode('utf-8') if isinstance(image_filename, bytes) else str(image_filename)
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

        return sources

    # ==================== Main Entry Point ====================

    def answer_question(self, question, include_wikidata=True, chat_history=None):
        """Answer a question using the RAG pipeline.

        Orchestrates three phases:
        1. Query analysis + retrieval strategy (_prepare_retrieval)
        2. Hybrid retrieval pipeline (retrieve)
        3. Context assembly + LLM generation (_generate_answer)

        Args:
            question: The user's question
            include_wikidata: Whether to fetch Wikidata context
            chat_history: Optional list of {"role": "user"|"assistant", "content": str} dicts
        """
        if not question or not question.strip():
            return {"answer": "Please provide a question.", "sources": []}

        logger.info(f"Answering question: '{question}'")

        # Phase 1: Query analysis
        query_analysis = self._analyze_query(question)
        k_map = {
            "SPECIFIC": RetrievalConfig.SPECIFIC_K,
            "ENUMERATION": RetrievalConfig.ENUMERATION_K,
            "AGGREGATION": RetrievalConfig.AGGREGATION_K,
        }
        k = k_map.get(query_analysis.query_type, RetrievalConfig.DEFAULT_RETRIEVAL_K)
        initial_pool_size = k * RetrievalConfig.POOL_MULTIPLIER
        logger.info(f"Dynamic retrieval: type={query_analysis.query_type}, k={k}, pool={initial_pool_size}")

        # Phase 2: Retrieval (handles pivot detection, dual retrieval, vague follow-ups)
        retrieved_docs = self._prepare_retrieval(
            question, chat_history, k, initial_pool_size, query_analysis
        )

        if not retrieved_docs:
            return {"answer": "I couldn't find relevant information to answer your question.", "sources": []}

        # Phase 3: Context assembly + LLM answer generation
        return self._generate_answer(
            question, retrieved_docs, query_analysis, include_wikidata, chat_history
        )