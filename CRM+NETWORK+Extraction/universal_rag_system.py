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
import types
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
from langchain_openai import ChatOpenAI

# Third-party data fetching
import requests

# Local imports
from graph_document_store import GraphDocumentStore
from llm_providers import get_llm_provider, BaseLLMProvider
from extract_ontology_labels import run_extraction

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


class UniversalRagSystem:
    """Universal RAG system with graph-based document retrieval"""

    # Class-level cache for property labels
    _property_labels = None
    _extraction_attempted = False  # Track if we've tried extraction to avoid infinite loops
    _missing_properties = set()  # Track properties that couldn't be found

    def __init__(self, endpoint_url, config=None):
        """
        Initialize the universal RAG system.

        Args:
            endpoint_url: SPARQL endpoint URL
            config: Configuration dictionary for LLM provider
        """
        self.endpoint_url = endpoint_url
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)

        # Initialize configuration
        self.config = config or {}

        # Initialize LLM provider
        provider_name = self.config.get("llm_provider", "openai")
        try:
            self.llm_provider = get_llm_provider(provider_name, self.config)
        except Exception as e:
            logger.error(f"Error initializing LLM provider: {str(e)}")
            raise

        # Initialize document store
        self.document_store = None

        # Load property labels from ontology extraction (cached at class level)
        if UniversalRagSystem._property_labels is None:
            UniversalRagSystem._property_labels = self._load_property_labels()

    def _load_property_labels(self, force_extract=False):
        """
        Load property labels from JSON file generated from ontologies.
        Automatically extracts labels from ontology files if JSON doesn't exist.

        Args:
            force_extract: If True, force re-extraction even if JSON exists

        Returns:
            dict: Property labels mapping
        """
        labels_file = 'property_labels.json'
        ontology_dir = 'ontology'

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
        
    @property
    def embeddings(self):
        """
        Return an embedding object compatible with FAISS and the rest of the code.
        This property maintains backward compatibility with existing code.
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
                return [self.provider.get_embeddings(text) for text in texts]
        
        return EmbeddingFunction(self.llm_provider)
    
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
        
        # Test connection
        if not self.test_connection():
            logger.error("Failed to connect to SPARQL endpoint")
            return False
        
        # Initialize document store
        self.document_store = GraphDocumentStore(self.embeddings)
        
        # Check if saved data exists
        doc_graph_path = 'document_graph.pkl'
        vector_index_path = 'vector_index/index.faiss'
        
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
            
            # Add graph document load method
            if not hasattr(self.document_store, 'load_document_graph'):
                # Define the method if it doesn't exist
                def load_document_graph(self, path='document_graph.pkl'):
                    """Load document graph from disk"""
                    if os.path.exists(path):
                        try:
                            with open(path, 'rb') as f:
                                self.docs = pickle.load(f)
                            logger.info(f"Document graph loaded from {path} with {len(self.docs)} documents")
                            return True
                        except Exception as e:
                            logger.error(f"Error loading document graph: {str(e)}")
                            return False
                    return False
                    
                # Add method to class
                self.document_store.load_document_graph = types.MethodType(load_document_graph, self.document_store)
            
            # Try to load document graph
            graph_loaded = self.document_store.load_document_graph(doc_graph_path)
            
            # Try to load vector store
            vector_loaded = False
            try:
                self.document_store.vector_store = FAISS.load_local(
                    'vector_index', 
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
        
        # Save the document graph
        if not hasattr(self.document_store, 'save_document_graph'):
            # Define the method if it doesn't exist
            def save_document_graph(self, path='document_graph.pkl'):
                """Save document graph to disk"""
                try:
                    with open(path, 'wb') as f:
                        pickle.dump(self.docs, f)
                    logger.info(f"Document graph saved to {path}")
                    return True
                except Exception as e:
                    logger.error(f"Error saving document graph: {str(e)}")
                    return False

            # Add method to class
            self.document_store.save_document_graph = types.MethodType(save_document_graph, self.document_store)
        
        # Save document graph
        self.document_store.save_document_graph(doc_graph_path)
        vector_index_path = 'vector_index'
        os.makedirs(vector_index_path, exist_ok=True)

        # Save the vector store
        if self.document_store.vector_store:
            self.document_store.vector_store.save_local(vector_index_path)
            logger.info(f"Vector store saved to {vector_index_path}")
        return True


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
        Check if a class name is a technical CIDOC-CRM identifier that should be filtered
        from natural language output.

        Technical patterns:
        - E\d+_Name (e.g., E22_Man-Made_Object, E53_Place)
        - D\d+_Name (e.g., D1_Digital_Object from CRMdig)
        - IC\d+_Name (e.g., IC9_Representation from VIR)
        - F\d+_Name (e.g., from FRBRoo)

        Args:
            class_name: The class name to check

        Returns:
            bool: True if it's a technical identifier, False if it's human-readable
        """
        # Pattern for technical CIDOC-CRM and extension class names
        # Matches: E22_..., D1_..., IC9_..., F38_..., etc.
        technical_pattern = r'^[A-Z]+\d+[a-z]?_'
        return bool(re.match(technical_pattern, class_name))

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

    def get_entity_context(self, entity_uri, depth=RetrievalConfig.ENTITY_CONTEXT_DEPTH):
        """Get entity context by traversing the graph bidirectionally"""

        context_statements = []
        visited = set()
        
        def traverse(uri, current_depth=0, direction="both"):
            if uri in visited or current_depth > depth:
                return

            visited.add(uri)

            # Get entity label using the improved label retrieval method
            entity_label = self.get_entity_label(uri)
            
            # Get outgoing relationships if direction is "both" or "outgoing"
            if direction in ["both", "outgoing"]:
                outgoing_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?pred ?predLabel ?obj ?objLabel WHERE {{
                    <{uri}> ?pred ?obj .
                    OPTIONAL {{ ?pred rdfs:label ?predLabel }}
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
                        pred_label = result.get("predLabel", {}).get("value", pred.split('/')[-1])
                        obj_label = result.get("objLabel", {}).get("value")
                        if not obj_label:
                            obj_label = self.get_entity_label(obj)

                        # Filter out self-referential relationships
                        # Skip if same URI or same label (redundant statements)
                        if uri == obj or (entity_label and obj_label and entity_label.lower() == obj_label.lower()):
                            continue

                        # Create natural language statement
                        statement = self.process_cidoc_relationship(
                            uri, pred, obj, entity_label, obj_label
                        )

                        context_statements.append(statement)

                        # Recursively traverse outgoing relationships
                        if current_depth < depth:
                            traverse(obj, current_depth + 1, "outgoing")
                except Exception as e:
                    logger.error(f"Error traversing outgoing relationships: {str(e)}")
            
            # Get incoming relationships if direction is "both" or "incoming"
            if direction in ["both", "incoming"]:
                incoming_query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?subj ?subjLabel ?pred ?predLabel WHERE {{
                    ?subj ?pred <{uri}> .
                    OPTIONAL {{ ?subj rdfs:label ?subjLabel }}
                    OPTIONAL {{ ?pred rdfs:label ?predLabel }}
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
                        pred_label = result.get("predLabel", {}).get("value", pred.split('/')[-1])

                        # Filter out self-referential relationships
                        # Skip if same URI or same label (redundant statements)
                        if subj == uri or (subj_label and entity_label and subj_label.lower() == entity_label.lower()):
                            continue

                        # Create natural language statement
                        statement = self.process_cidoc_relationship(
                            subj, pred, uri, subj_label, entity_label
                        )

                        context_statements.append(statement)

                        # Recursively traverse incoming relationships
                        if current_depth < depth:
                            traverse(subj, current_depth + 1, "incoming")
                except Exception as e:
                    logger.error(f"Error traversing incoming relationships: {str(e)}")
        
        # Start traversal
        traverse(entity_uri)
        
        # Return unique statements
        return list(set(context_statements))

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

            # Get entity type
            type_query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

            SELECT ?type ?typeLabel WHERE {{
                <{entity_uri}> rdf:type ?type .
                OPTIONAL {{ ?type rdfs:label ?typeLabel }}
                FILTER(STRSTARTS(STR(?type), "http://"))
            }}
            """

            entity_types = []
            try:
                self.sparql.setQuery(type_query)
                type_results = self.sparql.query().convert()

                for result in type_results["results"]["bindings"]:
                    type_uri = result["type"]["value"]
                    type_label = result.get("typeLabel", {}).get("value", type_uri.split('/')[-1])
                    entity_types.append(type_label)
            except Exception as e:
                logger.warning(f"Error getting entity types for {entity_uri}: {str(e)}")

            # Get relationships and convert to natural language
            try:
                context_statements = self.get_entity_context(entity_uri, depth=RetrievalConfig.ENTITY_CONTEXT_DEPTH)
            except Exception as e:
                logger.warning(f"Error getting entity context for {entity_uri}: {str(e)}")
                context_statements = []

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

            return text, entity_label, entity_types
        except Exception as e:
            logger.error(f"Error creating enhanced document for {entity_uri}: {str(e)}")
            # Return minimal document to prevent complete failure
            return f"Entity: {entity_uri}", entity_uri, []

    def save_entity_document(self, entity_uri, document_text, entity_label, output_dir="entity_documents"):
        """Save entity document to disk for transparency and reuse"""

        try:
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

    def process_rdf_data(self):
        """Process RDF data into graph documents with enhanced CIDOC-CRM understanding"""
        logger.info("Processing RDF data with enhanced CIDOC-CRM understanding...")

        # Get all entities
        entities = self.get_all_entities()
        total_entities = len(entities)
        logger.info(f"Found {total_entities} entities")

        # Clear entity_documents directory if it exists
        output_dir = "entity_documents"
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
        
        # Global rate limit tracking
        global_token_count = 0
        tokens_per_min_limit = RetrievalConfig.TOKENS_PER_MINUTE_LIMIT  # Set slightly below the actual limit of 1M
        last_reset_time = time.time()
        
        # First pass: create document nodes with enhanced content
        logger.info("Creating enhanced document nodes...")
        batch_size = RetrievalConfig.DEFAULT_BATCH_SIZE  # Process in reasonable batches for progress tracking
        
        for i in range(0, total_entities, batch_size):
            batch = entities[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_entities + batch_size - 1)//batch_size}")
            
            # Process batch
            for entity in tqdm(batch, desc=f"Batch {i//batch_size + 1}", unit="entity"):
                entity_uri = entity["entity"]
                
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
                
                # Create enhanced document with CIDOC-CRM aware natural language
                try:
                    doc_text, entity_label, entity_types = self.create_enhanced_document(entity_uri)

                    # Save document to disk for transparency and reuse
                    self.save_entity_document(entity_uri, doc_text, entity_label)

                    # Estimate token count - very rough estimate
                    # (1 token ≈ 4 chars in English on average)
                    estimated_tokens = len(doc_text) / 4
                    global_token_count += estimated_tokens

                    # Determine primary entity type (filter out technical CIDOC-CRM class names)
                    primary_type = "Unknown"
                    if entity_types:
                        # Get human-readable types only
                        human_readable_types = [
                            t for t in entity_types
                            if not self.is_technical_class_name(t)
                        ]
                        # Use first human-readable type, or fall back to "Unknown"
                        primary_type = human_readable_types[0] if human_readable_types else "Entity"

                    # Add to document store
                    self.document_store.add_document(
                        entity_uri,
                        doc_text,
                        {
                            "label": entity_label,
                            "type": primary_type,
                            "uri": entity_uri,
                            "all_types": entity_types  # Keep all types for debugging if needed
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing entity {entity_uri}: {str(e)}")
                    # Continue with next entity
                    continue
            
            # Save progress after each batch
            self.document_store.save_document_graph('document_graph_temp.pkl')
            
            # Pause for 2 seconds after each batch of 50
            logger.info("Completed batch of 50 documents, pausing for 2 seconds...")
            time.sleep(2)
        
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
        if os.path.exists('document_graph_temp.pkl'):
            os.replace('document_graph_temp.pkl', 'document_graph.pkl')
        
        # Build vector store with batched embedding requests
        logger.info("Building vector store...")
        self.build_vector_store_batched()
        
        logger.info("RDF data processing complete with enhanced CIDOC-CRM understanding")

    def build_vector_store_batched(self, batch_size=RetrievalConfig.DEFAULT_BATCH_SIZE):
        """Build vector store with batched embedding requests to avoid rate limits"""
        
        vector_index_path = 'vector_index'
        os.makedirs(vector_index_path, exist_ok=True)
        
        # Prepare documents for FAISS
        docs_for_faiss = []
        for doc_id, graph_doc in self.document_store.docs.items():
            doc = Document(
                page_content=graph_doc.text,
                metadata={**graph_doc.metadata, "doc_id": doc_id}
            )
            docs_for_faiss.append(doc)
        
        # Process in batches
        total_docs = len(docs_for_faiss)
        logger.info(f"Building vector store with {total_docs} documents in batches of {batch_size}")
        
        # Global rate limit tracking
        global_token_count = 0
        tokens_per_min_limit = RetrievalConfig.TOKENS_PER_MINUTE_LIMIT
        last_reset_time = time.time()
        
        # Process batches
        from langchain_community.vectorstores import FAISS
        vector_store = None
        
        for i in range(0, total_docs, batch_size):
            batch = docs_for_faiss[i:i+batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
            
            # Rate limit check
            current_time = time.time()
            if current_time - last_reset_time >= 60:
                global_token_count = 0
                last_reset_time = current_time
            
            # Skip if we're approaching the limit
            if global_token_count > tokens_per_min_limit:
                wait_time = 60 - (current_time - last_reset_time) + 1
                if wait_time > 0:
                    logger.info(f"Approaching rate limit. Waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    global_token_count = 0
                    last_reset_time = time.time()
            
            try:
                # Create or update vector store
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    vector_store.add_documents(batch)
                
                # Estimate token count (very rough)
                batch_text = " ".join([doc.page_content for doc in batch])
                estimated_tokens = len(batch_text) / 4
                global_token_count += estimated_tokens
                
                # Save progress after each batch
                vector_store.save_local(vector_index_path)
                logger.info(f"Saved progress after batch {i//batch_size + 1}")
                
                # Pause for 2 seconds after each batch
                logger.info("Completed batch of documents, pausing for 2 seconds...")
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing embedding batch: {str(e)}")
                
                if "rate_limit_exceeded" in str(e):
                    # If we hit a rate limit, wait longer
                    wait_time = 60
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    # Retry this batch
                    i -= batch_size
                
                # Continue with next batch otherwise
        
        # Store the final vector store
        self.document_store.vector_store = vector_store
        logger.info(f"Vector store built successfully with {total_docs} documents")

    def generate_sparql_query(self, question):
            """Generate a SPARQL query based on the question"""
            
            system_prompt = """You are an expert in converting natural language questions to SPARQL queries for CIDOC-CRM data.
            
            The data uses these key CIDOC-CRM classes:
            - E53_Place for locations
            - E18_Physical_Thing for physical objects
            - E21_Person for people
            - E55_Type for types/categories
            - E36_Visual_Item for visual representations
            - E41_Appellation for names (use labels instead)
            
            And these key properties:
            - P89_falls_within for spatial containment
            - P55_has_current_location for current location
            - P168_is_approximated_by for coordinates
            - P2_has_type for indicating categories
            - P1_is_identified_by for names
            - rdfs:label for names/labels
            - K24_portray for portray visual items
            
            Generate a SPARQL query that will answer the question.
            - Use PREFIX statements for common namespaces
            - Return relevant labels for all URIs
            - Only return the SPARQL query, no explanations
            - Do not include any markdown formatting or code blocks (no backticks)
            """
            
            prompt = f"""Generate a SPARQL query for the following question about CIDOC-CRM data:
            
            {question}
            """
            
            sparql_query = self.llm_provider.generate(system_prompt, prompt)
            
            # Remove any markdown code formatting (backticks)
            sparql_query = sparql_query.replace('```sparql', '').replace('```', '')
            
            return sparql_query

    def answer_with_direct_query(self, question):
        """Try to answer directly with a SPARQL query"""
        
        # Generate SPARQL query
        sparql_query = self.generate_sparql_query(question)
        
        try:
            # Execute the query
            self.sparql.setQuery(sparql_query)
            results = self.sparql.query().convert()
            
            # If we got results, use them
            if results["results"]["bindings"]:
                return {
                    "direct_answer": True,
                    "results": results["results"]["bindings"],
                    "query": sparql_query
                }
            
            # Otherwise, fall back to RAG approach
            return None
        except Exception as e:
            logger.error(f"Error executing generated SPARQL query: {str(e)}")
            return None

    def cidoc_aware_retrieval(self, query, k=20):
        """Enhanced retrieval using CIDOC-CRM aware scoring"""
        
        # Initial vector search
        vector_results = self.document_store.retrieve(query, k=k*2)
        
        if not vector_results:
            return []
        
        # Get entity URIs from results
        entity_uris = [doc.id for doc in vector_results]
        
        # Properly escape URIs for SPARQL query
        escaped_uris = ['<' + uri.replace('>', '\\>').replace('<', '\\<') + '>' for uri in entity_uris]
        
        # Define relationship importance scores
        relationship_weights = {
            "http://www.cidoc-crm.org/cidoc-crm/P89_falls_within": 0.9,  # High weight for spatial containment
            "http://www.cidoc-crm.org/cidoc-crm/P55_has_current_location": 0.9,  # High weight for location
            "http://www.cidoc-crm.org/cidoc-crm/P56_bears_feature": 0.8,  # Important for physical features
            "http://www.cidoc-crm.org/cidoc-crm/P46_is_composed_of": 0.8,  # Important for part-whole
            "http://www.cidoc-crm.org/cidoc-crm/P108i_was_produced_by": 0.7,  # Important for creation
            "http://w3id.org/vir#K24_portray": 0.7,  # Important for visual representation
            "http://www.cidoc-crm.org/cidoc-crm/P2_has_type": 0.6  # Moderate for type information
        }
        
        # Create a graph representation
        G = nx.DiGraph()
        
        # Add nodes (entities) to the graph
        for doc in vector_results:
            G.add_node(doc.id, score=0.0, label=doc.metadata.get("label", ""))
        
        # Get relationships between entities - fix the VALUES clause
        relationships_query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?subject ?predicate ?object WHERE {{
            ?subject ?predicate ?object .
            VALUES ?subject {{ {' '.join(escaped_uris)} }}
            VALUES ?object {{ {' '.join(escaped_uris)} }}
        }}
        """
        
        try:
            self.sparql.setQuery(relationships_query)
            results = self.sparql.query().convert()
            
            # Add edges with weights based on relationship type
            for result in results["results"]["bindings"]:
                subject = result["subject"]["value"]
                predicate = result["predicate"]["value"]
                object_uri = result["object"]["value"]
                
                # Get weight for this relationship type
                weight = relationship_weights.get(predicate, 0.5)  # Default weight for unspecified relationships
                
                G.add_edge(subject, object_uri, weight=weight, predicate=predicate)
        except Exception as e:
            logger.error(f"Error getting relationships: {str(e)}")
            logger.error(f"Problematic query: {relationships_query}")
                
        # Re-rank documents by combined vector similarity and graph centrality
        ranked_docs = []
        
        for i, doc in enumerate(vector_results):
            # Vector score (inversely proportional to rank)
            vector_score = (len(vector_results) - i) / len(vector_results)
            
            # Graph score (from PageRank)
            graph_score = G.nodes.get(doc.id, {}).get("score", 0.0)
            
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

    def hybrid_answer_question(self, question):
        """Hybrid approach that tries direct querying first, then falls back to RAG"""
        
        # Try direct querying first
        direct_results = self.answer_with_direct_query(question)
        
        if direct_results:
            # Convert SPARQL results to natural language
            
            # Format results for LLM
            formatted_results = "SPARQL Query Results:\n"
            for i, result in enumerate(direct_results["results"]):
                formatted_results += f"Result {i+1}:\n"
                for var, value in result.items():
                    formatted_results += f"  {var}: {value['value']}\n"
            
            system_prompt = """You are an expert in CIDOC-CRM who can convert SPARQL query results to natural language answers.
            
            Given the results of a SPARQL query, provide a clear, concise answer to the original question.
            - Translate URIs and technical terminology into plain language
            - Focus only on answering the question with the provided data
            - If the data seems insufficient, say so
            """
            
            prompt = f"""Original question: {question}

    {formatted_results}

    Please provide a clear natural language answer based on these results."""
            
            llm = ChatOpenAI(
                model=self.openai_model,
                temperature=self.temperature,
                openai_api_key=self.openai_api_key
            )
            
            response = llm.invoke(system_prompt + "\n\n" + prompt)
            
            return {
                "answer": response.content,
                "query_type": "direct_sparql",
                "sources": [{"type": "direct_query", "query": "Direct SPARQL query used for answer"}]
            }
        
        # Fall back to RAG approach
        logger.info("Direct querying failed. Falling back to RAG approach.")
        rag_response = self.answer_question(question)
        rag_response["query_type"] = "rag"
        
        return rag_response
    
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
        SELECT DISTINCT ?entity ?property ?value
        WHERE {
            ?entity ?property ?value .
            FILTER(isLiteral(?value))
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

            # Convert to list format with labels
            entities = []
            for entity_uri, literals in entity_map.items():
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

            logger.info(f"Retrieved {len(entities)} entities with literals")
            return entities
        except Exception as e:
            logger.error(f"Error fetching entities: {str(e)}")
            return []
    
    def get_entity_details(self, entity_uri):
        """Get details about an entity"""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?predicate ?predicateLabel ?object ?objectLabel WHERE {{
            <{entity_uri}> ?predicate ?object .
            OPTIONAL {{ ?predicate rdfs:label ?predicateLabel }}
            OPTIONAL {{ ?object rdfs:label ?objectLabel }}
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            details = []
            for result in results["results"]["bindings"]:
                detail = {
                    "predicate": result["predicate"]["value"],
                    "object": result["object"]["value"]
                }
                if "predicateLabel" in result:
                    detail["predicateLabel"] = result["predicateLabel"]["value"]
                if "objectLabel" in result:
                    detail["objectLabel"] = result["objectLabel"]["value"]
                details.append(detail)
                
            return details
        except Exception as e:
            logger.error(f"Error fetching entity details: {str(e)}")
            return []
    
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

    def extract_entities_from_query(self, query):
        """Extract entity URIs mentioned in the query for relationship-aware retrieval"""
        
        # Look for potential entity names
        potential_entities = []
        
        # Capitalized phrases
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', query)
        potential_entities.extend(cap_phrases)
        
        # Words surrounded by quotes
        quoted = re.findall(r'"([^"]+)"', query)
        potential_entities.extend(quoted)
        
        # Match potential entities against known entity labels
        matched_entities = []
        
        for potential in potential_entities:
            potential_lower = potential.lower().strip()
            
            # Skip very short potential entities
            if len(potential_lower) < 3:
                continue
                
            # Search entity labels for matches
            for doc_id, doc in self.document_store.docs.items():
                label = doc.metadata.get("label", "").lower()
                
                # Check for substantial overlap
                if potential_lower in label or label in potential_lower:
                    # Calculate token overlap
                    potential_tokens = set(potential_lower.split())
                    label_tokens = set(label.split())
                    overlap = len(potential_tokens & label_tokens) / max(len(potential_tokens), len(label_tokens))
                    
                    if overlap > 0.5:  # Require significant overlap
                        matched_entities.append(doc_id)
        
        return matched_entities

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

    def calculate_relationship_scores(self, entity_uris, damping=RetrievalConfig.PAGERANK_DAMPING, iterations=RetrievalConfig.PAGERANK_ITERATIONS):
        """Calculate personalized PageRank scores for all entities in the graph"""
        
        # Get full list of documents
        doc_ids = list(self.document_store.docs.keys())
        n = len(doc_ids)
        
        # Create mapping from doc_id to index
        id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        
        # Build adjacency matrix
        adjacency = np.zeros((n, n))
        
        # Fill with connections from the document graph
        for i, doc_id in enumerate(doc_ids):
            if doc_id not in self.document_store.docs:
                continue
                
            doc = self.document_store.docs[doc_id]
            
            # Add edges from neighbors
            for neighbor in doc.neighbors:
                neighbor_id = neighbor["doc_id"]
                if neighbor_id in id_to_idx:
                    j = id_to_idx[neighbor_id]
                    # Use edge weight if available
                    weight = neighbor.get("weight", 1.0)
                    adjacency[i, j] = weight
        
        # Normalize adjacency matrix
        row_sums = adjacency.sum(axis=1)
        # Handle rows with all zeros to avoid division by zero
        row_sums[row_sums == 0] = 1
        transition_matrix = adjacency / row_sums[:, np.newaxis]
        
        # Create personalization vector
        personalization = np.ones(n) / n
        
        # Boost the entities of interest
        for entity_uri in entity_uris:
            if entity_uri in id_to_idx:
                idx = id_to_idx[entity_uri]
                personalization[idx] = 1.0
        
        # Normalize personalization vector
        personalization = personalization / personalization.sum()
        
        # Run PageRank algorithm
        pr = np.ones(n) / n
        
        # Power iteration
        for _ in range(iterations):
            next_pr = (1 - damping) * personalization + damping * np.dot(pr, transition_matrix)
            
            # Check for convergence
            if np.linalg.norm(next_pr - pr) < 1e-6:
                break
                
            pr = next_pr
        
        # Convert back to document IDs
        return {doc_ids[i]: float(pr[i]) for i in range(n)}

    def relationship_aware_retrieval(self, query, k=20):
        """Enhanced retrieval using both vector similarity and relationship importance"""
        # Step 1: Standard vector-based retrieval
        vector_results = self.document_store.retrieve(query, k=k * RetrievalConfig.INITIAL_POOL_MULTIPLIER)

        if not vector_results:
            logger.warning("No documents found in vector retrieval")
            return []

        # Step 2: Extract entities from query
        query_entities = self.extract_entities_from_query(query)
        logger.info(f"Extracted entities from query: {query_entities}")

        # Step 3: Calculate personalized PageRank scores (raw)
        pr_scores_raw = {}
        if query_entities:
            pr_scores_raw = self.calculate_relationship_scores(query_entities)

        # Step 4: Collect raw vector similarity scores
        # Use rank-based scoring: higher rank = higher score
        vector_scores_raw = {}
        for i, doc in enumerate(vector_results):
            # Score inversely proportional to rank position
            vector_scores_raw[doc.id] = len(vector_results) - i

        # Step 5: Normalize both score sets to [0, 1] using min-max normalization
        vector_scores_norm = self.normalize_scores(vector_scores_raw)
        pr_scores_norm = self.normalize_scores(pr_scores_raw) if pr_scores_raw else {}

        # Step 6: Define weighting for score combination
        alpha = RetrievalConfig.VECTOR_PAGERANK_ALPHA  # Weight for vector similarity vs. relationship

        logger.info(f"\n{'='*80}")
        logger.info(f"RELATIONSHIP-AWARE RETRIEVAL SCORING")
        logger.info(f"{'='*80}")
        logger.info(f"Combining: Vector Similarity ({alpha:.1f}) + PageRank ({1-alpha:.1f})")
        logger.info(f"\n--- Raw Vector Scores (rank-based) ---")
        logger.info(f"  Min: {min(vector_scores_raw.values())}")
        logger.info(f"  Max: {max(vector_scores_raw.values())}")
        logger.info(f"  Mean: {np.mean(list(vector_scores_raw.values())):.1f}")

        if pr_scores_raw:
            logger.info(f"\n--- Raw PageRank Scores (probability distribution) ---")
            pr_values = list(pr_scores_raw.values())
            logger.info(f"  Min: {min(pr_values):.6f}")
            logger.info(f"  Max: {max(pr_values):.6f}")
            logger.info(f"  Mean: {np.mean(pr_values):.6f}")
            logger.info(f"  Sum: {sum(pr_values):.6f} (should ≈ 1.0)")

        logger.info(f"\n--- Normalized Vector Scores [0,1] ---")
        logger.info(f"  Min: {min(vector_scores_norm.values()):.3f}")
        logger.info(f"  Max: {max(vector_scores_norm.values()):.3f}")
        logger.info(f"  Mean: {np.mean(list(vector_scores_norm.values())):.3f}")

        if pr_scores_norm:
            logger.info(f"\n--- Normalized PageRank Scores [0,1] ---")
            pr_norm_values = list(pr_scores_norm.values())
            logger.info(f"  Min: {min(pr_norm_values):.3f}")
            logger.info(f"  Max: {max(pr_norm_values):.3f}")
            logger.info(f"  Mean: {np.mean(pr_norm_values):.3f}")

        # Step 7: Combine normalized scores with weights (alpha already defined above)
        combined_results = []

        for doc in vector_results:
            # Get normalized scores (default to 0 if not found)
            sim_score_norm = vector_scores_norm.get(doc.id, 0.0)
            rel_score_norm = pr_scores_norm.get(doc.id, 0.0)

            # Combined score on [0, 1] scale
            final_score = alpha * sim_score_norm + (1 - alpha) * rel_score_norm

            combined_results.append((doc, final_score, sim_score_norm, rel_score_norm))

        # Sort by combined score and take top k
        combined_results.sort(key=lambda x: x[1], reverse=True)

        # Log top results for transparency
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP 10 COMBINED RESULTS")
        logger.info(f"{'='*80}")
        for i, (doc, final, sim, rel) in enumerate(combined_results[:10]):
            label = doc.metadata.get('label', 'Unknown')
            logger.info(f"\n{i+1}. {label}")
            logger.info(f"   Vector Sim: {sim:.3f} (weight={alpha:.1f}) → contrib={alpha*sim:.3f}")
            logger.info(f"   PageRank: {rel:.3f} (weight={1-alpha:.1f}) → contrib={(1-alpha)*rel:.3f}")
            logger.info(f"   Combined: {final:.3f}")

        final_results = [doc for doc, _, _, _ in combined_results[:k]]

        logger.info(f"\n{'='*80}")
        logger.info(f"Relationship-aware retrieval returning top {len(final_results)} documents")
        logger.info(f"{'='*80}\n")
        return final_results


    def get_wikidata_for_entity(self, entity_uri):
        """Get Wikidata ID for an entity if available"""
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
            logger.error(f"Error fetching Wikidata ID: {str(e)}")
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
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
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

    def get_wikidata_entities(self):
        """Get all entities that have Wikidata references"""
        query = """
        PREFIX crmdig: <http://www.ics.forth.gr/isl/CRMdig/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?entity ?label ?wikidata WHERE {
            ?entity crmdig:L54_is_same-as ?wikidata .
            OPTIONAL { ?entity rdfs:label ?label }
            FILTER(STRSTARTS(STR(?wikidata), "http://www.wikidata.org/entity/"))
        }
        """

        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()

            entities = []
            for result in results["results"]["bindings"]:
                entity_uri = result["entity"]["value"]
                wikidata_uri = result["wikidata"]["value"]
                wikidata_id = wikidata_uri.split('/')[-1]

                # Get label, with fallback
                label = result.get("label", {}).get("value", entity_uri.split('/')[-1])

                entities.append({
                    "entity": entity_uri,
                    "label": label,
                    "wikidata_id": wikidata_id,
                    "wikidata_url": f"https://www.wikidata.org/wiki/{wikidata_id}"
                })

            return entities
        except Exception as e:
            logger.error(f"Error fetching Wikidata entities: {str(e)}")
            return []

    def batch_process_documents(self, entities, batch_size=RetrievalConfig.DEFAULT_BATCH_SIZE, sleep_time=2):
        """Process RDF data into graph documents with batch processing to avoid rate limits"""
        total_entities = len(entities)
        logger.info(f"Processing {total_entities} entities in batches of {batch_size}")
        
        # Process in batches
        for i in range(0, total_entities, batch_size):
            batch = entities[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_entities + batch_size - 1)//batch_size}")
            
            # Process batch
            for entity in tqdm(batch, desc=f"Batch {i//batch_size + 1}", unit="entity"):
                entity_uri = entity["entity"]
                
                # Create enhanced document with CIDOC-CRM aware natural language
                doc_text, entity_label, entity_types = self.create_enhanced_document(entity_uri)

                # Determine primary entity type (filter out technical CIDOC-CRM class names)
                primary_type = "Unknown"
                if entity_types:
                    # Get human-readable types only
                    human_readable_types = [
                        t for t in entity_types
                        if not self.is_technical_class_name(t)
                    ]
                    # Use first human-readable type, or fall back to "Entity"
                    primary_type = human_readable_types[0] if human_readable_types else "Entity"

                # Add to document store
                self.document_store.add_document(
                    entity_uri,
                    doc_text,
                    {
                        "label": entity_label,
                        "type": primary_type,
                        "uri": entity_uri,
                        "all_types": entity_types  # Keep all types for debugging if needed
                    }
                )
            
            # Save progress after each batch
            self.document_store.save_document_graph('document_graph_temp.pkl')
            
            # Sleep between batches to avoid rate limits
            if i + batch_size < total_entities:
                logger.info(f"Sleeping for {sleep_time} seconds to avoid rate limits...")
                time.sleep(sleep_time)
    
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
            logger.info(f"Answering question directly: '{question}'")
            
            # Retrieve relevant documents
            retrieved_docs = self.retrieve(question, k=RetrievalConfig.DEFAULT_RETRIEVAL_K)  # Get more documents
            
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
            
            for i, doc in enumerate(retrieved_docs):
                entity_uri = doc.id
                entity_label = doc.metadata.get("label", entity_uri.split('/')[-1])

                # Build context without technical type information
                context += f"Entity: {entity_label}\n"
                context += doc.text + "\n\n"
                
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
                
                sources.append({
                    "id": i,
                    "entity_uri": entity_uri,
                    "entity_label": entity_label,
                    "type": doc.metadata.get("type", "unknown")
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