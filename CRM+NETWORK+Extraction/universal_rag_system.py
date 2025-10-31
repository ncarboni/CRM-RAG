"""
Universal RAG system with graph-based document retrieval.
This system can be applied to any RDF dataset and uses coherent subgraph extraction
to enhance document retrieval using CIDOC-CRM relationship weights.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from SPARQLWrapper import SPARQLWrapper, JSON
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from graph_document_store import GraphDocumentStore
from llm_providers import get_llm_provider, BaseLLMProvider
import os
import time
from tqdm import tqdm
import networkx as nx

logger = logging.getLogger(__name__)



class UniversalRagSystem:
    """Universal RAG system with graph-based document retrieval"""
    
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
        import os
        
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
                    import pickle
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
                import types
                self.document_store.load_document_graph = types.MethodType(load_document_graph, self.document_store)
            
            # Try to load document graph
            graph_loaded = self.document_store.load_document_graph(doc_graph_path)
            
            # Try to load vector store
            vector_loaded = False
            try:
                from langchain_community.vectorstores import FAISS
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
                import pickle
                try:
                    with open(path, 'wb') as f:
                        pickle.dump(self.docs, f)
                    logger.info(f"Document graph saved to {path}")
                    return True
                except Exception as e:
                    logger.error(f"Error saving document graph: {str(e)}")
                    return False
                    
            # Add method to class
            import types
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
        """Convert CIDOC-CRM RDF relationships to natural language"""
        
        # Get the simplified predicate (without namespace)
        simple_pred = predicate.split('/')[-1]
        
        # Handle missing labels
        subject_label = subject_label or subject_uri.split('/')[-1]
        object_label = object_label or object_uri.split('/')[-1]
        
        # Core CIDOC-CRM properties with their natural language interpretations
        cidoc_relationships = {
            # Spatial relationships
            "P89_falls_within": f"{subject_label} is located within {object_label}",
            "P55_has_current_location": f"{subject_label} is currently located at {object_label}",
            "P53_has_former_or_current_location": f"{subject_label} is or was located at {object_label}",
            "P156_occupies": f"{subject_label} occupies {object_label}",
            "P157_is_at_rest_relative_to": f"{subject_label} is fixed relative to {object_label}",
            
            # Temporal relationships
            "P4_has_time-span": f"{subject_label} occurred during {object_label}",
            "P114_is_equal_in_time_to": f"{subject_label} occurred at the same time as {object_label}",
            "P115_finishes": f"{subject_label} finished at the same time as {object_label}",
            "P116_starts": f"{subject_label} started at the same time as {object_label}",
            "P117_occurs_during": f"{subject_label} occurred during {object_label}",
            "P118_overlaps_in_time_with": f"{subject_label} overlaps in time with {object_label}",
            
            # Physical relationships
            "P46_is_composed_of": f"{subject_label} is composed of {object_label}",
            "P56_bears_feature": f"{subject_label} has the feature {object_label}",
            "P128_carries": f"{subject_label} carries {object_label}",
            "P59_has_section": f"{subject_label} has section {object_label}",
            
            # Conceptual relationships
            "P2_has_type": f"{subject_label} is of type {object_label}",
            "P1_is_identified_by": f"{subject_label} is identified by {object_label}",
            "P67_refers_to": f"{subject_label} refers to {object_label}",
            "P129_is_about": f"{subject_label} is about {object_label}",
            "P138_represents": f"{subject_label} represents {object_label}",
            
            # Production and creation
            "P108i_was_produced_by": f"{subject_label} was produced by {object_label}",
            "P94i_was_created_by": f"{subject_label} was created by {object_label}",
            
            # VIR ontology (for visual items)
            "K1i_is_denoted_by": f"{subject_label} is denoted by {object_label}",
            "K17_has_attribute": f"{subject_label} has the attribute {object_label}",
            "K24_portray": f"{subject_label} portrays {object_label}",
            "K20i_is_composed_of": f"{subject_label} is composed of {object_label}"
        }
        
        # Return natural language interpretation if available, otherwise a default format
        return cidoc_relationships.get(simple_pred, f"{subject_label} {simple_pred.replace('_', ' ')} {object_label}")

    def get_entity_context(self, entity_uri, depth=2):
        """Get entity context by traversing the graph bidirectionally"""
        
        context_statements = []
        visited = set()
        
        def traverse(uri, current_depth=0, direction="both"):
            if uri in visited or current_depth > depth:
                return
                
            visited.add(uri)
            
            # Get entity label
            label_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE {{ <{uri}> rdfs:label ?label }}
            LIMIT 1
            """
            
            entity_label = None
            try:
                self.sparql.setQuery(label_query)
                label_results = self.sparql.query().convert()
                if label_results["results"]["bindings"]:
                    entity_label = label_results["results"]["bindings"][0]["label"]["value"]
            except Exception as e:
                logger.error(f"Error getting entity label: {str(e)}")
            
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
                        
                        # Get labels if available
                        pred_label = result.get("predLabel", {}).get("value", pred.split('/')[-1])
                        obj_label = result.get("objLabel", {}).get("value", obj.split('/')[-1])
                        
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
                        
                        # Get labels if available
                        subj_label = result.get("subjLabel", {}).get("value", subj.split('/')[-1])
                        pred_label = result.get("predLabel", {}).get("value", pred.split('/')[-1])
                        
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
            # Get entity label
            label_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE {{ <{entity_uri}> rdfs:label ?label }}
            LIMIT 1
            """
            
            entity_label = entity_uri.split('/')[-1]  # Default to URI fragment
            try:
                self.sparql.setQuery(label_query)
                label_results = self.sparql.query().convert()
                if label_results["results"]["bindings"]:
                    entity_label = label_results["results"]["bindings"][0]["label"]["value"]
            except Exception as e:
                logger.warning(f"Error getting entity label for {entity_uri}: {str(e)}")
            
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
                context_statements = self.get_entity_context(entity_uri, depth=2) 
            except Exception as e:
                logger.warning(f"Error getting entity context for {entity_uri}: {str(e)}")
                context_statements = []
            
            # Create document text
            text = f"# {entity_label}\n\n"
            
            # Add entity identifier
            text += f"URI: {entity_uri}\n\n"
            
            # Add entity types
            if entity_types:
                text += "## Types\n\n"
                for type_label in entity_types:
                    text += f"- {type_label}\n"
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

    def process_rdf_data(self):
        """Process RDF data into graph documents with enhanced CIDOC-CRM understanding"""
        import os
        import time
        from tqdm import tqdm
        
        logger.info("Processing RDF data with enhanced CIDOC-CRM understanding...")
        
        # Get all entities
        entities = self.get_all_entities()
        total_entities = len(entities)
        logger.info(f"Found {total_entities} entities")
        
        # Global rate limit tracking
        global_token_count = 0
        tokens_per_min_limit = 950000  # Set slightly below the actual limit of 1M
        last_reset_time = time.time()
        
        # First pass: create document nodes with enhanced content
        logger.info("Creating enhanced document nodes...")
        batch_size = 50  # Process in reasonable batches for progress tracking
        
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
                    
                    # Estimate token count - very rough estimate 
                    # (1 token ≈ 4 chars in English on average)
                    estimated_tokens = len(doc_text) / 4
                    global_token_count += estimated_tokens
                    
                    # Determine primary entity type
                    primary_type = "Unknown"
                    if entity_types:
                        primary_type = entity_types[0]
                    
                    # Add to document store
                    self.document_store.add_document(
                        entity_uri, 
                        doc_text, 
                        {
                            "label": entity_label,
                            "type": primary_type,
                            "uri": entity_uri,
                            "all_types": entity_types
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

    def build_vector_store_batched(self, batch_size=50):
        """Build vector store with batched embedding requests to avoid rate limits"""
        import os
        import time
        
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
        tokens_per_min_limit = 950000
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
            combined_score = 0.6 * vector_score + 0.4 * graph_score
            
            ranked_docs.append((doc, combined_score))
        
        # Sort by combined score
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return [doc for doc, _ in ranked_docs[:k]]

    def get_cidoc_system_prompt(self):
        """Get a system prompt with CIDOC-CRM knowledge"""
        
        return """You are a cultural heritage expert with deep knowledge of the CIDOC-CRM ontology.
        
ABOUT CIDOC-CRM:
CIDOC-CRM is an ontology that provides definitions and a formal structure for describing the concepts and relationships used in cultural heritage documentation. When interpreting data with CIDOC-CRM:

1. Entities are organized in a hierarchy where:
   - E1_CRM_Entity is the root
   - E77_Persistent_Item includes physical and conceptual items that persist over time
   - E53_Place represents physical spaces
   - E18_Physical_Thing represents tangible objects
   - E28_Conceptual_Object represents abstract concepts

2. Key relationship properties include:
   - P89_falls_within: spatial containment (X falls within Y means Y contains X)
   - P46_is_composed_of: part-whole relationships
   - P1_is_identified_by: connects entities to their identifiers
   - P2_has_type: classifies entities
   - P55_has_current_location: physical location of an object
   - P108i_was_produced_by: connects objects to their creation events

3. For spatial relationships:
   - Follow P89_falls_within chains to get complete location hierarchies
   - P168_is_approximated_by with WKT literals provides coordinates
   - P55_has_current_location points to current physical location

4. For temporal relationships:
   - P4_has_time-span connects to time periods
   - P82_at_some_time_within provides date ranges
   - P108i_was_produced_by connects to production events

5. For visual representations (VIR ontology):
   - IC9_Representation represents visual depictions
   - K24_portray connects images to what they depict
   - K17_has_attribute connects to visual attributes

When answering questions:
1. Interpret the CIDOC-CRM relationships to extract meaningful information
2. Follow chains of relationships to get complete context
3. Translate CIDOC-CRM terminology into natural language
4. Provide accurate information based solely on the data provided

For each answer, if the data is insufficient to provide a complete answer, explain what information is available and what is missing.
"""

    def hybrid_answer_question(self, question):
        """Hybrid approach that tries direct querying first, then falls back to RAG"""
        
        # Try direct querying first
        direct_results = self.answer_with_direct_query(question)
        
        if direct_results:
            # Convert SPARQL results to natural language
            from langchain_openai import ChatOpenAI
            
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
    
    def get_all_entities(self):
        """Get all labeled entities from SPARQL endpoint"""
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT DISTINCT ?entity ?label WHERE {
            ?entity rdfs:label ?label .
        }
        LIMIT 1000
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            entities = []
            for result in results["results"]["bindings"]:
                entities.append({
                    "entity": result["entity"]["value"],
                    "label": result["label"]["value"]
                })
                
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
        """Get outgoing relationships from an entity"""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?predicate ?object WHERE {{
            <{entity_uri}> ?predicate ?object .
            
            # Only include relationships to other entities with labels
            ?object rdfs:label ?objectLabel .
            
            # Filter for meaningful relationships
            FILTER(STRSTARTS(STR(?predicate), "http://"))
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            relationships = []
            for result in results["results"]["bindings"]:
                relationships.append({
                    "predicate": result["predicate"]["value"],
                    "object": result["object"]["value"]
                })
                
            return relationships
        except Exception as e:
            logger.error(f"Error fetching outgoing relationships: {str(e)}")
            return []
    
    def get_incoming_relationships(self, entity_uri):
        """Get incoming relationships to an entity"""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?subject ?predicate WHERE {{
            ?subject ?predicate <{entity_uri}> .
            
            # Only include relationships from other entities with labels
            ?subject rdfs:label ?subjectLabel .
            
            # Filter for meaningful relationships
            FILTER(STRSTARTS(STR(?predicate), "http://"))
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            relationships = []
            for result in results["results"]["bindings"]:
                relationships.append({
                    "subject": result["subject"]["value"],
                    "predicate": result["predicate"]["value"]
                })
                
            return relationships
        except Exception as e:
            logger.error(f"Error fetching incoming relationships: {str(e)}")
            return []

    def extract_entities_from_query(self, query):
        """Extract entity URIs mentioned in the query for relationship-aware retrieval"""
        import re
        
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

    def calculate_relationship_scores(self, entity_uris, damping=0.85, iterations=20):
        """Calculate personalized PageRank scores for all entities in the graph"""
        import numpy as np
        
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
        vector_results = self.document_store.retrieve(query, k=k*2)
        
        if not vector_results:
            logger.warning("No documents found in vector retrieval")
            return []
        
        # Step 2: Extract entities from query
        query_entities = self.extract_entities_from_query(query)
        logger.info(f"Extracted entities from query: {query_entities}")
        
        # Step 3: Calculate personalized PageRank scores
        pr_scores = {}
        if query_entities:
            pr_scores = self.calculate_relationship_scores(query_entities)
        
        # Step 4: Combine vector similarity with relationship scores
        combined_results = []
        alpha = 0.6  # Weight for vector similarity vs. relationship (0.6 vs 0.4)
        
        # Transform vector ranking to scores (higher rank = higher score)
        for i, doc in enumerate(vector_results):
            # Vector similarity score (inversely proportional to rank)
            sim_score = (len(vector_results) - i) / len(vector_results)
            
            # Relationship score (defaults to 0 if not found)
            rel_score = pr_scores.get(doc.id, 0.0)
            
            # Combined score
            final_score = alpha * sim_score + (1 - alpha) * rel_score
            
            combined_results.append((doc, final_score))
            
        # Sort by combined score and take top k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        final_results = [doc for doc, _ in combined_results[:k]]
        
        logger.info(f"Relationship-aware retrieval found {len(final_results)} documents")
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
        import requests
        import time
        
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
            ?entity rdfs:label ?label .
            FILTER(STRSTARTS(STR(?wikidata), "http://www.wikidata.org/entity/"))
        }
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            entities = []
            for result in results["results"]["bindings"]:
                wikidata_uri = result["wikidata"]["value"]
                wikidata_id = wikidata_uri.split('/')[-1]
                
                entities.append({
                    "entity": result["entity"]["value"],
                    "label": result["label"]["value"],
                    "wikidata_id": wikidata_id,
                    "wikidata_url": f"https://www.wikidata.org/wiki/{wikidata_id}"
                })
                
            return entities
        except Exception as e:
            logger.error(f"Error fetching Wikidata entities: {str(e)}")
            return []

    def batch_process_documents(self, entities, batch_size=50, sleep_time=2):
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
                
                # Determine primary entity type
                primary_type = "Unknown"
                if entity_types:
                    primary_type = entity_types[0]
                
                # Add to document store
                self.document_store.add_document(
                    entity_uri, 
                    doc_text, 
                    {
                        "label": entity_label,
                        "type": primary_type,
                        "uri": entity_uri,
                        "all_types": entity_types
                    }
                )
            
            # Save progress after each batch
            self.document_store.save_document_graph('document_graph_temp.pkl')
            
            # Sleep between batches to avoid rate limits
            if i + batch_size < total_entities:
                logger.info(f"Sleeping for {sleep_time} seconds to avoid rate limits...")
                time.sleep(sleep_time)
    
    def compute_coherent_subgraph(self, candidates, adjacency_matrix, initial_scores, k=10, alpha=0.7):
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
        
        # Normalize initial scores to [0, 1]
        if np.max(initial_scores) > 0:
            normalized_scores = initial_scores / np.max(initial_scores)
        else:
            normalized_scores = initial_scores
        
        logger.info(f"Starting coherent subgraph extraction with alpha={alpha}")
        
        # First selection: pick the highest-scoring document
        first_idx = np.argmax(normalized_scores)
        selected_indices.append(first_idx)
        selected_mask[first_idx] = True
        logger.info(f"Selected document 1/{k}: {candidates[first_idx].metadata.get('label', 'Unknown')}")
        
        # Iteratively select remaining documents
        for iteration in range(1, k):
            if len(selected_indices) >= n:
                break
            
            best_score = -np.inf
            best_idx = -1
            
            # Evaluate each unselected candidate
            for idx in range(n):
                if selected_mask[idx]:
                    continue
                
                # Individual relevance component
                relevance = normalized_scores[idx]
                
                # Connectivity component: sum of weighted edges to already-selected documents
                connectivity = 0.0
                for selected_idx in selected_indices:
                    # Check both directions in adjacency matrix
                    edge_weight = max(
                        adjacency_matrix[idx, selected_idx],
                        adjacency_matrix[selected_idx, idx]
                    )
                    connectivity += edge_weight
                
                # Normalize connectivity by number of selected documents to avoid bias toward later iterations
                if len(selected_indices) > 0:
                    connectivity = connectivity / len(selected_indices)
                
                # Combined score: balance individual relevance and connectivity
                combined_score = alpha * relevance + (1 - alpha) * connectivity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx == -1:
                logger.warning(f"Could not find more connected documents after {len(selected_indices)} selections")
                break
            
            selected_indices.append(best_idx)
            selected_mask[best_idx] = True
            logger.info(f"Selected document {iteration+1}/{k}: {candidates[best_idx].metadata.get('label', 'Unknown')} "
                       f"(relevance={normalized_scores[best_idx]:.3f}, connectivity={(1-alpha)*best_score/alpha if alpha > 0 else 0:.3f}, "
                       f"combined={best_score:.3f})")
        
        # Return selected documents in order
        return [candidates[idx] for idx in selected_indices]

    def retrieve(self, query, k=10, initial_pool_size=30, alpha=0.7):
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
        adjacency_matrix = self.document_store.create_adjacency_matrix(doc_ids, max_hops=2)
        
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
            retrieved_docs = self.retrieve(question, k=10)  # Get more documents
            
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
                entity_type = doc.metadata.get("type", "Unknown")
                
                context += f"Entity: {entity_label} (Type: {entity_type}, URI: {entity_uri})\n"
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

    Provide a comprehensive answer that accurately interprets the CIDOC-CRM relationships in the data.
    When referring to entities in your answer, use their proper names rather than saying "Document 1" or "Document 2".
    Refer to the entities by their actual names (like "Panagia Phorbiottisa" or "Nikitari") instead of document numbers.
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