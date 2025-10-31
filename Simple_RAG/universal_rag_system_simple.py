"""
Simplified Universal RAG system with basic vector-based document retrieval only.
This version removes GNN, network metrics, and complex CIDOC processing for comparison.
"""

import logging
from typing import List, Dict, Any, Optional
import os
import time
from tqdm import tqdm

from SPARQLWrapper import SPARQLWrapper, JSON
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from llm_providers import get_llm_provider

logger = logging.getLogger(__name__)


class UniversalRagSystemSimple:
    """Simplified Universal RAG system with basic vector retrieval only"""
    
    def __init__(self, endpoint_url, config=None):
        """
        Initialize the simplified universal RAG system.
        
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
        
        # Initialize vector store
        self.vector_store = None
        self.documents = {}  # Store documents by URI
        
    @property
    def embeddings(self):
        """Return an embedding object compatible with FAISS"""
        class EmbeddingFunction:
            def __init__(self, provider):
                self.provider = provider
            
            def __call__(self, text):
                return self.provider.get_embeddings(text)
            
            def embed_query(self, text):
                return self.provider.get_embeddings(text)
            
            def embed_documents(self, texts):
                """Batch embed multiple documents efficiently"""
                # Check if provider supports batch embeddings
                if hasattr(self.provider, 'get_embeddings_batch'):
                    return self.provider.get_embeddings_batch(texts)
                else:
                    # Fall back to one-by-one
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
        
        # Check if saved data exists
        vector_index_path = 'vector_index_simple/index.faiss'
        
        if os.path.exists(vector_index_path):
            logger.info("Found existing vector store, loading...")
            try:
                self.vector_store = FAISS.load_local(
                    'vector_index_simple', 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded existing vector store")
                
                # Load document metadata
                import pickle
                if os.path.exists('documents_simple.pkl'):
                    with open('documents_simple.pkl', 'rb') as f:
                        self.documents = pickle.load(f)
                    logger.info(f"Loaded {len(self.documents)} documents")
                
                return True
            except Exception as e:
                logger.error(f"Error loading vector store: {str(e)}")
                logger.info("Will rebuild from scratch...")
        
        logger.info("Building vector store from RDF data...")
        self.process_rdf_data()
        
        # Save vector store
        vector_index_path = 'vector_index_simple'
        os.makedirs(vector_index_path, exist_ok=True)
        if self.vector_store:
            self.vector_store.save_local(vector_index_path)
            logger.info(f"Vector store saved to {vector_index_path}")
        
        # Save document metadata
        import pickle
        with open('documents_simple.pkl', 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Saved {len(self.documents)} documents")
        
        return True

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

    def create_simple_document(self, entity_uri):
        """Create a simple document from RDF triples"""
        try:
            # Get entity label
            label_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE {{ <{entity_uri}> rdfs:label ?label }}
            LIMIT 1
            """
            
            entity_label = entity_uri.split('/')[-1]
            try:
                self.sparql.setQuery(label_query)
                label_results = self.sparql.query().convert()
                if label_results["results"]["bindings"]:
                    entity_label = label_results["results"]["bindings"][0]["label"]["value"]
            except Exception as e:
                logger.warning(f"Error getting entity label: {str(e)}")
            
            # Get entity type
            type_query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?type ?typeLabel WHERE {{
                <{entity_uri}> rdf:type ?type .
                OPTIONAL {{ ?type rdfs:label ?typeLabel }}
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
                logger.warning(f"Error getting entity types: {str(e)}")
            
            # Get all triples about this entity
            triples_query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?pred ?predLabel ?obj ?objLabel WHERE {{
                <{entity_uri}> ?pred ?obj .
                OPTIONAL {{ ?pred rdfs:label ?predLabel }}
                OPTIONAL {{ ?obj rdfs:label ?objLabel }}
            }}
            LIMIT 50
            """
            
            triples = []
            try:
                self.sparql.setQuery(triples_query)
                triple_results = self.sparql.query().convert()
                
                for result in triple_results["results"]["bindings"]:
                    pred = result["pred"]["value"]
                    obj = result["obj"]["value"]
                    pred_label = result.get("predLabel", {}).get("value", pred.split('/')[-1])
                    obj_label = result.get("objLabel", {}).get("value", obj if result["obj"]["type"] == "literal" else obj.split('/')[-1])
                    
                    triples.append({
                        "predicate": pred_label,
                        "object": obj_label
                    })
            except Exception as e:
                logger.warning(f"Error getting entity triples: {str(e)}")
            
            # Create simple document text
            text = f"Entity: {entity_label}\n\n"
            text += f"URI: {entity_uri}\n\n"
            
            if entity_types:
                text += "Types:\n"
                for entity_type in entity_types:
                    text += f"- {entity_type}\n"
                text += "\n"
            
            if triples:
                text += "Properties:\n"
                for triple in triples:
                    text += f"- {triple['predicate']}: {triple['object']}\n"
            
            return text, entity_label, entity_types[0] if entity_types else "Unknown"
            
        except Exception as e:
            logger.error(f"Error creating document for {entity_uri}: {str(e)}")
            return f"Entity: {entity_uri}", entity_uri, "Unknown"

    def process_rdf_data(self):
        """Process RDF data into documents with simple vector retrieval"""
        logger.info("Processing RDF data with simple vector retrieval...")
        
        # Get all entities
        entities = self.get_all_entities()
        total_entities = len(entities)
        logger.info(f"Found {total_entities} entities")
        
        # Process entities in batches
        batch_size = 50
        all_docs = []
        
        # Rate limiting
        global_token_count = 0
        tokens_per_min_limit = 950000
        last_reset_time = time.time()
        
        for i in range(0, total_entities, batch_size):
            batch = entities[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_entities + batch_size - 1)//batch_size}")
            
            for entity in tqdm(batch, desc=f"Batch {i//batch_size + 1}", unit="entity"):
                entity_uri = entity["entity"]
                
                # Rate limit check
                current_time = time.time()
                if current_time - last_reset_time >= 60:
                    global_token_count = 0
                    last_reset_time = current_time
                
                if global_token_count > tokens_per_min_limit:
                    wait_time = 60 - (current_time - last_reset_time) + 1
                    if wait_time > 0:
                        logger.info(f"Approaching rate limit. Waiting {wait_time:.1f} seconds...")
                        time.sleep(wait_time)
                        global_token_count = 0
                        last_reset_time = time.time()
                
                # Create document
                try:
                    doc_text, entity_label, entity_type = self.create_simple_document(entity_uri)
                    
                    # Estimate tokens
                    estimated_tokens = len(doc_text) / 4
                    global_token_count += estimated_tokens
                    
                    # Store document
                    self.documents[entity_uri] = {
                        "label": entity_label,
                        "type": entity_type,
                        "text": doc_text
                    }
                    
                    # Create Langchain document
                    doc = Document(
                        page_content=doc_text,
                        metadata={
                            "uri": entity_uri,
                            "label": entity_label,
                            "type": entity_type
                        }
                    )
                    all_docs.append(doc)
                    
                except Exception as e:
                    logger.error(f"Error processing entity {entity_uri}: {str(e)}")
                    continue
            
            # Pause between batches
            logger.info("Completed batch, pausing for 2 seconds...")
            time.sleep(2)
        
        # Build vector store with batched embeddings
        logger.info(f"Building vector store with {len(all_docs)} documents...")
        self.build_vector_store_batched(all_docs)
        
        logger.info("RDF data processing complete")
    
    def build_vector_store_batched(self, all_docs, embedding_batch_size=100):
        """Build vector store with batched embedding requests"""
        from langchain_community.vectorstores import FAISS
        
        total_docs = len(all_docs)
        logger.info(f"Creating embeddings for {total_docs} documents in batches of {embedding_batch_size}")
        
        # Process embeddings in batches with progress bar
        self.vector_store = None
        
        with tqdm(total=total_docs, desc="Creating embeddings", unit="doc") as pbar:
            for i in range(0, total_docs, embedding_batch_size):
                batch = all_docs[i:i+embedding_batch_size]
                
                try:
                    if self.vector_store is None:
                        # Create initial vector store
                        self.vector_store = FAISS.from_documents(batch, self.embeddings)
                    else:
                        # Add to existing vector store
                        self.vector_store.add_documents(batch)
                    
                    pbar.update(len(batch))
                    
                    # Small pause between batches
                    if i + embedding_batch_size < total_docs:
                        time.sleep(0.5)
                        
                except Exception as e:
                    logger.error(f"Error creating embeddings for batch starting at {i}: {str(e)}")
                    # Continue with next batch
                    pbar.update(len(batch))
        
        logger.info("Vector store built successfully")

    def retrieve(self, query, k=10):
        """Simple vector-based retrieval"""
        logger.info(f"Retrieving documents for query: '{query}'")
        
        if not self.vector_store:
            logger.warning("Vector store not initialized")
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} documents")
        
        return results

    def answer_question(self, question, include_wikidata=False):
        """Answer a question using simple vector retrieval"""
        logger.info(f"Answering question: '{question}'")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, k=10)
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }
        
        # Create context from retrieved documents
        context = ""
        for i, doc in enumerate(retrieved_docs):
            context += f"\n--- Document {i+1} ---\n"
            context += doc.page_content + "\n"
        
        # Simple system prompt
        system_prompt = """You are a helpful assistant that answers questions about cultural heritage data.
        
Answer questions based only on the information provided in the documents.
Be clear and concise in your answers.
If the information is not in the documents, say so."""
        
        # Create prompt
        prompt = f"""Based on the following documents, answer this question:

Question: {question}

Documents:
{context}

Provide a clear answer based on the information in the documents."""
        
        # Generate answer
        answer = self.llm_provider.generate(system_prompt, prompt)
        
        # Prepare sources
        sources = []
        for i, doc in enumerate(retrieved_docs):
            sources.append({
                "id": i,
                "entity_uri": doc.metadata.get("uri", ""),
                "entity_label": doc.metadata.get("label", ""),
                "type": doc.metadata.get("type", "unknown")
            })
        
        return {
            "answer": answer,
            "sources": sources
        }

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
                wikidata_id = wikidata_uri.split('/')[-1]
                return wikidata_id
            return None
        except Exception as e:
            logger.error(f"Error fetching Wikidata ID: {str(e)}")
            return None