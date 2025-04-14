import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple

from SPARQLWrapper import SPARQLWrapper, JSON
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import MergerRetriever
from ontology_processor import OntologyProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class FusekiRagSystem:
    """RAG system that integrates RDF data from Fuseki with ontology knowledge"""
    
    def __init__(self, endpoint_url="http://localhost:3030/asinou/sparql", 
                 embedding_model_name="text-embedding-3-small",  # Updated embedding model
                 openai_api_key=None,  # New parameter for API key
                 openai_model="o-mini",  # New parameter for model name
                 temperature=0.7,  # Added temperature parameter
                 ontology_docs_path=None):
        """
        Initialize the RAG system.
        
        Args:
            endpoint_url: URL of the Fuseki SPARQL endpoint
            embedding_model_name: Name of the embedding model to use
            openai_api_key: OpenAI API key
            openai_model: OpenAI model name to use
            temperature: Temperature setting for the language model
            ontology_docs_path: List of paths to ontology documentation files
        """
        self.endpoint_url = endpoint_url
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        self._embeddings = None
        self.rdf_vectorstore = None
        self.ontology_vectorstore = None
        self.embedding_model_name = embedding_model_name
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.openai_model = openai_model
        self.temperature = temperature
        self.chat_chain = None
        
        # Initialize ontology processor
        self.ontology_processor = OntologyProcessor(ontology_docs_path)
        
    @property
    def embeddings(self):
        """Lazy-load embeddings model"""
        if self._embeddings is None:
            logger.info("Initializing embeddings model...")
            try:
                # Use OpenAI embeddings instead of HuggingFace
                self._embeddings = OpenAIEmbeddings(
                    model=self.embedding_model_name,
                    openai_api_key=self.openai_api_key
                )
                logger.info("OpenAI embeddings model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings model: {str(e)}")
                raise
        return self._embeddings


    def ensure_vectorstores(self):
        """
        Ensure both RDF and ontology vector stores are available.
        
        Returns:
            bool: True if at least RDF vector store is available, False otherwise
        """
        # Initialize RDF vector store
        if self.rdf_vectorstore is None:
            logger.info("RDF vector store not initialized, attempting to load...")
            self.build_rdf_vectorstore()
            
        # Initialize ontology vector store
        if self.ontology_vectorstore is None:
            logger.info("Ontology vector store not initialized, attempting to load...")
            self.ontology_vectorstore = self.build_ontology_vectorstore()
        
        # Check if both vector stores are available
        if self.rdf_vectorstore is None:
            logger.error("Failed to initialize RDF vector store")
            return False
            
        if self.ontology_vectorstore is None:
            logger.warning("Failed to initialize ontology vector store, will continue with RDF only")
            
        return True
        
    def build_rdf_vectorstore(self, force_rebuild=False):
        """
        Build the RDF vector store from SPARQL endpoint data.
        
        Args:
            force_rebuild: Whether to force rebuilding the vector store
            
        Returns:
            FAISS vector store or None if failed
        """
        # Check if we already have a vectorstore saved that we can reuse
        if not force_rebuild and os.path.exists('rdf_index/index.faiss'):
            logger.info("RDF vector store already exists, loading from disk...")
            try:
                self.rdf_vectorstore = FAISS.load_local('rdf_index', self.embeddings,
                                                    allow_dangerous_deserialization=True)
                logger.info("Successfully loaded existing RDF vector store")
                return self.rdf_vectorstore
            except Exception as e:
                logger.error(f"Failed to load existing RDF vector store: {str(e)}")
                # If loading fails, we'll rebuild
        
        # Create documents from the RDF data
        logger.info("Building new RDF vector store...")
        documents = self.create_documents_from_endpoint()
        
        if not documents:
            logger.error("No RDF documents to index")
            return None
        
        # Log a sample document for debugging
        if documents:
            logger.debug(f"Sample RDF document content: {documents[0].page_content[:500]}...")
        
        # Split documents into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks can help with precision
            chunk_overlap=200,  # More overlap maintains context
            separators=["\n\n", "\n", ". ", " ", ""],  # Explicit separators
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        logger.info(f"Created {len(texts)} RDF document chunks")
        
        # Create vector store
        logger.info("Creating FAISS vector store for RDF data...")
        try:
            self.rdf_vectorstore = FAISS.from_documents(texts, self.embeddings)
            logger.info("RDF vector store created successfully")
            
            # Save vector store for future use
            if not os.path.exists('rdf_index'):
                os.makedirs('rdf_index')
            self.rdf_vectorstore.save_local('rdf_index')
            logger.info("RDF vector store saved to disk")
            
            return self.rdf_vectorstore
        except Exception as e:
            logger.error(f"Error creating RDF vector store: {str(e)}")
            return None

    def build_ontology_vectorstore(self, force_rebuild=False):
        """
        Build ontology vector store from ontology documentation.
        
        Args:
            force_rebuild: Whether to force rebuilding the vector store
            
        Returns:
            FAISS vector store or None if failed
        """
        if not self.ontology_processor:
            logger.error("Ontology processor not initialized")
            return None
            
        # Build the ontology vector store
        self.ontology_vectorstore = self.ontology_processor.build_vectorstore(self.embeddings, force_rebuild)
        return self.ontology_vectorstore

    def create_documents_from_endpoint(self):
        """
        Create documents from SPARQL endpoint data for indexing.
        
        Returns:
            List of Document objects
        """
        logger.info("Creating documents from SPARQL endpoint...")
        
        # Test connection
        if not self.test_connection():
            logger.error("Failed to connect to Fuseki endpoint")
            return []
        
        # Get all entities with labels
        entities = self.get_all_entities()
        logger.info(f"Found {len(entities)} entities")
        
        documents = []
        for i, entity_info in enumerate(entities):
            if i % 100 == 0:
                logger.info(f"Processed {i}/{len(entities)} entities")
                
            entity_uri = entity_info["entity"]
            entity_label = entity_info["label"]
            
            # Get entity details
            details = self.get_entity_details(entity_uri)
            
            # Get related entities
            related = self.get_related_entities(entity_uri)
            
            # Format as text
            text = f"Entity: {entity_label} ({entity_uri})\n"
            
            # Add details
            if details:
                text += "Properties:\n"
                for detail in details:
                    pred = detail.get("predicateLabel", detail["predicate"])
                    obj = detail.get("objectLabel", detail["object"])
                    text += f"  - {pred}: {obj}\n"
            
            # Add related entities
            if related:
                text += "Referenced by:\n"
                for relation in related:
                    subj = relation.get("subjectLabel", relation["subject"])
                    pred = relation.get("predicateLabel", relation["predicate"])
                    text += f"  - {subj} {pred}\n"
            
            # Extract entity type from URI or details if available
            entity_type = "Unknown"
            for detail in details:
                if "rdf-syntax-ns#type" in detail["predicate"]:
                    entity_type = detail["object"].split("/")[-1]
                    break
            
            # Create document
            doc = Document(
                page_content=text, 
                metadata={
                    "entity": entity_uri, 
                    "label": entity_label,
                    "type": entity_type,
                    "source": "rdf_data"
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} documents in total")
        return documents

    def get_entity_details(self, entity_uri):
        """
        Get all details about a specific entity.
        
        Args:
            entity_uri: URI of the entity to get details for
            
        Returns:
            List of dictionaries with entity details
        """
        logger.info(f"Fetching details for entity: {entity_uri}")
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
                
            logger.info(f"Retrieved {len(details)} details for entity")
            return details
        except Exception as e:
            logger.error(f"Error fetching entity details: {str(e)}")
            return []

    def get_related_entities(self, entity_uri):
        """
        Get all entities that reference this entity.
        
        Args:
            entity_uri: URI of the entity to get related entities for
            
        Returns:
            List of dictionaries with related entity information
        """
        logger.info(f"Fetching related entities for: {entity_uri}")
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?subject ?predicate ?predicateLabel ?subjectLabel WHERE {{
            ?subject ?predicate <{entity_uri}> .
            OPTIONAL {{ ?subject rdfs:label ?subjectLabel }}
            OPTIONAL {{ ?predicate rdfs:label ?predicateLabel }}
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            related = []
            for result in results["results"]["bindings"]:
                relation = {
                    "subject": result["subject"]["value"],
                    "predicate": result["predicate"]["value"]
                }
                if "subjectLabel" in result:
                    relation["subjectLabel"] = result["subjectLabel"]["value"]
                if "predicateLabel" in result:
                    relation["predicateLabel"] = result["predicateLabel"]["value"]
                related.append(relation)
                
            logger.info(f"Retrieved {len(related)} related entities")
            return related
        except Exception as e:
            logger.error(f"Error fetching related entities: {str(e)}")
            return []

    def search(self, query, k=5):
        """
        Search for information in the vector stores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        documents = []
        
        # Ensure vector stores are initialized
        if not self.ensure_vectorstores():
            logger.error("Cannot search without vector stores")
            return documents
        
        # Search RDF vector store
        if self.rdf_vectorstore:
            rdf_docs = self.rdf_vectorstore.similarity_search(query, k=k)
            documents.extend(rdf_docs)
        
        # Search ontology vector store (with fewer results)
        if self.ontology_vectorstore:
            ontology_docs = self.ontology_vectorstore.similarity_search(query, k=min(k//2, 3))
            documents.extend(ontology_docs)
        
        return documents


    def answer_question(self, question):
        """Answer a question using both RDF and ontology knowledge."""
        logger.info(f"Answering question: '{question}'")
        
        # Initialize chat chain if needed
        if not hasattr(self, 'chat_chain') or self.chat_chain is None:
            logger.info("Chat chain not initialized, setting up...")
            if not self.setup_chat_chain():
                logger.error("Failed to set up chat chain")
                return {
                    "answer": "I'm sorry, I couldn't set up the answering system. Please check the logs for details.",
                    "sources": []
                }
        
        try:
            # Use standard retrieval for regular questions
            result = self.chat_chain.invoke({"question": question})
            
            # Process answer and sources
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Format sources
            sources = []
            
            # Group sources by type
            rdf_sources = []
            ontology_sources = []
            
            for i, doc in enumerate(source_docs):
                source_type = doc.metadata.get("source", "unknown")
                
                if "rdf_data" in source_type:
                    entity_uri = doc.metadata.get("entity", "")
                    entity_label = doc.metadata.get("label", "")
                    
                    rdf_sources.append({
                        "id": i,
                        "entity_uri": entity_uri,
                        "entity_label": entity_label,
                        "type": source_type
                    })
                elif source_type == "ontology_documentation":
                    concept_id = doc.metadata.get("concept_id", "")
                    concept_name = doc.metadata.get("concept_name", "")
                    
                    ontology_sources.append({
                        "id": i,
                        "concept_id": concept_id,
                        "concept_name": concept_name,
                        "type": "ontology_documentation"
                    })
            
            # Add sources in order
            sources.extend(rdf_sources)
            sources.extend(ontology_sources)
            
            logger.info(f"Generated answer with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"I'm sorry, I encountered an error while answering your question: {str(e)}",
                "sources": []
            }


    def answer_question_with_graph(self, question):
        """Answer a question using enhanced graph retrieval"""
        # Initialize chat chain if needed
        if not hasattr(self, 'chat_chain') or self.chat_chain is None:
            if not self.setup_chat_chain():
                return {
                    "answer": "I'm sorry, I couldn't set up the answering system.",
                    "sources": []
                }
        
        # Get base retrieval
        base_docs = self.search(question, k=5)
        
        # Enhance with graph context
        enhanced_docs = self.enhance_retrieval_with_graph(question, base_docs, k=10)
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in enhanced_docs])
        
        # Now use the chat chain with the enhanced context
        result = self.chat_chain.invoke({"question": question, "context": context})
        
        # Process answer and sources
        answer = result["answer"]
        source_docs = result.get("source_documents", enhanced_docs)
        
        # Format sources
        sources = []
        for i, doc in enumerate(source_docs):
            if "entity" in doc.metadata and "label" in doc.metadata:
                sources.append({
                    "id": i,
                    "entity_uri": doc.metadata["entity"],
                    "entity_label": doc.metadata["label"],
                    "type": doc.metadata.get("source", "unknown")
                })
        
        return {
            "answer": answer,
            "sources": sources
        }

    def retrieve_with_graph(self, query, k=10):
        """Enhanced retrieval that combines vector search with graph traversal"""
        # Get query embedding for later scoring
        query_embedding = self.embeddings.embed_query(query)
        
        # Get regular vector search results
        vector_docs = self.search(query, k=k//2)
        
        # Get graph-traversal results
        graph_docs = self.graph_retrieval(query, initial_k=3, expansion_depth=2, max_nodes=k)
        
        # Score and rank all documents
        all_docs = []
        for doc in vector_docs:
            # Vector docs already have relevance built in from the retrieval
            all_docs.append((doc, 0.9))  # High base score for direct vector matches
        
        for doc in graph_docs:
            entity_uri = doc.metadata.get("entity", "")
            if entity_uri:
                # For graph docs, compute relevance based on combination
                # of vector similarity and graph structure
                path_length = doc.metadata.get("path_length", 1)
                score = self.compute_graph_relevance(entity_uri, query_embedding, path_length)
                all_docs.append((doc, score))
        
        # Sort by score
        all_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k unique documents
        unique_docs = []
        seen_ids = set()
        for doc, score in all_docs:
            doc_id = doc.metadata.get("entity", doc.metadata.get("concept_id", None))
            if doc_id and doc_id not in seen_ids:
                unique_docs.append(doc)
                seen_ids.add(doc_id)
                if len(unique_docs) >= k:
                    break
        
        return unique_docs

    def compute_graph_relevance(self, entity_uri, query_embedding, graph_path_length):
        """Compute relevance score combining vector similarity and graph path length"""
        # Get entity details
        details = self.get_entity_details(entity_uri)
        
        # Create text representation of entity
        entity_text = ""
        for detail in details:
            if "label" in detail["predicate"].lower():
                entity_text += detail["object"] + " "
            if "type" in detail["predicate"].lower():
                entity_text += detail["object"].split("/")[-1] + " "
        
        # Compute vector similarity using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Embed the entity text
        entity_embedding = self.embeddings.embed_query(entity_text)
        
        # Reshape for sklearn
        query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)
        entity_embedding_reshaped = np.array(entity_embedding).reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(query_embedding_reshaped, entity_embedding_reshaped)[0][0]
        
        # Discount by path length (closer nodes get higher scores)
        path_discount = 1.0 / (1.0 + graph_path_length)
        
        # Final score combines vector similarity and graph proximity
        final_score = similarity * path_discount
        
        return final_score

    def graph_retrieval(self, query, initial_k=5, expansion_depth=2, max_nodes=15):
        """
        Graph-aware retrieval that combines vector search with graph traversal
        
        Args:
            query: User query
            initial_k: Number of initial nodes to retrieve
            expansion_depth: How many hops to traverse from seed nodes
            max_nodes: Maximum total nodes to return
            
        Returns:
            List of Document objects from graph traversal
        """
        # Step 1: Get seed nodes from vector search
        seed_documents = self.search(query, k=initial_k)
        
        # Extract entity URIs from seed documents
        seed_entities = set()
        for doc in seed_documents:
            if doc.metadata.get("source") == "rdf_data":
                entity_uri = doc.metadata.get("entity", "")
                if entity_uri:
                    seed_entities.add(entity_uri)
        
        # Step 2: Traverse the graph to find related entities
        all_entities = set(seed_entities)
        expanded_entities = set()
        
        # For each depth level
        for depth in range(expansion_depth):
            # Current frontier to expand from
            frontier = seed_entities if depth == 0 else expanded_entities
            expanded_entities = set()
            
            # For each entity in the frontier
            for entity_uri in frontier:
                # Get directly related entities
                query = f"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                
                SELECT ?related WHERE {{
                    # Outgoing relationships
                    {{ <{entity_uri}> ?p ?related . }}
                    UNION
                    # Incoming relationships
                    {{ ?related ?p <{entity_uri}> . }}
                    
                    # Filter for entities with labels only
                    ?related rdfs:label ?label .
                }}
                LIMIT 10
                """
                
                try:
                    self.sparql.setQuery(query)
                    results = self.sparql.query().convert()
                    
                    for result in results["results"]["bindings"]:
                        related_uri = result["related"]["value"]
                        if related_uri not in all_entities:
                            expanded_entities.add(related_uri)
                            all_entities.add(related_uri)
                            
                            # Stop if we've reached the maximum number of nodes
                            if len(all_entities) >= max_nodes:
                                break
                    
                    if len(all_entities) >= max_nodes:
                        break
                        
                except Exception as e:
                    logger.error(f"Error during graph traversal: {str(e)}")
            
            # If we've reached the maximum, stop traversal
            if len(all_entities) >= max_nodes:
                break
        
        # Step 3: Convert entities to documents
        graph_documents = []
        
        for entity_uri in all_entities:
            # Get entity details and format as document
            details = self.get_entity_details(entity_uri)
            
            # Get entity label
            entity_label = ""
            for detail in details:
                if "label" in detail["predicate"].lower():
                    entity_label = detail["object"]
                    break
            
            # Create text representation of entity
            text = f"Entity: {entity_label} ({entity_uri})\n"
            text += "Properties:\n"
            for detail in details[:10]:  # Limit to avoid huge documents
                pred = detail.get("predicateLabel", detail["predicate"])
                obj = detail.get("objectLabel", detail["object"])
                text += f"  - {pred}: {obj}\n"
            
            # Create document
            doc = Document(
                page_content=text, 
                metadata={
                    "entity": entity_uri, 
                    "label": entity_label,
                    "source": "rdf_data_graph"
                }
            )
            graph_documents.append(doc)
        
        return graph_documents


    def get_entity_label(self, entity_uri):
        """Get label for an entity URI"""
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?label WHERE {{
            <{entity_uri}> rdfs:label ?label .
        }}
        LIMIT 1
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            if results["results"]["bindings"]:
                return results["results"]["bindings"][0]["label"]["value"]
            return entity_uri.split("/")[-1]  # Fallback to last segment of URI
        except Exception:
            return entity_uri.split("/")[-1]  # Fallback to last segment of URI

    def extract_reasoning_path(self, entity_uri1, entity_uri2, max_depth=3):
        """Find reasoning paths between two entities in the graph"""
        # Use BFS to find paths
        visited = set()
        queue = [(entity_uri1, [])]
        
        while queue:
            current_uri, path = queue.pop(0)
            
            # If we reached the target
            if current_uri == entity_uri2:
                return path
                
            # If we've seen this node before or exceeded max depth
            if current_uri in visited or len(path) >= max_depth:
                continue
                
            visited.add(current_uri)
            
            # Query for neighbors
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?related ?predicate ?predicateLabel WHERE {{
                {{ <{current_uri}> ?predicate ?related . }}
                
                OPTIONAL {{ ?predicate rdfs:label ?predicateLabel }}
                ?related rdfs:label ?relatedLabel .
            }}
            LIMIT 20
            """
            
            try:
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                
                for result in results["results"]["bindings"]:
                    neighbor = result["related"]["value"]
                    predicate = result["predicate"]["value"]
                    predicate_label = result.get("predicateLabel", {"value": predicate})["value"]
                    
                    # Add to queue with extended path
                    new_path = path + [(current_uri, predicate_label, neighbor)]
                    queue.append((neighbor, new_path))
                    
            except Exception as e:
                logger.error(f"Error finding path: {str(e)}")
        
        # No path found
        return None

    def location_specific_graph_query(self, entity_uri):
        """
        Extract complete location hierarchy for an entity using graph traversal.
        This implements a targeted GraphRAG approach for location queries.
        
        Args:
            entity_uri: URI of the entity to get location for
            
        Returns:
            Dictionary with location information
        """
        logger.info(f"Extracting location hierarchy for: {entity_uri}")
        
        # First, find direct location relationship
        query = f"""
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?location ?locationLabel WHERE {{
            # Direct location relationship
            <{entity_uri}> crm:P55_has_current_location ?location .
            OPTIONAL {{ ?location rdfs:label ?locationLabel }}
        }}
        LIMIT 1
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            location_info = {
                "entity_uri": entity_uri,
                "entity_label": self.get_entity_label(entity_uri),
                "locations": []
            }
            
            # Process direct location
            if results["results"]["bindings"]:
                location_uri = results["results"]["bindings"][0]["location"]["value"]
                location_label = results["results"]["bindings"][0].get("locationLabel", {"value": location_uri.split("/")[-1]})["value"]
                
                location_entry = {
                    "uri": location_uri,
                    "label": location_label,
                    "level": 0,
                    "coordinates": self.get_coordinates(location_uri)
                }
                location_info["locations"].append(location_entry)
                
                # Now traverse up the location hierarchy using P89_falls_within
                self._traverse_location_hierarchy(location_uri, location_info, 1)
                
            return location_info
        
        except Exception as e:
            logger.error(f"Error extracting location hierarchy: {str(e)}")
            return {"entity_uri": entity_uri, "error": str(e)}

    def _traverse_location_hierarchy(self, location_uri, location_info, level, max_levels=5):
        """
        Recursively traverse the location hierarchy to build a complete path.
        
        Args:
            location_uri: Current location URI
            location_info: Dictionary to update with hierarchy info
            level: Current hierarchy level
            max_levels: Maximum levels to traverse
        """
        if level >= max_levels:
            return
        
        # Query for parent location
        query = f"""
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?parent ?parentLabel WHERE {{
            <{location_uri}> crm:P89_falls_within ?parent .
            OPTIONAL {{ ?parent rdfs:label ?parentLabel }}
        }}
        LIMIT 1
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            if results["results"]["bindings"]:
                parent_uri = results["results"]["bindings"][0]["parent"]["value"]
                parent_label = results["results"]["bindings"][0].get("parentLabel", {"value": parent_uri.split("/")[-1]})["value"]
                
                # Add parent to hierarchy
                parent_entry = {
                    "uri": parent_uri,
                    "label": parent_label,
                    "level": level,
                    "coordinates": self.get_coordinates(parent_uri)
                }
                location_info["locations"].append(parent_entry)
                
                # Continue traversing upward
                self._traverse_location_hierarchy(parent_uri, location_info, level + 1, max_levels)
        
        except Exception as e:
            logger.error(f"Error traversing location hierarchy: {str(e)}")

    def get_coordinates(self, location_uri):
        """
        Get coordinates for a location if available.
        
        Args:
            location_uri: URI of the location
            
        Returns:
            Dictionary with latitude and longitude or None
        """
        query = f"""
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        
        SELECT ?coordinates WHERE {{
            <{location_uri}> crm:P168_is_approximated_by ?coordinates .
        }}
        LIMIT 1
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            if results["results"]["bindings"]:
                wkt = results["results"]["bindings"][0]["coordinates"]["value"]
                
                # Parse WKT format
                if "POINT" in wkt:
                    # Format: POINT(lon lat)
                    coords = wkt.replace("POINT(", "").replace(")", "").split()
                    if len(coords) >= 2:
                        return {
                            "longitude": float(coords[0]),
                            "latitude": float(coords[1])
                        }
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting coordinates: {str(e)}")
            return None

    def build_relationship_cache(self):
        """
        Pre-compute and cache important relationships between entities.
        This greatly speeds up graph traversal during query time.
        
        Returns:
            Dictionary of cached relationships
        """
        logger.info("Building relationship cache...")
        
        # Initialize cache structure
        relationship_cache = {
            "location": {},  # Entity -> location hierarchy
            "temporal": {},  # Entity -> temporal information
            "iconographic": {}  # Entity -> iconography relationships
        }
        
        # Build location cache
        # Get all entities with location information
        query = """
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?entity ?entityLabel ?location ?locationLabel WHERE {
            ?entity crm:P55_has_current_location ?location .
            
            OPTIONAL { ?entity rdfs:label ?entityLabel }
            OPTIONAL { ?location rdfs:label ?locationLabel }
        }
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            # Process location relationships
            for result in results["results"]["bindings"]:
                entity_uri = result["entity"]["value"]
                entity_label = result.get("entityLabel", {"value": entity_uri.split("/")[-1]})["value"]
                location_uri = result["location"]["value"]
                location_label = result.get("locationLabel", {"value": location_uri.split("/")[-1]})["value"]
                
                # Get location hierarchy
                location_hierarchy = self._get_location_hierarchy(location_uri)
                
                # Store in cache
                relationship_cache["location"][entity_uri] = {
                    "entity_label": entity_label,
                    "direct_location": {
                        "uri": location_uri,
                        "label": location_label
                    },
                    "hierarchy": location_hierarchy
                }
            
            logger.info(f"Built location cache with {len(relationship_cache['location'])} entries")
            
            # Store cache to disk for persistence
            try:
                import json
                import os
                
                if not os.path.exists('cache'):
                    os.makedirs('cache')
                    
                with open('cache/relationship_cache.json', 'w') as f:
                    json.dump(relationship_cache, f)
                
                logger.info("Relationship cache saved to disk")
                
            except Exception as e:
                logger.error(f"Error saving relationship cache: {str(e)}")
            
            return relationship_cache
        
        except Exception as e:
            logger.error(f"Error building relationship cache: {str(e)}")
            return relationship_cache

    def _get_location_hierarchy(self, location_uri, max_levels=5):
        """
        Get the complete hierarchy for a location.
        
        Args:
            location_uri: URI of the location
            max_levels: Maximum levels to traverse
            
        Returns:
            List of dictionaries with location hierarchy
        """
        hierarchy = []
        current_uri = location_uri
        
        for level in range(max_levels):
            # Query for parent location
            query = f"""
            PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?parent ?parentLabel WHERE {{
                <{current_uri}> crm:P89_falls_within ?parent .
                OPTIONAL {{ ?parent rdfs:label ?parentLabel }}
            }}
            LIMIT 1
            """
            
            try:
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                
                if not results["results"]["bindings"]:
                    break
                    
                parent_uri = results["results"]["bindings"][0]["parent"]["value"]
                parent_label = results["results"]["bindings"][0].get("parentLabel", {"value": parent_uri.split("/")[-1]})["value"]
                
                # Add to hierarchy
                hierarchy.append({
                    "uri": parent_uri,
                    "label": parent_label,
                    "level": level + 1
                })
                
                # Move up the hierarchy
                current_uri = parent_uri
                
            except Exception as e:
                logger.error(f"Error getting location hierarchy: {str(e)}")
                break
        
        return hierarchy

    def load_relationship_cache(self):
        """
        Load pre-computed relationship cache from disk.
        
        Returns:
            Dictionary of cached relationships
        """
        try:
            import json
            import os
            
            cache_path = 'cache/relationship_cache.json'
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                    
                logger.info(f"Loaded relationship cache with {len(cache.get('location', {}))} location entries")
                return cache
            else:
                logger.info("No relationship cache found, building new cache...")
                return self.build_relationship_cache()
                
        except Exception as e:
            logger.error(f"Error loading relationship cache: {str(e)}")
            return {
                "location": {},
                "temporal": {},
                "iconographic": {}
            }

    def get_entity_location_from_cache(self, entity_uri):
        """
        Get location information for an entity from the cache.
        
        Args:
            entity_uri: URI of the entity
            
        Returns:
            Dictionary with location information or None
        """
        # Ensure cache is loaded
        if not hasattr(self, 'relationship_cache'):
            self.relationship_cache = self.load_relationship_cache()
        
        # Check if entity is in location cache
        if entity_uri in self.relationship_cache["location"]:
            return self.relationship_cache["location"][entity_uri]
        
        return None

    def _extract_entity_name(self, question):
        """Extract entity name from question more intelligently."""
        # Remove common question words and patterns
        cleaned_question = question.lower()
        for phrase in ["where is", "located", "location of", "where can i find"]:
            cleaned_question = cleaned_question.replace(phrase, "")
        
        # Remove question marks and trim
        cleaned_question = cleaned_question.replace("?", "").strip()
        
        # If the result seems too long, try to be smarter about extraction
        if len(cleaned_question.split()) > 3:
            # Try matching against known entities
            entities = self.get_all_entities()
            best_match = None
            best_score = 0
            
            for entity in entities:
                entity_label = entity["label"].lower()
                if entity_label in cleaned_question:
                    # Score by length of match
                    score = len(entity_label)
                    if score > best_score:
                        best_score = score
                        best_match = entity["label"]
            
            if best_match:
                return best_match
        
        return cleaned_question

    def traverse_graph_relationships(self, entity_uri, relationship_patterns=None, max_depth=3):
        """
        Generic graph traversal following CIDOC-CRM relationships.
        
        Args:
            entity_uri: Starting entity URI
            relationship_patterns: Optional list of relationship patterns to follow
            max_depth: Maximum traversal depth
            
        Returns:
            Dictionary with traversal results
        """
        # If no specific patterns provided, use common CIDOC-CRM relationships
        if not relationship_patterns:
            relationship_patterns = [
                "P55_has_current_location",  # Location relationships
                "P89_falls_within",
                "P108i_was_produced_by",     # Temporal relationships
                "P4_has_time-span",
                "P82_at_some_time_within",
                "K24_portray",               # Iconographic relationships
                "K17_has_attribute",
                "K14_symbolize"
            ]
        
        # Initialize result
        result = {
            "entity_uri": entity_uri,
            "entity_label": self.get_entity_label(entity_uri),
            "relationships": []
        }
        
        # Traverse graph using BFS
        visited = set([entity_uri])
        queue = [(entity_uri, 0)]  # (uri, depth)
        
        while queue:
            current_uri, depth = queue.pop(0)
            
            # Stop if we've reached max depth
            if depth >= max_depth:
                continue
            
            # Find relationships
            for pattern in relationship_patterns:
                # Forward traversal
                query = f"""
                PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
                PREFIX vir: <http://w3id.org/vir#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                
                SELECT ?predicate ?object ?objectLabel WHERE {{
                    <{current_uri}> ?predicate ?object .
                    OPTIONAL {{ ?object rdfs:label ?objectLabel }}
                    
                    # Match relationship pattern
                    FILTER(CONTAINS(STR(?predicate), "{pattern}"))
                }}
                LIMIT 5
                """
                
                try:
                    self.sparql.setQuery(query)
                    results = self.sparql.query().convert()
                    
                    for result_row in results["results"]["bindings"]:
                        predicate = result_row["predicate"]["value"]
                        object_uri = result_row["object"]["value"]
                        object_label = result_row.get("objectLabel", {"value": object_uri.split("/")[-1]})["value"]
                        
                        # Add relationship
                        result["relationships"].append({
                            "source_uri": current_uri,
                            "source_label": self.get_entity_label(current_uri),
                            "predicate": predicate,
                            "predicate_name": predicate.split("/")[-1],
                            "target_uri": object_uri,
                            "target_label": object_label,
                            "depth": depth
                        })
                        
                        # Continue traversal if not visited
                        if object_uri not in visited:
                            visited.add(object_uri)
                            queue.append((object_uri, depth + 1))
                
                except Exception as e:
                    logger.error(f"Error in graph traversal: {str(e)}")
        
        return result

    def enhance_retrieval_with_graph(self, question, base_docs, k=5):
        """
        Enhance retrieval with graph context from key entities.
        
        Args:
            question: User question
            base_docs: Base retrieval documents
            k: Number of documents to return
            
        Returns:
            Enhanced list of documents
        """
        # Extract entities from base documents
        entities = set()
        for doc in base_docs:
            if doc.metadata.get("source") == "rdf_data":
                entity_uri = doc.metadata.get("entity", "")
                if entity_uri:
                    entities.add(entity_uri)
        
        # For each entity, get graph context
        graph_docs = []
        for entity_uri in entities:
            # Identify relationship patterns to follow based on question
            if "where" in question.lower() or "located" in question.lower():
                patterns = ["P55_has_current_location", "P89_falls_within"]
            elif "when" in question.lower() or "date" in question.lower():
                patterns = ["P108i_was_produced_by", "P4_has_time-span", "P82_at_some_time_within"]
            elif "depict" in question.lower() or "represent" in question.lower():
                patterns = ["K24_portray", "K17_has_attribute", "K14_symbolize"]
            else:
                patterns = None  # Use default patterns
            
            # Get graph context
            graph_result = self.traverse_graph_relationships(entity_uri, patterns, max_depth=2)
            
            # Convert to document
            if graph_result["relationships"]:
                # Format graph context as text
                text = f"Entity: {graph_result['entity_label']} ({graph_result['entity_uri']})\n\n"
                text += "Relationships:\n"
                
                for rel in graph_result["relationships"]:
                    depth_indent = "  " * rel["depth"]
                    text += f"{depth_indent}- {rel['source_label']} → {rel['predicate_name']} → {rel['target_label']}\n"
                
                # Create document
                doc = Document(
                    page_content=text,
                    metadata={
                        "entity": entity_uri,
                        "label": graph_result["entity_label"],
                        "source": "graph_context"
                    }
                )
                graph_docs.append(doc)
        
        # Combine with base docs and return
        all_docs = base_docs + graph_docs
        
        # Remove duplicates by entity
        unique_docs = []
        seen_entities = set()
        
        for doc in all_docs:
            entity = doc.metadata.get("entity")
            if entity is None or entity not in seen_entities:
                if entity is not None:
                    seen_entities.add(entity)
                unique_docs.append(doc)
        
        return unique_docs[:k]

    def _format_location_sources(self, locations):
        """
        Format location hierarchy as sources for the API response.
        
        Args:
            locations: List of location dictionaries
            
        Returns:
            List of formatted source dictionaries
        """
        sources = []
        
        for i, location in enumerate(locations):
            sources.append({
                "id": i,
                "entity_uri": location.get("uri", ""),
                "entity_label": location.get("label", ""),
                "level": location.get("level", 0),
                "type": "location_hierarchy"
            })
        
        return sources

    def get_location_specific_prompt():
        """
        Create a specialized prompt template for location questions.
        
        Returns:
            PromptTemplate for location questions
        """
        location_prompt_template = """You are an expert in Byzantine art and architecture with deep knowledge about CIDOC-CRM and VIR ontologies.
        
        When answering location questions, pay special attention to these relationship types:
        1. P55_has_current_location - Indicates where an entity is directly located
        2. P89_falls_within - Indicates that a place is contained within another place
        3. P168_is_approximated_by - Often contains coordinate information
        
        The retrieved information includes a location hierarchy that shows exactly where the entity is located,
        from the immediate location to broader geographical regions.
        
        Retrieved location hierarchy:
        {context}
        
        User question: {question}
        
        Provide a clear, concise answer that:
        1. Directly states where the entity is located
        2. Includes the complete geographical context (e.g., "X is in Y, which is in Z")
        3. Mentions coordinates if available
        4. Uses natural language rather than ontology terminology
        """
        
        return PromptTemplate.from_template(location_prompt_template)

    def create_location_context(self, entity_name, location_info):
        """
        Create a structured context string for location information.
        
        Args:
            entity_name: Name of the entity
            location_info: Dictionary with location information
            
        Returns:
            Formatted context string
        """
        context = f"Entity: {entity_name}\n\nLocation hierarchy:\n"
        
        if not location_info.get("locations"):
            return context + "No location information found."
        
        # Sort locations by level
        locations = sorted(location_info["locations"], key=lambda x: x.get("level", 0))
        
        for i, location in enumerate(locations):
            level_str = "  " * i  # Indentation to show hierarchy
            context += f"{level_str}Level {i}: {location.get('label', 'Unknown')}"
            
            # Add coordinates if available
            if location.get("coordinates"):
                coords = location["coordinates"]
                context += f" (Latitude: {coords.get('latitude')}, Longitude: {coords.get('longitude')})"
            
            context += "\n"
        
        return context

    def _add_entity_to_graph(self, entity_uri, graph, processed_entities, max_depth, current_depth):
        """
        Recursively add an entity and its relationships to the graph.
        
        Args:
            entity_uri: URI of the entity to add
            graph: Graph dictionary to update
            processed_entities: Set of already processed entities
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
        """
        # Stop if we've already processed this entity or reached max depth
        if entity_uri in processed_entities or current_depth >= max_depth:
            return
        
        # Mark as processed
        processed_entities.add(entity_uri)
        
        # Get entity details
        entity_label = self.get_entity_label(entity_uri)
        
        # Add to nodes
        graph["nodes"].append({
            "id": entity_uri,
            "label": entity_label,
            "depth": current_depth
        })
        
        # Get outgoing relationships
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?predicate ?predicateLabel ?object ?objectLabel WHERE {{
            <{entity_uri}> ?predicate ?object .
            OPTIONAL {{ ?predicate rdfs:label ?predicateLabel }}
            OPTIONAL {{ ?object rdfs:label ?objectLabel }}
            
            # Filter for meaningful predicates and objects
            FILTER(STRSTARTS(STR(?predicate), "http://www.cidoc-crm.org/cidoc-crm/") || 
                   STRSTARTS(STR(?predicate), "http://w3id.org/vir#"))
            FILTER(STRSTARTS(STR(?object), "http://"))
        }}
        LIMIT 50
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            for result in results["results"]["bindings"]:
                predicate = result["predicate"]["value"]
                predicate_label = result.get("predicateLabel", {"value": predicate.split("/")[-1]})["value"]
                object_uri = result["object"]["value"]
                object_label = result.get("objectLabel", {"value": object_uri.split("/")[-1]})["value"]
                
                # Add edge
                graph["edges"].append({
                    "source": entity_uri,
                    "target": object_uri,
                    "label": predicate_label,
                    "predicate": predicate
                })
                
                # Recursively add connected entity
                if current_depth + 1 < max_depth:
                    self._add_entity_to_graph(object_uri, graph, processed_entities, max_depth, current_depth + 1)
        
        except Exception as e:
            logger.error(f"Error processing entity relationships: {str(e)}")

    def explicit_path_finding(self, question, entity_name, relationship_type=None):
        """
        Explicitly find paths in the graph based on question type.
        
        Args:
            question: User's question
            entity_name: Name of the entity to start from
            relationship_type: Type of relationship to traverse (optional)
            
        Returns:
            Dictionary with path information
        """
        # Step 1: Identify the starting entity
        entity_uri = self._find_entity_by_name(entity_name)
        if not entity_uri:
            return {
                "success": False,
                "error": f"Could not find entity named '{entity_name}'"
            }
        
        # Step 2: Determine relationship type if not provided
        if not relationship_type:
            relationship_type = self._infer_relationship_from_question(question)
        
        # Step 3: Find paths based on relationship type
        if relationship_type == "location":
            return self._find_location_path(entity_uri)
        elif relationship_type == "temporal":
            return self._find_temporal_path(entity_uri)
        elif relationship_type == "iconographic":
            return self._find_iconographic_path(entity_uri)
        else:
            # Generic path finding
            return self._find_generic_paths(entity_uri)

    def _find_entity_by_name(self, entity_name):
        """
        Find entity URI by name/label.
        
        Args:
            entity_name: Name of the entity
            
        Returns:
            Entity URI or None
        """
        query = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?entity WHERE {{
            ?entity rdfs:label ?label .
            FILTER(CONTAINS(LCASE(?label), LCASE("{entity_name}")))
        }}
        LIMIT 1
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            if results["results"]["bindings"]:
                return results["results"]["bindings"][0]["entity"]["value"]
            return None
        except Exception as e:
            logger.error(f"Error finding entity by name: {str(e)}")
            return None

    def _infer_relationship_from_question(self, question):
        """
        Infer relationship type from question text.
        
        Args:
            question: User's question
            
        Returns:
            Relationship type string
        """
        question_lower = question.lower()
        
        # Location patterns
        if any(pattern in question_lower for pattern in ["where", "located", "location", "place"]):
            return "location"
        
        # Temporal patterns
        if any(pattern in question_lower for pattern in ["when", "date", "year", "period", "century"]):
            return "temporal"
        
        # Iconographic patterns
        if any(pattern in question_lower for pattern in ["depict", "represent", "symbol", "iconography"]):
            return "iconographic"
        
        # Default
        return "generic"


    def _find_generic_paths(self, entity_uri, max_depth=2):
        """
        Find generic paths starting from an entity.
        
        Args:
            entity_uri: URI of the entity
            max_depth: Maximum path depth
            
        Returns:
            Dictionary with path information
        """
        # Initialize result
        result = {
            "success": True,
            "entity_uri": entity_uri,
            "entity_label": self.get_entity_label(entity_uri),
            "path_type": "generic",
            "path": []
        }
        
        # Process paths using BFS
        visited = set([entity_uri])
        queue = [(entity_uri, 0)]  # (uri, depth)
        
        while queue:
            current_uri, depth = queue.pop(0)
            
            # Stop if we've reached max depth
            if depth >= max_depth:
                continue
            
            # Get current entity label
            current_label = self.get_entity_label(current_uri)
            
            # Find outgoing relationships
            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?predicate ?predicateLabel ?object ?objectLabel WHERE {{
                <{current_uri}> ?predicate ?object .
                OPTIONAL {{ ?predicate rdfs:label ?predicateLabel }}
                OPTIONAL {{ ?object rdfs:label ?objectLabel }}
                
                # Filter for meaningful relationships
                FILTER(STRSTARTS(STR(?predicate), "http://www.cidoc-crm.org/cidoc-crm/") || 
                       STRSTARTS(STR(?predicate), "http://w3id.org/vir#"))
                FILTER(STRSTARTS(STR(?object), "http://"))
            }}
            LIMIT 20
            """
            
            try:
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                
                for result_row in results["results"]["bindings"]:
                    predicate = result_row["predicate"]["value"]
                    predicate_label = result_row.get("predicateLabel", {"value": predicate.split("/")[-1]})["value"]
                    object_uri = result_row["object"]["value"]
                    object_label = result_row.get("objectLabel", {"value": object_uri.split("/")[-1]})["value"]
                    
                    # Skip if we've already visited this object
                    if object_uri in visited:
                        continue
                    
                    visited.add(object_uri)
                    
                    # Add to path
                    result["path"].append({
                        "source_uri": current_uri,
                        "source_label": current_label,
                        "relationship": predicate.split("/")[-1],
                        "relationship_label": predicate_label,
                        "target_uri": object_uri,
                        "target_label": object_label,
                        "level": depth
                    })
                    
                    # Continue BFS
                    queue.append((object_uri, depth + 1))
            
            except Exception as e:
                logger.error(f"Error finding generic paths: {str(e)}")
        
        return result

    def format_direct_query_answer(self, query_results, llm=None):
        """
        Format direct query results into a natural language answer.
        
        Args:
            query_results: Results from execute_direct_query
            llm: Optional LLM for more sophisticated formatting (will use OpenAI if provided)
            
        Returns:
            Dictionary with formatted answer
        """
        if not query_results["success"]:
            return {
                "answer": f"I'm sorry, I couldn't find information about {query_results.get('entity_name', 'that entity')}. {query_results.get('error', '')}",
                "sources": []
            }
        
        query_type = query_results["query_type"]
        entity_name = query_results["entity_name"]
        results = query_results["results"]
        
        if not results:
            return {
                "answer": f"I found {entity_name} in the knowledge base, but couldn't find any {query_type} information for it.",
                "sources": []
            }
        
        # If we have an LLM, use it for better formatting
        if llm:
            # Create a context from the results
            context = f"Query type: {query_type}\nEntity: {entity_name}\n\nResults:\n"
            for i, result in enumerate(results):
                context += f"Result {i+1}:\n"
                for key, value in result.items():
                    context += f"  {key}: {value}\n"
                context += "\n"
            
            # Create a prompt
            prompt = f"""You are an expert in Byzantine art with deep knowledge of CIDOC-CRM and VIR ontologies.
            Based on the following query results, provide a clear and concise answer about {entity_name}.
            
            {context}
            
            Format your answer focusing on the {query_type} information, using natural language that avoids ontology terminology.
            Make the answer direct, succinct, and informative, based solely on the provided data.
            """
            
            # Get response from LLM
            try:
                response = llm.invoke(prompt)
                answer = response.content
                
                # Create sources
                sources = []
                for i, result in enumerate(results):
                    sources.append({
                        "id": i,
                        "entity_uri": result.get("entity", ""),
                        "entity_label": result.get("entityLabel", entity_name),
                        "type": f"direct_query_{query_type}"
                    })
                
                return {
                    "answer": answer,
                    "sources": sources
                }
            except Exception as e:
                logger.error(f"Error using LLM for formatting: {str(e)}")
                # Fall back to rule-based formatting
        
        # Rule-based formatting for each query type
        if query_type == "location":
            # Extract key information
            entity_label = results[0].get("entityLabel", entity_name)
            location_label = results[0].get("locationLabel", "unknown location")
            parent_label = None
            coordinates = None
            
            for result in results:
                if "parentLabel" in result:
                    parent_label = result["parentLabel"]
                if "coordinates" in result:
                    coordinates = result["coordinates"]
            
            # Format answer
            answer = f"{entity_label} is located at {location_label}"
            if parent_label:
                answer += f", which is within {parent_label}"
            if coordinates:
                # Try to extract coordinates from WKT format
                try:
                    if "POINT" in coordinates:
                        coords = coordinates.replace("POINT(", "").replace(")", "").split()
                        if len(coords) >= 2:
                            lat = coords[1]
                            lon = coords[0]
                            answer += f". The coordinates are latitude {lat}, longitude {lon}"
                except:
                    pass
            
            answer += "."
            
        elif query_type == "temporal":
            # Extract key information
            entity_label = results[0].get("entityLabel", entity_name)
            date_label = results[0].get("dateLabel", results[0].get("date", "unknown date"))
            
            # Format answer
            answer = f"{entity_label} was created in {date_label}."
            
        elif query_type == "iconographic":
            # Extract key information
            entity_label = results[0].get("entityLabel", entity_name)
            characters = []
            attributes = []
            
            for result in results:
                if "characterLabel" in result and result["characterLabel"] not in characters:
                    characters.append(result["characterLabel"])
                if "attributeLabel" in result and result["attributeLabel"] not in attributes:
                    attributes.append(result["attributeLabel"])
            
            # Format answer
            answer = f"{entity_label} "
            
            if characters:
                if len(characters) == 1:
                    answer += f"depicts {characters[0]}"
                else:
                    answer += f"depicts {', '.join(characters[:-1])} and {characters[-1]}"
            
            if attributes:
                if characters:
                    answer += " with "
                else:
                    answer += "has "
                    
                if len(attributes) == 1:
                    answer += f"the attribute {attributes[0]}"
                else:
                    answer += f"the attributes {', '.join(attributes[:-1])} and {attributes[-1]}"
            
            answer += "."
            
        else:
            # Generic formatting for other query types
            entity_label = results[0].get("entityLabel", entity_name)
            answer = f"Information about {entity_label}:\n\n"
            
            # Group results by key types
            for result in results:
                for key, value in result.items():
                    if key not in ["entity", "entityLabel"]:
                        answer += f"- {key}: {value}\n"
        
        # Create sources
        sources = []
        for i, result in enumerate(results):
            sources.append({
                "id": i,
                "entity_uri": result.get("entity", ""),
                "entity_label": result.get("entityLabel", entity_name),
                "type": f"direct_query_{query_type}"
            })
        
        return {
            "answer": answer,
            "sources": sources
        }

    def smart_query_routing(self, question):
        """
        Use a smarter approach to routing questions based on both pattern matching and LLM analysis.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (query_type, entity_name, confidence)
        """
        # First, try rule-based pattern matching
        analysis = self.analyze_question_type(question)
        
        # If high confidence match, use it directly
        if analysis["confidence"] >= 0.7 and analysis.get("entity_name"):
            return (analysis["type"], analysis["entity_name"], analysis["confidence"])
        
        # Otherwise, use LLM for more sophisticated analysis
        prompt = f"""Analyze the following question about Byzantine art and determine:
        1. The primary question type (location, temporal, iconographic, or general)
        2. The main entity being asked about
        
        Question: {question}
        
        Respond with a JSON object with the following structure:
        {{
            "query_type": "location|temporal|iconographic|general",
            "entity_name": "extracted entity name",
            "confidence": 0.0-1.0 (your confidence in this classification)
        }}
        """
        
        try:
            # Set up OpenAI LLM with very low temperature for classification
            llm = ChatOpenAI(
                model=self.openai_model,
                temperature=0.1,
                openai_api_key=self.openai_api_key
            )
            
            # Get response
            response = llm.invoke(prompt)
            
            # Parse JSON response
            import json
            llm_analysis = json.loads(response.content)
            
            # Return the LLM's analysis
            return (
                llm_analysis["query_type"],
                llm_analysis["entity_name"],
                llm_analysis["confidence"]
            )
        except Exception as e:
            logger.error(f"Error using LLM for question analysis: {str(e)}")
            # Fall back to rule-based result
            return (analysis["type"], analysis.get("entity_name", ""), analysis["confidence"])

    def optimized_answer_question(self, question):
        """
        Answer a question using optimized graph-aware techniques.
        This method combines direct SPARQL queries, graph traversal, and RAG as appropriate.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Answering question with optimized approach: '{question}'")
        
        # Step 1: Analyze the question to determine type and entity
        query_type, entity_name, confidence = self.smart_query_routing(question)
        logger.info(f"Query analysis: type={query_type}, entity={entity_name}, confidence={confidence}")
        
        # If we have a specific entity and high confidence, use direct query
        if entity_name and confidence >= 0.7 and query_type in self.get_query_patterns():
            # Use direct SPARQL query for this question type
            logger.info(f"Using direct query for {query_type} question about {entity_name}")
            query_results = self.execute_direct_query(query_type, entity_name)
            
            if query_results["success"] and query_results["results"]:
                # Set up LLM for better formatting
                llm = ChatOpenAI(
                    model=self.openai_model,
                    temperature=0.7,
                    openai_api_key=self.openai_api_key
                )
                
                # Format answer using LLM
                return self.format_direct_query_answer(query_results, llm)
        
        # If direct query wasn't used or didn't work, try explicit path finding
        if entity_name and query_type in ["location", "temporal", "iconographic"]:
            logger.info(f"Using explicit path finding for {query_type} question about {entity_name}")
            path_results = self.explicit_path_finding(question, entity_name, query_type)
            
            if path_results["success"]:
                # Create context from path results
                context = self.format_path_results_as_context(path_results)
                
                # Set up LLM
                llm = ChatOpenAI(
                    model=self.openai_model,
                    temperature=0.7,
                    openai_api_key=self.openai_api_key
                )
                
                # Create specialized prompt
                if query_type == "location":
                    prompt_template = self.get_location_specific_prompt()
                elif query_type == "temporal":
                    prompt_template = self.get_temporal_specific_prompt()
                elif query_type == "iconographic":
                    prompt_template = self.get_iconographic_specific_prompt()
                else:
                    prompt_template = self.get_generic_prompt()
                
                # Get answer
                response = llm.invoke(
                    prompt_template.format(context=context, question=question)
                )
                
                # Format sources from path
                sources = []
                for i, step in enumerate(path_results.get("path", [])):
                    sources.append({
                        "id": i,
                        "entity_uri": step.get("source_uri", ""),
                        "entity_label": step.get("source_label", ""),
                        "relationship": step.get("relationship", ""),
                        "target_uri": step.get("target_uri", ""),
                        "target_label": step.get("target_label", ""),
                        "type": f"graph_path_{query_type}"
                    })
                
                return {
                    "answer": response.content,
                    "sources": sources
                }
        
        # Fall back to standard RAG approach
        logger.info("Falling back to standard RAG approach")
        return self.answer_question_with_graph(question)

    def format_path_results_as_context(self, path_results):
        """
        Format path results into a context string for the LLM.
        
        Args:
            path_results: Results from explicit_path_finding
            
        Returns:
            Formatted context string
        """
        context = f"Entity: {path_results.get('entity_label', 'Unknown')}\n\n"
        
        if path_results.get("path_type"):
            context += f"Relationship type: {path_results['path_type']}\n\n"
        
        if not path_results.get("path"):
            return context + "No relationships found."
        
        context += "Relationship path:\n"
        
        # Sort path by level for hierarchical display
        path = sorted(path_results["path"], key=lambda x: x.get("level", 0))
        
        for i, step in enumerate(path):
            indent = "  " * step.get("level", 0)
            context += f"{indent}{i+1}. {step.get('source_label', 'Unknown')} {step.get('relationship_label', 'is related to')} {step.get('target_label', 'Unknown')}\n"
        
        return context

    def get_temporal_specific_prompt():
        """
        Create a specialized prompt template for temporal questions.
        
        Returns:
            PromptTemplate for temporal questions
        """
        temporal_prompt_template = """You are an expert in Byzantine art and architecture with deep knowledge about CIDOC-CRM and VIR ontologies.
        
        When answering temporal questions, pay special attention to these relationship types:
        1. P108i_was_produced_by - Links an entity to its production event
        2. P4_has_time-span - Links a production event to a time span
        3. P82_at_some_time_within - Links a time span to a specific date or period
        
        The retrieved information includes a temporal path that shows the creation time of the entity.
        
        Retrieved temporal information:
        {context}
        
        User question: {question}
        
        Provide a clear, concise answer that:
        1. Directly states when the entity was created
        2. Places it in historical context if possible
        3. Uses natural language rather than ontology terminology
        """
        
        return PromptTemplate.from_template(temporal_prompt_template)

    def get_iconographic_specific_prompt():
        """
        Create a specialized prompt template for iconographic questions.
        
        Returns:
            PromptTemplate for iconographic questions
        """
        iconographic_prompt_template = """You are an expert in Byzantine art and architecture with deep knowledge about CIDOC-CRM and VIR ontologies.
        
        When answering iconographic questions, pay special attention to these relationship types:
        1. K24_portray - Indicates that a representation depicts a character
        2. K17_has_attribute - Indicates visual attributes of a representation
        3. K14_symbolize - Indicates the symbolic meaning of an attribute
        
        The retrieved information includes iconographic details about what the entity depicts or represents.
        
        Retrieved iconographic information:
        {context}
        
        User question: {question}
        
        Provide a clear, concise answer that:
        1. Describes what the entity depicts or represents
        2. Explains any visual attributes and their symbolic meanings
        3. Uses natural language rather than ontology terminology
        """
        
        return PromptTemplate.from_template(iconographic_prompt_template)

    def get_generic_prompt():
        """
        Create a generic prompt template for other question types.
        
        Returns:
            PromptTemplate for generic questions
        """
        generic_prompt_template = """You are an expert in Byzantine art and architecture with deep knowledge about CIDOC-CRM and VIR ontologies.
        
        The retrieved information includes details about the entity in question.
        
        Retrieved information:
        {context}
        
        User question: {question}
        
        Provide a clear, concise answer that:
        1. Directly addresses the question
        2. Is based solely on the information provided
        3. Uses natural language rather than ontology terminology
        """
        
        return PromptTemplate.from_template(generic_prompt_template)

    def answer_location_question_with_llm(self, entity_name, location_info):
        """
        Use a specialized prompt with the LLM to answer location questions.
        
        Args:
            entity_name: Name of the entity
            location_info: Dictionary with location information
            
        Returns:
            Dictionary with answer and sources
        """
        # Create a specialized prompt
        location_prompt = get_location_specific_prompt()
        
        # Create context from location information
        context = self.create_location_context(entity_name, location_info)
        
        # Create a simple question
        question = f"Where is {entity_name} located?"
        
        # Set up OpenAI LLM
        llm = ChatOpenAI(
            model=self.openai_model,
            temperature=0.3,  # Lower temperature for more factual responses
            openai_api_key=self.openai_api_key
        )
        
        # Get answer
        result = llm.invoke(
            location_prompt.format(context=context, question=question)
        )
        
        # Extract answer text
        answer = result.content
        
        return {
            "answer": answer,
            "locations": location_info.get("locations", [])
        }


    def test_connection(self):
        """
        Test connection to Fuseki endpoint.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            query = """
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o
            } LIMIT 1
            """
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            logger.info("Successfully connected to Fuseki endpoint")
            return True
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            return False

    def get_all_entities(self):
        """
        Get all labeled entities from the SPARQL endpoint.
        
        Returns:
            List of dictionaries with entity information
        """
        query = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?entity ?label WHERE {
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
                
            logger.info(f"Retrieved {len(entities)} entities with labels")
            return entities
        except Exception as e:
            logger.error(f"Error fetching entities: {str(e)}")
            return []

    def setup_chat_chain(self):
        """
        Set up a conversational chain using OpenAI and both vector store retrievers.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        logger.info("Setting up conversational chain with OpenAI...")
        
        if not self.ensure_vectorstores():
            logger.error("Cannot set up chat chain without at least the RDF vector store")
            return False
        
        try:
            # Set up OpenAI LLM instead of Ollama
            llm = ChatOpenAI(
                model=self.openai_model,
                temperature=self.temperature,
                openai_api_key=self.openai_api_key
            )
            
            # Set up retrievers for both vector stores
            rdf_retriever = self.rdf_vectorstore.as_retriever(search_kwargs={"k": 5})
            
            # If ontology vector store is available, create a merger retriever
            if self.ontology_vectorstore:
                ontology_retriever = self.ontology_vectorstore.as_retriever(search_kwargs={"k": 3})
                
                # Merge retrievers with weight bias toward RDF data
                # This is important as we want factual data to be prioritized over concept definitions
                retriever = MergerRetriever(
                    retrievers=[rdf_retriever, ontology_retriever],
                    weights=[0.7, 0.3]  # 70% weight to RDF data, 30% to ontology
                )
            else:
                # Use only RDF retriever if ontology is not available
                retriever = rdf_retriever
            
            # Create enhanced Byzantine art expert prompt that understands ontology
            condense_question_prompt = PromptTemplate.from_template(
                """You are an expert in Byzantine art and architecture who helps answer questions based on specific knowledge.
                You understand the CIDOC-CRM ontology and can interpret relationships between entities in that framework.
                
                Given the following conversation and a follow up question, rephrase the follow up question to be a standalone 
                question that captures all relevant context from the conversation.
                
                Chat History:
                {chat_history}
                
                Follow Up Input: {question}
                Standalone question:"""
            )
            
            qa_prompt = PromptTemplate.from_template(
            """You are an expert in Byzantine art and architecture with deep knowledge about CIDOC-CRM and VIR ontologies.

            When interpreting CIDOC-CRM graph relationships, pay attention to these key patterns:

            1. Location relationships:
               - P55_has_current_location means "is located at"
               - P89_falls_within means "is contained within" or "is part of"

            2. Temporal relationships:
               - P108i_was_produced_by connects an object to its production event
               - P4_has_time-span connects an event to a time period
               - P82_at_some_time_within connects a time span to a specific date

            3. Iconographic relationships:
               - K24_portray means "depicts" or "represents"
               - K17_has_attribute means "has the visual attribute"
               - K14_symbolize means "symbolizes" or "represents symbolically"

            Look for graph relationship patterns in the context that show paths between concepts.
            When you see a relationship like "Entity A → P89_falls_within → Entity B", interpret it as "Entity A is contained within Entity B".

            Retrieved information:
            {context}

            User question: {question}

            Provide a comprehensive answer that:
            1. Directly answers the question using the relevant graph relationships
            2. Uses natural language to explain relationships instead of ontology terminology
            3. Is accurate to the information in the knowledge base
            """
        )

            
            # Create a memory buffer to store the conversation history
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                input_key="question"  
            )
            
            # Create the conversational chain
            self.chat_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                condense_question_prompt=condense_question_prompt,
                combine_docs_chain_kwargs={"prompt": qa_prompt}
            )
            
            logger.info("Conversational chain set up successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting up chat chain: {str(e)}")
            return False