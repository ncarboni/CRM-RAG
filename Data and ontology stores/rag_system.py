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

    def add_geographic_context(self):
        """Process geographic relationships between entities"""
        logger.info("Building geographic context index...")
        
        # Get all locations
        query = """
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?place ?placeLabel ?parent ?parentLabel WHERE {
            ?place a crm:E53_Place ;
                   rdfs:label ?placeLabel .
            OPTIONAL {
                ?place crm:P89_falls_within ?parent .
                ?parent rdfs:label ?parentLabel .
            }
        }
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            # Build geographic hierarchy
            places = {}
            for result in results["results"]["bindings"]:
                place_uri = result["place"]["value"]
                place_label = result["placeLabel"]["value"]
                
                if place_uri not in places:
                    places[place_uri] = {
                        "uri": place_uri,
                        "label": place_label,
                        "parent": None,
                        "children": []
                    }
                    
                if "parent" in result:
                    parent_uri = result["parent"]["value"]
                    parent_label = result["parentLabel"]["value"]
                    
                    places[place_uri]["parent"] = parent_uri
                    
                    if parent_uri not in places:
                        places[parent_uri] = {
                            "uri": parent_uri,
                            "label": parent_label,
                            "parent": None,
                            "children": []
                        }
                        
                    places[parent_uri]["children"].append(place_uri)
            
            # Add this context to vector store documents
            geo_documents = []
            for entity_uri, place_info in places.items():
                # Create geographic context text
                context_text = f"Location: {place_info['label']}\n"
                
                # Add parent info
                if place_info["parent"]:
                    parent = places.get(place_info["parent"])
                    if parent:
                        context_text += f"Located within: {parent['label']}\n"
                        
                        # Add grandparent
                        if parent["parent"] and parent["parent"] in places:
                            grandparent = places.get(parent["parent"])
                            context_text += f"Region: {grandparent['label']}\n"
                
                # Add children info
                if place_info["children"]:
                    child_labels = [places[child]["label"] for child in place_info["children"] if child in places]
                    if child_labels:
                        context_text += f"Contains: {', '.join(child_labels)}\n"
                
                # Create document for indexing
                doc = Document(
                    page_content=context_text,
                    metadata={
                        "entity": entity_uri,
                        "label": place_info["label"],
                        "type": "E53_Place",
                        "source": "rdf_data_geographic"
                    }
                )
                geo_documents.append(doc)
                
            logger.info(f"Created {len(geo_documents)} geographic context documents")
            
            # Create/update geographic context vector store
            if geo_documents:
                geo_vectorstore = FAISS.from_documents(geo_documents, self.embeddings)
                
                # Save for future use
                if not os.path.exists('geo_index'):
                    os.makedirs('geo_index')
                geo_vectorstore.save_local('geo_index')
                logger.info("Geographic context vector store created and saved")
                
                return geo_vectorstore
            return None
        except Exception as e:
            logger.error(f"Error building geographic context: {str(e)}")
            return None

    def add_temporal_context(self):
        """Process temporal relationships and historical periods"""
        logger.info("Building temporal context index...")
        
        # Define important Byzantine historical periods
        byzantine_periods = [
            {"name": "Early Byzantine", "start": 330, "end": 650, 
             "description": "Period from Constantine to the Arab conquests"},
            {"name": "Middle Byzantine", "start": 650, "end": 1204, 
             "description": "From the Arab conquests to the Fourth Crusade"},
            {"name": "Late Byzantine", "start": 1204, "end": 1453, 
             "description": "From the Fourth Crusade to the Fall of Constantinople"},
            {"name": "Post-Byzantine", "start": 1453, "end": 1800, 
             "description": "After the Fall of Constantinople"}
        ]
        
        # Get all entities with temporal properties
        query = """
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?entity ?entityLabel ?date WHERE {
            ?entity crm:P108i_was_produced_by ?production .
            OPTIONAL { ?entity rdfs:label ?entityLabel }
            OPTIONAL {
                ?production crm:P4_has_time-span ?timespan .
                ?timespan crm:P82_at_some_time_within ?date .
            }
        }
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            # Process and add temporal context
            temporal_documents = []
            for result in results["results"]["bindings"]:
                if "date" in result:
                    date_value = result["date"]["value"]
                    entity_uri = result["entity"]["value"]
                    entity_label = result.get("entityLabel", {"value": entity_uri})["value"]
                    
                    # Parse date (could be year, range, etc.)
                    # Simplified example assumes year only
                    try:
                        year = int(date_value)
                        
                        # Find matching period
                        period = next((p for p in byzantine_periods if p["start"] <= year <= p["end"]), None)
                        
                        if period:
                            # Create temporal context
                            context_text = f"Entity: {entity_label}\n"
                            context_text += f"Created in: {year} ({period['name']} period)\n"
                            context_text += f"Historical context: {period['description']}\n"
                            
                            # Create document for the vector store
                            doc = Document(
                                page_content=context_text,
                                metadata={
                                    "entity": entity_uri,
                                    "label": entity_label,
                                    "year": year,
                                    "period": period["name"],
                                    "source": "rdf_data_temporal"
                                }
                            )
                            temporal_documents.append(doc)
                    except ValueError:
                        # Handle non-numeric dates
                        pass
            
            logger.info(f"Created {len(temporal_documents)} temporal context documents")
            
            # Create/update temporal context vector store
            if temporal_documents:
                temp_vectorstore = FAISS.from_documents(temporal_documents, self.embeddings)
                
                # Save for future use
                if not os.path.exists('temporal_index'):
                    os.makedirs('temporal_index')
                temp_vectorstore.save_local('temporal_index')
                logger.info("Temporal context vector store created and saved")
                
                return temp_vectorstore
            return None
        except Exception as e:
            logger.error(f"Error building temporal context: {str(e)}")
            return None

    def add_iconographic_context(self):
        """Process iconographic themes and build connections between related imagery"""
        logger.info("Building iconographic context index...")
        
        # Query for iconographic representation
        query = """
        PREFIX vir: <http://w3id.org/vir#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?representation ?representationLabel ?character ?characterLabel WHERE {
            ?representation a vir:IC9_Representation ;
                           rdfs:label ?representationLabel .
            OPTIONAL {
                ?representation vir:K24_portray ?character .
                ?character rdfs:label ?characterLabel .
            }
        }
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            # Process iconographic representations
            representations = {}
            for result in results["results"]["bindings"]:
                rep_uri = result["representation"]["value"]
                rep_label = result["representationLabel"]["value"]
                
                if rep_uri not in representations:
                    representations[rep_uri] = {
                        "uri": rep_uri,
                        "label": rep_label,
                        "characters": [],
                        "attributes": []
                    }
                    
                if "character" in result:
                    char_uri = result["character"]["value"]
                    char_label = result["characterLabel"]["value"]
                    
                    if {"uri": char_uri, "label": char_label} not in representations[rep_uri]["characters"]:
                        representations[rep_uri]["characters"].append({"uri": char_uri, "label": char_label})
            
            # Query for attributes
            query = """
            PREFIX vir: <http://w3id.org/vir#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?representation ?attribute ?attributeLabel WHERE {
                ?representation a vir:IC9_Representation ;
                               vir:K17_has_attribute ?attribute .
                ?attribute rdfs:label ?attributeLabel .
            }
            """
            
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            # Add attributes to representations
            for result in results["results"]["bindings"]:
                rep_uri = result["representation"]["value"]
                attr_uri = result["attribute"]["value"]
                attr_label = result["attributeLabel"]["value"]
                
                if rep_uri in representations:
                    if {"uri": attr_uri, "label": attr_label} not in representations[rep_uri]["attributes"]:
                        representations[rep_uri]["attributes"].append({"uri": attr_uri, "label": attr_label})
            
            # Create documents with rich iconographic context
            iconographic_documents = []
            for rep_uri, rep_info in representations.items():
                context_text = f"Iconographic representation: {rep_info['label']}\n"
                
                # Add character information
                if rep_info["characters"]:
                    context_text += "Depicts characters:\n"
                    for char in rep_info["characters"]:
                        context_text += f"- {char['label']}\n"
                
                # Add attribute information
                if rep_info["attributes"]:
                    context_text += "Visual attributes:\n"
                    for attr in rep_info["attributes"]:
                        context_text += f"- {attr['label']}\n"
                
                # Create document for the vector store
                doc = Document(
                    page_content=context_text,
                    metadata={
                        "entity": rep_uri,
                        "label": rep_info["label"],
                        "type": "vir:IC9_Representation",
                        "source": "rdf_data_iconographic"
                    }
                )
                iconographic_documents.append(doc)
            
            logger.info(f"Created {len(iconographic_documents)} iconographic context documents")
            
            # Create/update iconographic context vector store
            if iconographic_documents:
                icon_vectorstore = FAISS.from_documents(iconographic_documents, self.embeddings)
                
                # Save for future use
                if not os.path.exists('iconographic_index'):
                    os.makedirs('iconographic_index')
                icon_vectorstore.save_local('iconographic_index')
                logger.info("Iconographic context vector store created and saved")
                
                return icon_vectorstore
            return None
        except Exception as e:
            logger.error(f"Error building iconographic context: {str(e)}")
            return None

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
        """Answer a question using both vector search and graph reasoning"""
        # Initialize chat chain if needed
        if not hasattr(self, 'chat_chain') or self.chat_chain is None:
            logger.info("Chat chain not initialized, setting up...")
            if not self.setup_chat_chain():
                logger.error("Failed to set up chat chain")
                return {
                    "answer": "I'm sorry, I couldn't set up the answering system. Please check the logs for details.",
                    "sources": []
                }
        
        # Use graph-aware retrieval
        retrieved_docs = self.retrieve_with_graph(question, k=10)
        
        # Extract key entities from retrieved documents
        key_entities = []
        for doc in retrieved_docs:
            if doc.metadata.get("source", "").startswith("rdf_data"):
                entity_uri = doc.metadata.get("entity", "")
                if entity_uri and len(key_entities) < 5:  # Limit to 5 key entities
                    key_entities.append(entity_uri)
        
        # If we have multiple entities, extract reasoning paths between them
        reasoning_paths = []
        if len(key_entities) >= 2:
            # Find paths between key entities
            for i in range(len(key_entities) - 1):
                path = self.extract_reasoning_path(key_entities[i], key_entities[i+1])
                if path:
                    reasoning_paths.append(path)
        
        # Create context from documents and reasoning paths
        context = ""
        
        # Add documents to context
        for doc in retrieved_docs:
            context += doc.page_content + "\n\n"
        
        # Add reasoning paths to context
        if reasoning_paths:
            context += "Reasoning paths between entities:\n"
            for path in reasoning_paths:
                path_str = " -> ".join([f"{self.get_entity_label(step[0])} {step[1]} {self.get_entity_label(step[2])}" for step in path])
                context += path_str + "\n"
        
        # Now use the chat chain with the enhanced context
        result = self.chat_chain.invoke({"question": question, "context": context})
        
        # Process answer and sources
        answer = result["answer"]
        source_docs = result.get("source_documents", retrieved_docs)
        
        # Format sources
        sources = []
        for i, doc in enumerate(source_docs):
            if doc.metadata.get("source", "").startswith("rdf_data"):
                entity_uri = doc.metadata.get("entity", "")
                entity_label = doc.metadata.get("label", "")
                
                sources.append({
                    "id": i,
                    "entity_uri": entity_uri,
                    "entity_label": entity_label,
                    "type": doc.metadata.get("source", "unknown")
                })
            elif doc.metadata.get("source") == "ontology_documentation":
                concept_id = doc.metadata.get("concept_id", "")
                concept_name = doc.metadata.get("concept_name", "")
                
                sources.append({
                    "id": i,
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "type": "ontology_documentation"
                })
        
        return {
            "answer": answer,
            "sources": sources,
            "reasoning_paths": [{"path": path} for path in reasoning_paths]
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
            
            When interpreting the information from the knowledge base, pay careful attention to the meaning of ontological relationships:
            
            1. For each relationship (like P89_falls_within, P7_took_place_at, etc.), make sure you understand the direction correctly:
               - Subject is the entity that HAS the relationship
               - Object is the entity that the relationship POINTS TO
            
            2. "Ontology interpretations" sections provide the correct semantic meaning of relationships.
               Always prioritize these interpretations when determining how entities relate to each other.
            
            Retrieved information:
            {context}
            
            User question: {question}
            
            Provide a comprehensive answer that:
            1. Correctly interprets the direction of relationships (especially containment, location, and temporal relationships)
            2. Uses natural language to explain relationships instead of ontology terminology
            3. Is accurate to the information in the knowledge base without introducing speculation
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