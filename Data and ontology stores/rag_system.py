"""
RAG system module that integrates RDF data and ontology knowledge.
This module handles retrieving data from Fuseki, building vector stores,
and answering questions using a dual-store RAG architecture.
"""

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
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import MergerRetriever

from ontology_processor import OntologyProcessor

# Set up logging
logger = logging.getLogger(__name__)

class FusekiRagSystem:
    """RAG system that integrates RDF data from Fuseki with ontology knowledge"""
    
    def __init__(self, endpoint_url="http://localhost:3030/Asinou/sparql", 
                 embedding_model_name="sentence-transformers/all-mpnet-base-v2",
                 ollama_model="llama3",
                 ontology_docs_path=None):
        """
        Initialize the RAG system.
        
        Args:
            endpoint_url: URL of the Fuseki SPARQL endpoint
            embedding_model_name: Name of the embedding model to use
            ollama_model: Name of the Ollama model to use
            ontology_docs_path: List of paths to ontology documentation files
        """
        self.endpoint_url = endpoint_url
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        self._embeddings = None
        self.rdf_vectorstore = None
        self.ontology_vectorstore = None
        self.embedding_model_name = embedding_model_name
        self.ollama_model = ollama_model
        self.chat_chain = None
        
        # Initialize ontology processor
        self.ontology_processor = OntologyProcessor(ontology_docs_path)
        
    @property
    def embeddings(self):
        """Lazy-load embeddings model"""
        if self._embeddings is None:
            logger.info("Initializing embeddings model...")
            try:
                self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
                logger.info("Embeddings model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings model: {str(e)}")
                raise
        return self._embeddings

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

    def get_all_churches(self):
        """
        Get all church buildings and their locations.
        
        Returns:
            List of dictionaries with church information
        """
        logger.info("Fetching all churches with locations")
        query = """
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?church ?churchLabel ?location ?coordinates WHERE {
            ?church rdf:type crm:E22_Man-Made_Object ;
                   rdfs:label ?churchLabel ;
                   crm:P55_has_current_location ?location .
            
            OPTIONAL {
                ?location crm:P168_is_approximated_by ?coordinates .
            }
            
            ?church crm:P2_has_type ?type .
            ?type rdfs:label "Church" .
        }
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            churches = []
            for result in results["results"]["bindings"]:
                church = {
                    "uri": result["church"]["value"],
                    "label": result["churchLabel"]["value"],
                    "location": result["location"]["value"]
                }
                
                if "coordinates" in result:
                    # Extract coordinates from WKT format
                    wkt = result["coordinates"]["value"]
                    if "POINT" in wkt:
                        # Parse POINT(lon lat) format
                        coords = wkt.replace("POINT(", "").replace(")", "").split()
                        if len(coords) >= 2:
                            church["longitude"] = float(coords[0])
                            church["latitude"] = float(coords[1])
                
                churches.append(church)
            
            logger.info(f"Retrieved {len(churches)} churches with location data")
            return churches
        except Exception as e:
            logger.error(f"Error fetching churches: {str(e)}")
            return []

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

    def identify_cidoc_predicates(self, rdf_documents):
        """
        Identify CIDOC-CRM predicates in documents to enhance with ontology info.
        
        Args:
            rdf_documents: List of Document objects containing RDF data
            
        Returns:
            List of CIDOC-CRM predicate IDs found in the documents
        """
        # List of RDF predicates that match CIDOC-CRM patterns
        cidoc_predicates = []
        
        # Regex patterns for CIDOC-CRM classes and properties
        class_pattern = r'(E\d+)_([A-Za-z_-]+)'
        property_pattern = r'(P\d+)_([A-Za-z_-]+)'
        
        for doc in rdf_documents:
            content = doc.page_content
            
            # Find all CIDOC-CRM classes
            class_matches = re.finditer(class_pattern, content)
            for match in class_matches:
                class_id = match.group(1)
                if class_id not in cidoc_predicates:
                    cidoc_predicates.append(class_id)
            
            # Find all CIDOC-CRM properties
            property_matches = re.finditer(property_pattern, content)
            for match in property_matches:
                prop_id = match.group(1)
                if prop_id not in cidoc_predicates:
                    cidoc_predicates.append(prop_id)
        
        return cidoc_predicates
    
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
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


    def setup_chat_chain(self):
        """
        Set up a conversational chain using Ollama and both vector store retrievers.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        logger.info("Setting up conversational chain with Ollama...")
        
        if not self.ensure_vectorstores():
            logger.error("Cannot set up chat chain without at least the RDF vector store")
            return False
        
        try:
            # Set up Ollama LLM
            llm = ChatOllama(
                model=self.ollama_model,
                temperature=0.7,
                top_p=0.9
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
                """You are an expert in Byzantine art and architecture with deep knowledge about churches, iconography, 
                attributes, and historical context. Use the following retrieved information to answer the user's question.
                
                Some of the information comes from a local RDF database about Byzantine art, and some comes from 
                documentation about the CIDOC-CRM ontology. CIDOC-CRM is an ontology framework used to describe
                cultural heritage objects. It uses "E" classes for entities (like E22_Man-Made_Object for physical objects)
                and "P" properties for relationships (like P55_has_current_location to describe where something is located).
                
                When ontological information is available, use it to provide deeper context for your explanation.
                When explaining relationships between entities, try to use proper ontological terminology.
                
                Retrieved information:
                {context}
                
                User question: {question}
                
                Provide a comprehensive but concise answer. Include specific details from the information provided.
                If the information doesn't contain a clear answer, say so. Don't make things up.
                Where appropriate, cite specific sources using [1], [2], etc.
                
                Answer:"""
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

    def answer_question(self, question):
        """
        Answer a question using both RDF and ontology knowledge.
        
        Args:
            question: Question to answer
            
        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Answering question: '{question}'")
        
        if not hasattr(self, 'chat_chain') or self.chat_chain is None:
            logger.info("Chat chain not initialized, setting up...")
            if not self.setup_chat_chain():
                logger.error("Failed to set up chat chain")
                return {
                    "answer": "I'm sorry, I couldn't set up the answering system. Please check the logs for details.",
                    "sources": []
                }
        
        try:
            # Get the answer from the chain
            result = self.chat_chain.invoke({"question": question})
            
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Format source documents for citation
            sources = []
            
            # Group sources by type (RDF data vs ontology)
            rdf_sources = []
            ontology_sources = []
            
            for i, doc in enumerate(source_docs):
                source_type = doc.metadata.get("source", "unknown")
                
                if source_type == "rdf_data":
                    entity_uri = doc.metadata.get("entity", "")
                    entity_label = doc.metadata.get("label", "")
                    
                    rdf_sources.append({
                        "id": i,
                        "entity_uri": entity_uri,
                        "entity_label": entity_label,
                        "type": "rdf_data"
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
            
            # Add RDF sources first
            sources.extend(rdf_sources)
            
            # Add ontology sources
            sources.extend(ontology_sources)
            
            # Add ontology explanation if ontology sources were used
            if ontology_sources:
                ontology_note = "\n\nNote on CIDOC-CRM concepts used: "
                
                for src in ontology_sources[:3]:  # Limit to top 3
                    concept_id = src.get("concept_id", "")
                    concept_name = src.get("concept_name", "")
                    
                    if concept_id and concept_name:
                        if self.ontology_processor and concept_id in self.ontology_processor.concepts:
                            concept = self.ontology_processor.concepts[concept_id]
                            definition = concept.get("definition", "")
                            short_def = definition[:100] + "..." if len(definition) > 100 else definition
                            
                            ontology_note += f"\n- {concept_id} ({concept_name}): {short_def}"
                
                # Only add the note if we actually have concept definitions
                if len(ontology_note) > 30:  # More than just the header
                    answer += ontology_note
            
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
        
    def get_wikidata_entities(self):
        """
        Get all entities that have Wikidata references.
        
        Returns:
            List of dictionaries with Wikidata entity information
        """
        logger.info("Finding entities with Wikidata references")
        
        query = """
        PREFIX crmdig: <http://www.ics.forth.gr/isl/CRMdig/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?entity ?entityLabel ?wikidata WHERE {
            ?entity crmdig:L54_is_same-as ?wikidata .
            ?entity rdfs:label ?entityLabel .
            FILTER(STRSTARTS(STR(?wikidata), "http://www.wikidata.org/entity/"))
        }
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            wikidata_entities = []
            for result in results["results"]["bindings"]:
                wikidata_uri = result["wikidata"]["value"]
                wikidata_id = wikidata_uri.split("/")[-1]
                
                wikidata_entities.append({
                    "entity_uri": result["entity"]["value"],
                    "entity_label": result["entityLabel"]["value"],
                    "wikidata_uri": wikidata_uri,
                    "wikidata_id": wikidata_id
                })
            
            logger.info(f"Found {len(wikidata_entities)} entities with Wikidata references")
            return wikidata_entities
        except Exception as e:
            logger.error(f"Error fetching Wikidata entities: {str(e)}")
            return []

    def get_wikidata_for_entity(self, entity_uri):
        """
        Get Wikidata ID for a specific entity if it exists.
        
        Args:
            entity_uri: URI of the entity to get Wikidata ID for
            
        Returns:
            Wikidata ID or None if not found
        """
        logger.info(f"Getting Wikidata ID for entity: {entity_uri}")
        
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
                wikidata_id = wikidata_uri.split("/")[-1]
                logger.info(f"Found Wikidata ID: {wikidata_id}")
                return wikidata_id
            else:
                logger.info("No Wikidata ID found for this entity")
                return None
        except Exception as e:
            logger.error(f"Error fetching Wikidata ID: {str(e)}")
            return None

    def fetch_wikidata_info(self, wikidata_id):
        """
        Fetch information from Wikidata for a given entity ID.
        
        Args:
            wikidata_id: Wikidata entity ID
            
        Returns:
            Dictionary with Wikidata information or None if failed
        """
        logger.info(f"Fetching Wikidata info for: {wikidata_id}")
        
        try:
            # Using the Wikidata SPARQL endpoint
            wikidata_endpoint = "https://query.wikidata.org/sparql"
            sparql = SPARQLWrapper(wikidata_endpoint)
            sparql.setReturnFormat(JSON)
            
            # Query to get basic information and description
            query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX schema: <http://schema.org/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT ?item ?itemLabel ?itemDescription ?image ?inception ?coordinates WHERE {{
              BIND(wd:{wikidata_id} AS ?item)
              OPTIONAL {{ ?item wdt:P18 ?image. }}
              OPTIONAL {{ ?item wdt:P571 ?inception. }}
              OPTIONAL {{ ?item wdt:P625 ?coordinates. }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            """
            
            sparql.setQuery(query)
            results = sparql.query().convert()
            
            if not results["results"]["bindings"]:
                logger.warning(f"No Wikidata results for ID: {wikidata_id}")
                return None
            
            info = results["results"]["bindings"][0]
            
            # Format the result
            wikidata_info = {
                "id": wikidata_id,
                "url": f"https://www.wikidata.org/wiki/{wikidata_id}"
            }
            
            if "itemLabel" in info:
                wikidata_info["label"] = info["itemLabel"]["value"]
            
            if "itemDescription" in info:
                wikidata_info["description"] = info["itemDescription"]["value"]
            
            if "image" in info:
                wikidata_info["image"] = info["image"]["value"]
            
            if "inception" in info:
                wikidata_info["inception"] = info["inception"]["value"]
            
            if "coordinates" in info:
                wikidata_info["coordinates"] = info["coordinates"]["value"]
            
            # Query to get additional properties
            properties_query = f"""
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX p: <http://www.wikidata.org/prop/>
            PREFIX ps: <http://www.wikidata.org/prop/statement/>
            PREFIX wikibase: <http://wikiba.se/ontology#>
            
            SELECT ?propLabel ?valLabel WHERE {{
              {{
                wd:{wikidata_id} ?p ?statement .
                ?statement ?ps ?val .
                
                ?prop wikibase:claim ?p .
                ?prop wikibase:statementProperty ?ps .
                
                FILTER(STRSTARTS(STR(?val), "http://www.wikidata.org/entity/"))
              }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
              
              # Limit to common relevant properties
              VALUES ?prop {{
                wdt:P31  # instance of
                wdt:P279 # subclass of
                wdt:P180 # depicts
                wdt:P186 # material used
                wdt:P170 # creator
                wdt:P276 # location
                wdt:P1343 # described by source
                wdt:P571 # inception
                wdt:P136 # genre
                wdt:P921 # main subject
              }}
            }}
            LIMIT 50
            """
            
            sparql.setQuery(properties_query)
            prop_results = sparql.query().convert()
            
            properties = {}
            for prop_result in prop_results["results"]["bindings"]:
                prop_label = prop_result["propLabel"]["value"]
                val_label = prop_result["valLabel"]["value"]
                
                if prop_label not in properties:
                    properties[prop_label] = []
                
                if val_label not in properties[prop_label]:
                    properties[prop_label].append(val_label)
            
            wikidata_info["properties"] = properties
            
            logger.info(f"Successfully retrieved Wikidata info for {wikidata_id}")
            return wikidata_info
        
        except Exception as e:
            logger.error(f"Error fetching from Wikidata: {str(e)}")
            return None
