import logging
import os
from typing import List, Dict, Any, Optional, Tuple

from SPARQLWrapper import SPARQLWrapper, JSON
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import MergerRetriever
from ontology_processor import OntologyProcessor
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class FusekiRagSystem:
    """RAG system that integrates RDF data from Fuseki with ontology knowledge"""
    
    def __init__(self, endpoint_url="http://localhost:3030/asinou/sparql", 
                 embedding_model_name="text-embedding-3-small",
                 openai_api_key=None,
                 openai_model="o-mini",
                 temperature=0.7,
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
                # Use OpenAI embeddings
                self._embeddings = OpenAIEmbeddings(
                    model=self.embedding_model_name,
                    openai_api_key=self.openai_api_key
                )
                logger.info("OpenAI embeddings model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings model: {str(e)}")
                raise
        return self._embeddings

    def initialize(self):
        """Initialize the system - build vector stores and setup chat chain"""
        # Ensure vector stores are built
        self.ensure_vectorstores()
        
        # Setup chat chain
        self.setup_chat_chain()
        
        return True

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
            # Get relationships
            outgoing = self.get_entity_details(entity_uri)
            incoming = self.get_related_entities(entity_uri)
            
            # Format as document text
            text = f"Entity: {self.get_entity_label(entity_uri)} ({entity_uri})\n\n"
            
            if outgoing:
                text += "Outgoing relationships:\n"
                for rel in outgoing[:10]:
                    pred = rel.get("predicateLabel", rel["predicate"].split('/')[-1])
                    obj = rel.get("objectLabel", rel["object"].split('/')[-1])
                    text += f"  - {pred} → {obj}\n"
                    
            if incoming:
                text += "\nIncoming relationships:\n"
                for rel in incoming[:10]:
                    subj = rel.get("subjectLabel", rel["subject"].split('/')[-1])
                    pred = rel.get("predicateLabel", rel["predicate"].split('/')[-1])
                    text += f"  - {subj} → {pred}\n"
            
            # Create document
            doc = Document(
                page_content=text,
                metadata={
                    "entity": entity_uri,
                    "label": self.get_entity_label(entity_uri),
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
            # Set up OpenAI LLM
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
                retriever = MergerRetriever(
                    retrievers=[rdf_retriever, ontology_retriever],
                    weights=[0.7, 0.3]  # 70% weight to RDF data, 30% to ontology
                )
            else:
                # Use only RDF retriever if ontology is not available
                retriever = rdf_retriever
            
            # Create enhanced Byzantine art expert prompt
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