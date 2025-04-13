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
from langchain.text_splitter import RecursiveCharacterTextSplitter



# Set up logging
logger = logging.getLogger(__name__)

class FusekiRagSystem:
    """RAG system that integrates RDF data from Fuseki with ontology knowledge"""
    
    def __init__(self, endpoint_url="http://localhost:3030/asinou/sparql", 
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

    def get_entity_by_taxonomy_group(self, taxonomy_group, limit=10):
        """Get entities that belong to a specific taxonomy group"""
        logger.info(f"Finding entities in taxonomy group: {taxonomy_group}")
        
        # Map taxonomy groups to CIDOC-CRM classes
        taxonomy_mapping = {
            "physical_entities": "E77_Persistent_Item",
            "temporal_entities": "E2_Temporal_Entity",
            "conceptual_entities": "E28_Conceptual_Object"
        }
        
        cidoc_class = taxonomy_mapping.get(taxonomy_group)
        if not cidoc_class:
            return []
        
        query = f"""
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?entity ?entityLabel WHERE {{
            ?entity rdf:type/rdfs:subClassOf* crm:{cidoc_class} ;
                    rdfs:label ?entityLabel .
        }}
        LIMIT {limit}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            entities = []
            for result in results["results"]["bindings"]:
                entities.append({
                    "entity_uri": result["entity"]["value"],
                    "entity_label": result["entityLabel"]["value"],
                    "taxonomy_group": taxonomy_group
                })
            
            return entities
        except Exception as e:
            logger.error(f"Error querying taxonomy group: {str(e)}")
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
            
            You should understand that the information is organized using the CIDOC-CRM ontology which classifies:
            1. Physical entities (churches, artworks, physical objects)
            2. Temporal entities (events like creations and historical periods)
            3. Conceptual entities (visual elements, symbols, and meanings)
            
            But DO NOT mention these ontology terms in your answer. Instead, use natural language to explain 
            relationships between entities. For example, instead of saying "This E22_Man-Made_Object has a 
            P55_has_current_location relationship", simply say "This church is located in".
            
            Retrieved information:
            {context}
            
            User question: {question}
            
            Provide a comprehensive answer using the retrieved information. When mentioning locations, churches, 
            or artworks, specify their geographic and historical context where possible. Use natural, conversational 
            language that a non-expert would understand.
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
            # First, check for explicit ontology terms
            ontology_terms = self.extract_ontology_terms(question)
            
            # Then, map natural language to ontology concepts
            nl_mapping = self.map_natural_language_to_ontology(question)
            
            # Determine query focus for context-aware retrieval
            query_focus = self.identify_query_focus(question)
            
            # Enhanced retrieval based on query understanding
            if nl_mapping or ontology_terms:
                logger.info(f"Found ontology mapping: {nl_mapping}")
                # Use context-aware retrieval if we have taxonomy groups identified
                taxonomy_groups = nl_mapping.get("taxonomy_groups", []) if nl_mapping else []
                
                if taxonomy_groups:
                    # Temporary improvement to retrieval
                    docs_by_group = []
                    
                    # Get docs from each relevant taxonomy group
                    for group in taxonomy_groups:
                        entities = self.get_entity_by_taxonomy_group(group, limit=5)
                        for entity in entities:
                            entity_docs = self.search(f"{question} {entity['entity_label']}", k=2)
                            docs_by_group.extend(entity_docs)
                    
                    # Combine with regular retrieval
                    regular_docs = self.search(question, k=5)
                    all_docs = regular_docs + docs_by_group
                    
                    # Deduplicate
                    unique_docs = []
                    seen_ids = set()
                    for doc in all_docs:
                        doc_id = doc.metadata.get("entity", doc.metadata.get("concept_id", None))
                        if doc_id and doc_id not in seen_ids:
                            unique_docs.append(doc)
                            seen_ids.add(doc_id)
                    
                    # Limit to top k
                    context_docs = unique_docs[:8]  # Increase to 8 for better coverage
                    
                    # Create temporary retriever with enhanced docs
                    from langchain.schema import Document
                    
                    class ContextRetriever:
                        def __init__(self, docs):
                            self.docs = docs
                        
                        def get_relevant_documents(self, query):
                            return self.docs
                    
                    # Swap retrievers
                    original_retriever = self.chat_chain.retriever
                    self.chat_chain.retriever = ContextRetriever(context_docs)
                    
                    # Get answer
                    result = self.chat_chain.invoke({"question": question})
                    
                    # Restore original retriever
                    self.chat_chain.retriever = original_retriever
                else:
                    # Use standard retrieval
                    result = self.chat_chain.invoke({"question": question})
            else:
                # Use standard retrieval for regular questions
                result = self.chat_chain.invoke({"question": question})
            
            # Process answer and sources
            answer = result["answer"]
            source_docs = result.get("source_documents", [])
            
            # Format sources like in original code...
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
            
            # Don't add ontology explanations to the answer since users 
            # shouldn't need to see technical ontology details
            
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

    def extract_ontology_terms(self, query):
        """Extract ontology terms from a query"""
        terms = []
        
        # Look for CIDOC-CRM class patterns: E1, E22, etc.
        class_matches = re.findall(r'E\d+(?:_[A-Za-z_-]+)?', query)
        terms.extend(class_matches)
        
        # Look for CIDOC-CRM property patterns: P1, P2, etc.
        property_matches = re.findall(r'P\d+(?:_[A-Za-z_-]+)?', query)
        terms.extend(property_matches)
        
        # Look for VIR patterns: IC1, IC9, etc.
        vir_matches = re.findall(r'IC\d+(?:_[A-Za-z_-]+)?', query)
        terms.extend(vir_matches)
        
        # Look for K-pattern properties from VIR
        vir_prop_matches = re.findall(r'K\d+(?:_[A-Za-z_-]+)?', query)
        terms.extend(vir_prop_matches)
        
        return terms

    def generate_ontology_enhanced_queries(self, query):
        """Generate multiple ontology-aware queries from a single user query"""
        ontology_terms = self.extract_ontology_terms(query)
        
        enhanced_queries = [query]  # Start with original query
        
        # Add taxonomy-aware variations
        for term in ontology_terms:
            if not self.ontology_processor or not self.ontology_processor.concepts:
                continue
                
            concept = self.ontology_processor.concepts.get(term)
            if concept:
                # Add query with full concept definition
                enhanced_queries.append(f"{query} related to {term} which is defined as {concept.get('definition', '')}")
                
                # Add query with taxonomy group
                for group in concept.get('taxonomy_group', []):
                    enhanced_queries.append(f"{query} related to {group} {term}")
                    
                # Add query with related concepts
                context = self.ontology_processor.get_concept_context(term)
                if context and 'related' in context:
                    related_terms = [rel['concept']['id'] for rel in context['related'] if rel.get('concept')]
                    for rel_term in related_terms[:2]:  # Limit to avoid too many queries
                        enhanced_queries.append(f"{query} involving {term} and {rel_term}")
        
        return enhanced_queries[:5]  # Limit to 5 queries

    # In rag_system.py
    def map_natural_language_to_ontology(self, query):
        """Map natural language terms to ontology concepts"""
        # Common natural language mappings to CIDOC-CRM concepts
        mappings = {
            "church": ["E22_Man-Made_Object", "church"],
            "building": ["E22_Man-Made_Object", "building"],
            "artwork": ["E22_Man-Made_Object", "artwork"],
            "painting": ["E22_Man-Made_Object", "painting"],
            "icon": ["E22_Man-Made_Object", "icon"],
            "mosaic": ["E22_Man-Made_Object", "mosaic"],
            "fresco": ["E22_Man-Made_Object", "fresco"],
            "location": ["E53_Place"],
            "place": ["E53_Place"],
            "creation": ["E65_Creation"],
            "production": ["E12_Production"],
            "time": ["E52_Time-Span"],
            "date": ["E52_Time-Span"],
            "period": ["E4_Period"],
            "person": ["E21_Person"],
            "character": ["vir:IC16_Character"],
            "saint": ["vir:IC16_Character", "saint"],
            "attribute": ["vir:IC10_Attribute"],
            "symbol": ["vir:IC10_Attribute", "symbol"],
            "representation": ["vir:IC9_Representation"],
            "iconography": ["vir:IC9_Representation"]
        }
        
        # Check if query contains any of the terms
        enhanced_query = query
        matched_terms = []
        
        for term, ontology_concepts in mappings.items():
            # Use word boundary pattern to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, query.lower()):
                matched_terms.append((term, ontology_concepts))
        
        # If we found matches, enhance the query internally
        if matched_terms:
            # Create an internal representation that will help the retrieval
            # but won't be shown to the user
            internal_representation = {
                "original_query": query,
                "ontology_mappings": matched_terms,
                "taxonomy_groups": []
            }
            
            # Determine which taxonomy groups might be relevant
            if any(concept[0].startswith("E77") or concept[0] in ["E22_Man-Made_Object", "E53_Place"] for _, concepts in matched_terms for concept in concepts):
                internal_representation["taxonomy_groups"].append("physical_entities")
                
            if any(concept[0].startswith("E2") or concept[0] in ["E12_Production", "E65_Creation"] for _, concepts in matched_terms for concept in concepts):
                internal_representation["taxonomy_groups"].append("temporal_entities")
                
            if any(concept[0].startswith("E28") or concept[0].startswith("vir:") for _, concepts in matched_terms for concept in concepts):
                internal_representation["taxonomy_groups"].append("conceptual_entities")
                
            if any(concept[0].startswith("vir:") for _, concepts in matched_terms for concept in concepts):
                internal_representation["taxonomy_groups"].append("visual_representation")
                
            return internal_representation
            
        return None

    def identify_query_focus(self, query):
        """Identify the focus of a query (location, temporal, conceptual)"""
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Location patterns
        location_patterns = ["where", "located", "location", "place", "region", "area", "country"]
        for pattern in location_patterns:
            if pattern in query_lower:
                return "location"
        
        # Temporal patterns
        temporal_patterns = ["when", "date", "period", "century", "year", "time", "era"]
        for pattern in temporal_patterns:
            if pattern in query_lower:
                return "temporal"
        
        # Conceptual patterns
        conceptual_patterns = ["meaning", "symbol", "represent", "concept", "iconography", "attribute"]
        for pattern in conceptual_patterns:
            if pattern in query_lower:
                return "conceptual"
        
        # Default to generic
        return "generic"

    def get_entity_location_context(self, entity_uri):
        """Get location context for an entity"""
        # Check if entity has location information
        query = f"""
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?location ?locationLabel ?parent ?parentLabel WHERE {{
            <{entity_uri}> crm:P55_has_current_location ?location .
            OPTIONAL {{ ?location rdfs:label ?locationLabel }}
            OPTIONAL {{ 
                ?location crm:P89_falls_within ?parent .
                ?parent rdfs:label ?parentLabel .
            }}
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            context_docs = []
            for result in results["results"]["bindings"]:
                location_uri = result["location"]["value"]
                
                # Format text with location context
                text = f"Entity: {entity_uri}\n"
                text += f"Location: {result.get('locationLabel', {'value': location_uri})['value']}\n"
                
                if "parent" in result:
                    text += f"Within: {result.get('parentLabel', {'value': result['parent']['value']})['value']}\n"
                
                # Create document
                doc = Document(
                    page_content=text,
                    metadata={
                        "entity": entity_uri,
                        "location": location_uri,
                        "source": "rdf_data_location"
                    }
                )
                context_docs.append(doc)
                
            return context_docs
        except Exception as e:
            logger.error(f"Error fetching location context: {str(e)}")
            return []

    def get_entity_temporal_context(self, entity_uri):
        """Get temporal context for an entity"""
        # Check if entity has temporal information
        query = f"""
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?production ?date ?dateLabel WHERE {{
            <{entity_uri}> crm:P108i_was_produced_by ?production .
            OPTIONAL {{ 
                ?production crm:P4_has_time-span/crm:P82_at_some_time_within ?date .
                OPTIONAL {{ ?date rdfs:label ?dateLabel }}
            }}
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            context_docs = []
            for result in results["results"]["bindings"]:
                # Format text with temporal context
                text = f"Entity: {entity_uri}\n"
                
                if "date" in result:
                    date_value = result.get('dateLabel', {'value': result['date']['value']})['value']
                    text += f"Created: {date_value}\n"
                
                # Create document
                doc = Document(
                    page_content=text,
                    metadata={
                        "entity": entity_uri,
                        "source": "rdf_data_temporal"
                    }
                )
                context_docs.append(doc)
                
            return context_docs
        except Exception as e:
            logger.error(f"Error fetching temporal context: {str(e)}")
            return []

    def get_entity_conceptual_context(self, entity_uri):
        """Get conceptual context for an entity (e.g., iconography, symbolism)"""
        # Check if entity has conceptual information
        query = f"""
        PREFIX vir: <http://w3id.org/vir#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?attribute ?attributeLabel ?symbol ?symbolLabel WHERE {{
            <{entity_uri}> vir:K17_has_attribute ?attribute .
            OPTIONAL {{ ?attribute rdfs:label ?attributeLabel }}
            OPTIONAL {{ 
                ?attribute vir:K14_symbolize ?symbol .
                OPTIONAL {{ ?symbol rdfs:label ?symbolLabel }}
            }}
        }}
        """
        
        try:
            self.sparql.setQuery(query)
            results = self.sparql.query().convert()
            
            context_docs = []
            for result in results["results"]["bindings"]:
                # Format text with conceptual context
                text = f"Entity: {entity_uri}\n"
                
                if "attribute" in result:
                    attr_label = result.get('attributeLabel', {'value': result['attribute']['value']})['value']
                    text += f"Attribute: {attr_label}\n"
                    
                    if "symbol" in result:
                        symbol_label = result.get('symbolLabel', {'value': result['symbol']['value']})['value']
                        text += f"Symbolizes: {symbol_label}\n"
                
                # Create document
                doc = Document(
                    page_content=text,
                    metadata={
                        "entity": entity_uri,
                        "source": "rdf_data_conceptual"
                    }
                )
                context_docs.append(doc)
                
            return context_docs
        except Exception as e:
            logger.error(f"Error fetching conceptual context: {str(e)}")
            return []


    def retrieve_with_context(self, query, k=5):
        """Enhanced retrieval that considers entity types and taxonomy"""
        
        # Identify the query intent
        query_focus = self.identify_query_focus(query)
        
        # First, get base results
        base_docs = self.search(query, k=k)
        
        # Get entities from base results
        entities = set()
        for doc in base_docs:
            if doc.metadata.get("source") == "rdf_data":
                entity_uri = doc.metadata.get("entity", "")
                if entity_uri:
                    entities.add(entity_uri)
        
        # Get context for each entity
        context_docs = []
        for entity_uri in entities:
            # Get entity details including type
            details = self.get_entity_details(entity_uri)
            
            entity_type = None
            for detail in details:
                if "rdf-syntax-ns#type" in detail["predicate"]:
                    entity_type = detail["object"].split("/")[-1]
                    break
            
            # Based on query focus, get additional context
            if query_focus == "location" and entity_type and "E22_" in entity_type:
                # If query is about location and entity is a physical object
                location_details = self.get_entity_location_context(entity_uri)
                if location_details:
                    context_docs.extend(location_details)
            elif query_focus == "temporal" and entity_type:
                # If query is about time
                temporal_details = self.get_entity_temporal_context(entity_uri)
                if temporal_details:
                    context_docs.extend(temporal_details)
            elif query_focus == "conceptual" and entity_type:
                # If query is about concepts
                concept_details = self.get_entity_conceptual_context(entity_uri)
                if concept_details:
                    context_docs.extend(concept_details)
        
        # Combine original results with context documents
        all_docs = base_docs + context_docs
        
        # Deduplicate
        unique_docs = []
        seen_ids = set()
        for doc in all_docs:
            doc_id = doc.metadata.get("entity", doc.metadata.get("concept_id", None))
            if doc_id and doc_id not in seen_ids:
                unique_docs.append(doc)
                seen_ids.add(doc_id)
        
        return unique_docs[:k]  # Return top k results



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
