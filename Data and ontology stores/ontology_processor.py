"""
Ontology processor module for extracting and processing CIDOC-CRM concepts.
This module handles loading PDF documentation, extracting ontology concepts,
and building a graph representation of the ontology.
"""

# Rest of your existing imports and code

import logging
import os
import re
import networkx as nx
import rdflib
from rdflib import Namespace
from rdflib.namespace import RDF, RDFS
from typing import Dict, List, Optional, Any

import fitz  # PyMuPDF for PDF processing
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Set up logging
logger = logging.getLogger(__name__)

class OntologyProcessor:
    """Process ontology documentation and extract key concepts and relationships"""
    
    def __init__(self, ontology_docs_path=None):
        """
        Initialize the ontology processor.
        
        Args:
            ontology_docs_path: List of paths to ontology documentation files (PDFs)
        """
        self.ontology_docs_path = ontology_docs_path or []
        self.ontology_graph = nx.DiGraph()
        self.concepts = {}
        self.vectorstore = None
        
    def load_pdf_document(self, pdf_path: str) -> str:
        """
        Load a PDF document and extract text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text from the PDF
        """
        logger.info(f"Loading PDF document: {pdf_path}")
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            logger.error(f"Error loading PDF document: {str(e)}")
            return ""
    
    def load_turtle_document(self, turtle_path: str):
        """
        Load a Turtle RDF document and extract CIDOC-CRM concepts.
        
        Args:
            turtle_path: Path to the Turtle (.ttl) file
        
        Returns:
            Dictionary of extracted concepts
        """
        logger.info(f"Extracting concepts from Turtle file: {turtle_path}")
        
        # Create a graph
        g = rdflib.Graph()
        g.parse(turtle_path, format='turtle')
        
        # Namespaces
        cidoc_crm = rdflib.Namespace("http://www.cidoc-crm.org/cidoc-crm/")
        RDFS = rdflib.RDFS
        
        concepts = {}
        
        # Find all classes in the CIDOC-CRM namespace
        for subj, pred, obj in g.triples((None, rdflib.RDF.type, RDFS.Class)):
            # Check if the subject is in the CIDOC-CRM namespace
            if str(subj).startswith(str(cidoc_crm)):
                # Extract class ID (last part of the URI)
                class_id = str(subj).split('/')[-1]
                
                # Get the comment (definition)
                comments = list(g.objects(subj, RDFS.comment))
                definition = str(comments[0]) if comments else "No definition available"
                
                # Get labels
                labels = list(g.objects(subj, RDFS.label))
                class_name = str(labels[0]) if labels else class_id
                
                # Get subclass relationships
                subclasses = list(g.objects(subj, RDFS.subClassOf))
                
                # Create concept entry
                concepts[class_id] = {
                    'id': class_id,
                    'name': class_name,
                    'type': 'class',
                    'definition': definition.strip(),
                    'subclasses': [str(sc).split('/')[-1] for sc in subclasses if str(sc).startswith(str(cidoc_crm))]
                }
        
        logger.info(f"Extracted {len(concepts)} CIDOC-CRM concepts from Turtle file")
        return concepts


    def extract_cidoc_crm_concepts(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract CIDOC-CRM concepts and their definitions from text.
        
        Args:
            text: Text containing CIDOC-CRM concept definitions
            
        Returns:
            Dictionary of concept IDs to concept information
        """
        # Pattern for CIDOC-CRM class definitions (E1, E2, etc.)
        class_pattern = r'(E\d+)\s+([A-Za-z\s]+)(?:\n|.)+?(?:Scope note:)(.+?)(?:Examples:|Properties:|$)'
        class_matches = re.finditer(class_pattern, text, re.DOTALL)
        
        # Pattern for CIDOC-CRM property definitions (P1, P2, etc.)
        property_pattern = r'(P\d+)\s+([A-Za-z\s]+)\s+\(([^)]+)\s*,\s*([^)]+)\)(?:\n|.)+?(?:Scope note:)(.+?)(?:Examples:|$)'
        property_matches = re.finditer(property_pattern, text, re.DOTALL)
        
        concepts = {}
        
        # Process class matches
        for match in class_matches:
            class_id = match.group(1).strip()
            class_name = match.group(2).strip()
            scope_note = match.group(3).strip().replace('\n', ' ')
            
            concepts[class_id] = {
                'id': class_id,
                'name': class_name,
                'type': 'class',
                'definition': scope_note
            }
            
        # Process property matches
        for match in property_matches:
            prop_id = match.group(1).strip()
            prop_name = match.group(2).strip()
            domain = match.group(3).strip()
            range_ = match.group(4).strip()
            scope_note = match.group(5).strip().replace('\n', ' ')
            
            concepts[prop_id] = {
                'id': prop_id,
                'name': prop_name,
                'type': 'property',
                'domain': domain,
                'range': range_,
                'definition': scope_note
            }
            
        return concepts
    
    def build_ontology_graph(self) -> nx.DiGraph:
        """
        Build a graph representation of the ontology.
        
        Returns:
            A NetworkX DiGraph of the ontology
        """
        # Add nodes for each concept
        for concept_id, concept in self.concepts.items():
            self.ontology_graph.add_node(concept_id, **concept)
        
        # Add edges for subclass relationships
        for concept_id, concept in self.concepts.items():
            if 'subclasses' in concept:
                for subclass in concept['subclasses']:
                    if subclass in self.concepts:
                        self.ontology_graph.add_edge(
                            concept_id, 
                            subclass, 
                            relation='subClassOf',
                            relation_name='is a subclass of'
                        )
        
        return self.ontology_graph
    
    def process_ontology_docs(self):
        """
        Process ontology documentation.
        
        Returns:
            Dictionary of concepts extracted from documentation
        """
        # Prioritize Turtle file for concept extraction
        turtle_docs = [doc for doc in self.ontology_docs_path if doc.lower().endswith('.ttl')]
        pdf_docs = [doc for doc in self.ontology_docs_path if doc.lower().endswith('.pdf')]
        
        if turtle_docs:
            # Extract concepts from the first Turtle file
            self.concepts = self.load_turtle_document(turtle_docs[0])
        elif pdf_docs:
            # Fallback to PDF processing if no Turtle file
            all_text = ""
            for doc_path in pdf_docs:
                if os.path.exists(doc_path):
                    doc_text = self.load_pdf_document(doc_path)
                    all_text += doc_text
            
            # Extract concepts from PDF text
            self.concepts = self.extract_cidoc_crm_concepts(all_text)
        else:
            logger.warning("No ontology documentation found")
            return {}
        
        # Build ontology graph
        self.build_ontology_graph()
        
        return self.concepts
    
    def get_concept_context(self, concept_id: str, depth: int = 1) -> Optional[Dict[str, Any]]:
        """
        Get context for a specific concept, including related concepts.
        
        Args:
            concept_id: ID of the concept to get context for
            depth: How many levels of related concepts to include
            
        Returns:
            Dictionary with concept and related concepts
        """
        if concept_id not in self.concepts:
            return None
        
        context = {
            'concept': self.concepts[concept_id],
            'related': []
        }
        
        if depth > 0 and concept_id in self.ontology_graph:
            # Get predecessors (incoming relationships)
            for pred in self.ontology_graph.predecessors(concept_id):
                edge_data = self.ontology_graph.get_edge_data(pred, concept_id)
                relation_info = {
                    'concept': self.concepts.get(pred),
                    'relation': edge_data.get('relation_name', 'is related to'),
                    'direction': 'incoming'
                }
                context['related'].append(relation_info)
            
            # Get successors (outgoing relationships)
            for succ in self.ontology_graph.successors(concept_id):
                edge_data = self.ontology_graph.get_edge_data(concept_id, succ)
                relation_info = {
                    'concept': self.concepts.get(succ),
                    'relation': edge_data.get('relation_name', 'is related to'),
                    'direction': 'outgoing'
                }
                context['related'].append(relation_info)
                
        return context
    
    def create_concept_documents(self) -> List[Document]:
        """
        Create Document objects for each concept for vector indexing.
        
        Returns:
            List of Document objects
        """
        documents = []
        
        for concept_id, concept in self.concepts.items():
            # Get related concepts context
            context = self.get_concept_context(concept_id)
            
            # Format as text
            text = f"Concept: {concept['id']} - {concept['name']}\n"
            text += f"Type: {concept['type']}\n"
            text += f"Definition: {concept['definition']}\n"
            
            if concept['type'] == 'property':
                text += f"Domain: {concept.get('domain', 'Not specified')}\n"
                text += f"Range: {concept.get('range', 'Not specified')}\n"
            
            if context and 'related' in context:
                text += "Related Concepts:\n"
                for rel in context['related']:
                    rel_concept = rel['concept']
                    if rel_concept:
                        text += f"  - {rel['direction'].capitalize()} {rel['relation']} {rel_concept['id']} ({rel_concept['name']})\n"
            
            # Create document
            doc = Document(
                page_content=text, 
                metadata={
                    "concept_id": concept_id,
                    "concept_name": concept['name'],
                    "concept_type": concept['type'],
                    "source": "ontology_documentation"
                }
            )
            documents.append(doc)
            
        return documents
    
    def build_vectorstore(self, embeddings: Embeddings, force_rebuild: bool = False) -> Optional[FAISS]:
        """
        Build vector store from ontology concepts.
        
        Args:
            embeddings: Embedding model to use
            force_rebuild: Whether to force rebuilding the vector store
            
        Returns:
            FAISS vector store or None if failed
        """
        if not self.concepts:
            logger.info("Processing ontology documentation first")
            self.process_ontology_docs()
            
        if not self.concepts:
            logger.error("No ontology concepts found to index")
            return None
            
        # Check if vector store already exists
        if not force_rebuild and os.path.exists('ontology_index/index.faiss'):
            logger.info("Loading existing ontology vector store")
            try:
                self.vectorstore = FAISS.load_local('ontology_index', embeddings,
                                                   allow_dangerous_deserialization=True)
                logger.info("Ontology vector store loaded successfully")
                return self.vectorstore
            except Exception as e:
                logger.error(f"Failed to load ontology vector store: {str(e)}")
                # If loading fails, we'll rebuild
                
        # Create concept documents
        logger.info("Creating ontology concept documents")
        documents = self.create_concept_documents()
        
        if not documents:
            logger.error("No ontology documents created")
            return None
        
        # Create vector store
        logger.info("Building ontology vector store")
        try:
            self.vectorstore = FAISS.from_documents(documents, embeddings)
            
            # Save for future use
            if not os.path.exists('ontology_index'):
                os.makedirs('ontology_index')
            self.vectorstore.save_local('ontology_index')
            logger.info("Ontology vector store created and saved")
            
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error creating ontology vector store: {str(e)}")
            return None