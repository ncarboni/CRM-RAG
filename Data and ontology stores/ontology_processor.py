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

    def is_subclass_of(self, concept_id, parent_id):
        """Determine if a concept is a subclass of another concept"""
    
        # Direct match
        if concept_id == parent_id:
            return True
            
        # Check graph for path
        if concept_id in self.ontology_graph and parent_id in self.ontology_graph:
            try:
                path = nx.has_path(self.ontology_graph, parent_id, concept_id)
                return path
            except:
                pass
        
        # Check subclasses recursively
        concept = self.concepts.get(concept_id, {})
        subclass_of = concept.get("subclasses", [])
        
        if parent_id in subclass_of:
            return True
            
        # Recursive check for each parent
        for parent in subclass_of:
            if self.is_subclass_of(parent, parent_id):
                return True
                
        return False

    def process_ontology_docs(self):
        """
        Process ontology documentation.
        
        Returns:
            Dictionary of concepts extracted from documentation
        """
        # Prioritize Turtle file for concept extraction
        turtle_docs = [doc for doc in self.ontology_docs_path if doc.lower().endswith('.ttl')]
        pdf_docs = [doc for doc in self.ontology_docs_path if doc.lower().endswith('.pdf')]
        
        try:
            self.concepts = {}
            
            # Process turtle documents first
            for turtle_path in turtle_docs:
                # Extract concepts from the Turtle file
                new_concepts = self.load_ontology_document(turtle_path)
                # Merge with existing concepts
                self.concepts.update(new_concepts)
                
            # If no concepts found and PDF docs exist, use PDF processing
            if not self.concepts and pdf_docs:
                logger.info("No concepts from Turtle files, falling back to PDF processing")
                all_text = ""
                for doc_path in pdf_docs:
                    if os.path.exists(doc_path):
                        doc_text = self.load_pdf_document(doc_path)
                        all_text += doc_text
                
                # Extract concepts from PDF text
                self.concepts = self.extract_cidoc_crm_concepts(all_text)
            
            if not self.concepts:
                logger.warning("No ontology concepts found")
                return {}
                
            # Build ontology graph
            try:
                self.build_ontology_graph()
                logger.info("Built ontology graph successfully")
            except Exception as e:
                logger.error(f"Error building ontology graph: {str(e)}")
            
            # Process core taxonomy
            try:
                self.process_core_taxonomy()
                logger.info("Processed core taxonomy successfully")
            except Exception as e:
                logger.error(f"Error processing core taxonomy: {str(e)}")
                
            return self.concepts
        except Exception as e:
            logger.error(f"Unhandled error in process_ontology_docs: {str(e)}")
            return {}
    

    def process_core_taxonomy(self):
        """Add core CIDOC-CRM taxonomy relationships to improve context understanding"""
        logger.info("Processing core taxonomy relationships...")
        
        # Define core taxonomy groups as specified
        taxonomy = {
            "physical_entities": {"parent": "E77_Persistent_Item"},
            "temporal_entities": {"parent": "E2_Temporal_Entity"},
            "conceptual_entities": {"parent": "E28_Conceptual_Object"}
        }
        
        # Build subclass relationships
        for concept_id, concept in self.concepts.items():
            # Find taxonomy group for each concept
            for group_name, group_info in taxonomy.items():
                parent = group_info["parent"]
                if self.is_subclass_of(concept_id, parent):
                    if "taxonomy_group" not in concept:
                        concept["taxonomy_group"] = []
                    concept["taxonomy_group"].append(group_name)
                    
            # Special handling for VIR ontology concepts
            if concept_id.startswith("IC") or "vir" in concept.get("ontology", ""):
                if "taxonomy_group" not in concept:
                    concept["taxonomy_group"] = []
                concept["taxonomy_group"].append("visual_representation")
        
        logger.info(f"Added taxonomy groups to {sum(1 for c in self.concepts.values() if 'taxonomy_group' in c)} concepts")
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

    def load_ontology_document(self, turtle_path: str):
        """
        Load an ontology RDF document and extract concepts.
        Works for both CIDOC-CRM and VIR ontologies.
        
        Args:
            turtle_path: Path to the Turtle (.ttl) file
        
        Returns:
            Dictionary of extracted concepts
        """
        logger.info(f"Extracting concepts from ontology file: {turtle_path}")
        
        # Create a graph
        g = rdflib.Graph()
        g.parse(turtle_path, format='turtle')
        
        # Log namespaces for debugging
        logger.info("Namespaces in ontology file:")
        for prefix, namespace in g.namespaces():
            logger.info(f"  {prefix}: {namespace}")
        
        # Supported namespaces
        cidoc_crm = rdflib.Namespace("http://www.cidoc-crm.org/cidoc-crm/")
        vir = rdflib.Namespace("http://w3id.org/vir#")
        
        # Get class types
        class_types = [rdflib.RDFS.Class, rdflib.OWL.Class]
        
        concepts = {}
        concepts_count = 0
        
        # Process all classes in both namespaces
        for class_type in class_types:
            for subj, pred, obj in g.triples((None, rdflib.RDF.type, class_type)):
                # Check if subject is in one of our supported namespaces
                is_crm = str(subj).startswith(str(cidoc_crm))
                is_vir = str(subj).startswith(str(vir))
                
                if is_crm or is_vir:
                    # Extract ID (last part of the URI)
                    if "#" in str(subj):
                        class_id = str(subj).split('#')[-1]
                    else:
                        class_id = str(subj).split('/')[-1]
                    
                    # Get definition
                    comments = list(g.objects(subj, rdflib.RDFS.comment))
                    definition = str(comments[0]) if comments else "No definition available"
                    
                    # Get label
                    labels = list(g.objects(subj, rdflib.RDFS.label))
                    class_name = str(labels[0]) if labels else class_id
                    
                    # Get subclass relationships
                    subclasses = []
                    for _, _, parent in g.triples((subj, rdflib.RDFS.subClassOf, None)):
                        if "#" in str(parent):
                            parent_id = str(parent).split('#')[-1]
                        else:
                            parent_id = str(parent).split('/')[-1]
                            
                        subclasses.append(parent_id)
                    
                    # Create concept entry
                    concepts[class_id] = {
                        'id': class_id,
                        'name': class_name,
                        'type': 'class',
                        'uri': str(subj),
                        'ontology': 'vir' if is_vir else 'cidoc_crm',
                        'definition': definition.strip(),
                        'subclasses': subclasses
                    }
                    concepts_count += 1
                    
                    logger.info(f"Found concept: {class_id} - {class_name} from {concepts[class_id]['ontology']}")
        
        # As a fallback for VIR, also look for classes that are used in subClassOf relationships
        if "vir.ttl" in turtle_path.lower():
            for subj, pred, obj in g.triples((None, rdflib.RDFS.subClassOf, None)):
                # Check if subject is in VIR namespace but not already added
                if str(subj).startswith(str(vir)) and not any(c.get('uri') == str(subj) for c in concepts.values()):
                    # Extract ID
                    class_id = str(subj).split('#')[-1]
                    
                    # Get definition
                    comments = list(g.objects(subj, rdflib.RDFS.comment))
                    definition = str(comments[0]) if comments else "No definition available"
                    
                    # Get label
                    labels = list(g.objects(subj, rdflib.RDFS.label))
                    class_name = str(labels[0]) if labels else class_id
                    
                    if "#" in str(obj):
                        parent_id = str(obj).split('#')[-1]
                    else:
                        parent_id = str(obj).split('/')[-1]
                    
                    logger.info(f"Found VIR subclass: {class_id} - {class_name} (subclass of {parent_id})")
                    
                    # Create concept entry
                    concepts[class_id] = {
                        'id': class_id,
                        'name': class_name,
                        'type': 'class',
                        'uri': str(subj),
                        'ontology': 'vir',
                        'definition': definition.strip(),
                        'subclasses': [parent_id]
                    }
                    concepts_count += 1
        
        # If VIR and still no concepts found, use a hardcoded list
        if "vir.ttl" in turtle_path.lower() and not any(c.get('ontology') == 'vir' for c in concepts.values()):
            logger.info("Using hardcoded VIR class list")
            
            # Key VIR classes
            vir_classes = [
                "IC10_Attribute", "IC16_Character", "IC11_Personification", 
                "IC12_Visual_Recognition", "IC19_Recto", "IC1_Iconographical_Atom",
                "IC20_Verso", "IC9_Representation", "PCK4_is_visual_prototype_of",
                "IC21_Similarity_Statement"
            ]
            
            for class_id in vir_classes:
                class_uri = vir[class_id]
                
                # Get definition
                comments = list(g.objects(class_uri, rdflib.RDFS.comment))
                definition = str(comments[0]) if comments else "No definition available"
                
                # Get label
                labels = list(g.objects(class_uri, rdflib.RDFS.label))
                class_name = str(labels[0]) if labels else class_id
                
                # Get subclass relationships
                subclasses = []
                for _, _, parent in g.triples((class_uri, rdflib.RDFS.subClassOf, None)):
                    if "#" in str(parent):
                        parent_id = str(parent).split('#')[-1]
                    else:
                        parent_id = str(parent).split('/')[-1]
                        
                    subclasses.append(parent_id)
                
                logger.info(f"Added hardcoded VIR class: {class_id} - {class_name}")
                
                concepts[class_id] = {
                    'id': class_id,
                    'name': class_name,
                    'type': 'class',
                    'uri': str(class_uri),
                    'ontology': 'vir',
                    'definition': definition.strip(),
                    'subclasses': subclasses
                }
                concepts_count += 1
        
        logger.info(f"Extracted {concepts_count} concepts from ontology file")
        return concepts
    
    
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
