import logging
import os
import re
import networkx as nx
import rdflib
from rdflib.namespace import RDF, RDFS
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class OntologyProcessor:
    def __init__(self, ontology_docs: List[str]):
        self.ontology_docs = ontology_docs
        self.ontology_graph = nx.DiGraph()
        self.concepts = {}
        self.vectorstore = None

    def extract_ontology_concepts(self, ttl_path: str) -> Dict[str, Dict[str, Any]]:
        logger.info(f"Extracting ontology concepts from: {ttl_path}")
        g = rdflib.Graph()
        g.parse(ttl_path, format='turtle')

        cidoc_ns = rdflib.Namespace("http://www.cidoc-crm.org/cidoc-crm/")
        vir_ns = rdflib.Namespace("http://w3id.org/vir#")
        class_types = [RDFS.Class, rdflib.OWL.Class]
        concepts = {}

        for class_type in class_types:
            for subj, _, _ in g.triples((None, RDF.type, class_type)):
                if not (str(subj).startswith(str(cidoc_ns)) or str(subj).startswith(str(vir_ns))):
                    continue

                class_id = subj.split('#')[-1] if '#' in str(subj) else subj.split('/')[-1]
                labels = list(g.objects(subj, RDFS.label))
                comments = list(g.objects(subj, RDFS.comment))
                subclasses = [
                    obj.split('#')[-1] if '#' in str(obj) else obj.split('/')[-1]
                    for _, _, obj in g.triples((subj, RDFS.subClassOf, None))
                ]

                concepts[class_id] = {
                    'id': class_id,
                    'name': str(labels[0]) if labels else class_id,
                    'type': 'class',
                    'uri': str(subj),
                    'ontology': 'vir' if str(subj).startswith(str(vir_ns)) else 'cidoc_crm',
                    'definition': str(comments[0]).strip() if comments else "No definition available",
                    'subclasses': subclasses
                }

        logger.info(f"Extracted {len(concepts)} concepts from {ttl_path}")
        return concepts

    def process_ontology(self):
        self.concepts = {}
        for doc in self.ontology_docs:
            self.concepts.update(self.extract_ontology_concepts(doc))
        self._build_ontology_graph()
        self._assign_taxonomy_groups()

    def _build_ontology_graph(self):
        for concept_id, concept in self.concepts.items():
            self.ontology_graph.add_node(concept_id, **concept)
        for concept_id, concept in self.concepts.items():
            for subclass in concept.get('subclasses', []):
                if subclass in self.concepts:
                    self.ontology_graph.add_edge(
                        concept_id, subclass, relation='subClassOf', relation_name='is a subclass of'
                    )

    def _assign_taxonomy_groups(self):
        taxonomy = {
            "physical_entities": "E77_Persistent_Item",
            "temporal_entities": "E2_Temporal_Entity",
            "conceptual_entities": "E28_Conceptual_Object"
        }

        for cid, concept in self.concepts.items():
            for group, parent in taxonomy.items():
                if self._is_subclass_of(cid, parent):
                    concept.setdefault("taxonomy_group", []).append(group)
            if cid.startswith("IC") or concept.get("ontology") == "vir":
                concept.setdefault("taxonomy_group", []).append("visual_representation")

    def _is_subclass_of(self, cid: str, parent: str) -> bool:
        if cid == parent:
            return True
        if cid not in self.ontology_graph or parent not in self.ontology_graph:
            return False
        try:
            return nx.has_path(self.ontology_graph, parent, cid)
        except:
            return False

    def get_concept_context(self, concept_id: str, depth: int = 1) -> Optional[Dict[str, Any]]:
        if concept_id not in self.concepts:
            return None
        context = {'concept': self.concepts[concept_id], 'related': []}
        if depth > 0 and concept_id in self.ontology_graph:
            for pred in self.ontology_graph.predecessors(concept_id):
                rel = self.ontology_graph.get_edge_data(pred, concept_id)
                context['related'].append({
                    'concept': self.concepts.get(pred),
                    'relation': rel.get('relation_name', 'related to'),
                    'direction': 'incoming'
                })
            for succ in self.ontology_graph.successors(concept_id):
                rel = self.ontology_graph.get_edge_data(concept_id, succ)
                context['related'].append({
                    'concept': self.concepts.get(succ),
                    'relation': rel.get('relation_name', 'related to'),
                    'direction': 'outgoing'
                })
        return context

    def create_concept_documents(self) -> List[Document]:
        docs = []
        for cid, concept in self.concepts.items():
            context = self.get_concept_context(cid)
            text = f"Concept: {cid} - {concept['name']}\n"
            text += f"Type: {concept['type']}\n"
            text += f"Definition: {concept['definition']}\n"
            if context:
                for rel in context['related']:
                    rel_con = rel['concept']
                    if rel_con:
                        text += f"  - {rel['direction'].capitalize()} {rel['relation']} {rel_con['id']} ({rel_con['name']})\n"
            docs.append(Document(page_content=text, metadata={
                "concept_id": cid,
                "concept_name": concept['name'],
                "concept_type": concept['type'],
                "source": "ontology_documentation"
            }))
        return docs

    def build_vectorstore(self, embeddings: Embeddings, force_rebuild: bool = False) -> Optional[FAISS]:
        if not self.concepts:
            self.process_ontology()
        if not self.concepts:
            logger.error("No ontology concepts available.")
            return None
        if not force_rebuild and os.path.exists('ontology_index/index.faiss'):
            try:
                self.vectorstore = FAISS.load_local('ontology_index', embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded existing ontology vector store.")
                return self.vectorstore
            except Exception as e:
                logger.warning(f"Failed to load existing index, rebuilding: {e}")
        docs = self.create_concept_documents()
        if not docs:
            logger.error("No ontology documents created.")
            return None
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        os.makedirs('ontology_index', exist_ok=True)
        self.vectorstore.save_local('ontology_index')
        logger.info("Ontology vector store created and saved.")
        return self.vectorstore
