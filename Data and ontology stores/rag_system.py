import logging
import asyncio
from SPARQLWrapper import SPARQLWrapper, JSON
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from ontology_processor import OntologyProcessor

# Set Ollama LLM and local embedding model globally for LlamaIndex
Settings.llm = Ollama(model="llama3")
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FusekiRagSystem:
    def __init__(
        self,
        endpoint_url="http://localhost:3030/asinou/sparql",
        ontology_docs=None,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        self.ontology_processor = OntologyProcessor(ontology_docs=ontology_docs or [])
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        self.graph_index = None
        self.vector_index = None

    def test_connection(self):
        self.sparql.setQuery("ASK { ?s ?p ?o }")
        try:
            result = self.sparql.query().convert()
            return result.get("boolean", False)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    async def build_graph_index(self, limit: int = 50):
        logger.info("Building PropertyGraphIndex from RDF triples...")
        self.sparql.setQuery(f"""
            SELECT ?s ?p ?o WHERE {{
                ?s ?p ?o .
            }} LIMIT {limit}
        """)
        results = self.sparql.query().convert()

        triples = [
            (r["s"]["value"], r["p"]["value"], r["o"]["value"])
            for r in results["results"]["bindings"]
        ]

        nodes = [TextNode(text=f"({s}, {p}, {o})") for s, p, o in triples]

        self.graph_index = await PropertyGraphIndex.ainit(
            nodes=nodes,
            show_progress=True,
            use_async=True
        )

        logger.info(f"PropertyGraphIndex built with {len(nodes)} nodes.")

        logger.info("Processing ontology documents...")
        self.ontology_processor.process_ontology()
        vectorstore = self.ontology_processor.build_vectorstore(self.embeddings)
        if vectorstore:
            self.vector_index = VectorStoreIndex.from_vector_store(vectorstore)
            logger.info("Vector index created from ontology.")
        else:
            logger.warning("Ontology vectorstore not created.")

    def query_graphrag(self, question: str) -> str:
        if not self.graph_index or not self.vector_index:
            raise ValueError("Graph and vector indices must be built before querying.")

        graph_docs = self.graph_index.as_retriever(similarity_top_k=5).retrieve(question)
        vec_docs = self.vector_index.as_retriever(similarity_top_k=5).retrieve(question)

        combined_docs = {doc.metadata.get("source", "") + doc.metadata.get("concept_id", "") + doc.get_content(): doc for doc in graph_docs + vec_docs}
        sorted_docs = sorted(combined_docs.values(), key=lambda d: getattr(d, "score", 0), reverse=True)

        engine = RetrieverQueryEngine.from_args(retriever=lambda q: sorted_docs)
        response = engine.query(question)
        return str(response)