"""
Graph-based document store for universal RAG system.
Unified graph structure for document storage with weighted edges and vector retrieval.
"""

import logging
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

try:
    import bm25s
    HAS_BM25S = True
except ImportError:
    HAS_BM25S = False

logger = logging.getLogger(__name__)

@dataclass
class GraphDocument:
    """Document node in the graph store"""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    neighbors: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_neighbor(self, neighbor_doc, edge_type, weight=1.0):
        """Add a connection to another document with weight"""
        self.neighbors.append({
            "doc_id": neighbor_doc.id,
            "edge_type": edge_type,
            "weight": weight
        })


class GraphDocumentStore:
    """Store for graph-connected documents with vectorized retrieval"""
    
    def __init__(self, embeddings_model):
        self.docs = {}  # Document ID to GraphDocument
        self.embeddings_model = embeddings_model
        self.vector_store = None
        self.bm25_retriever = None  # BM25 sparse retrieval index
        self._bm25_doc_ids = []     # Ordered doc IDs matching BM25 corpus order
        
    def add_document(self, doc_id, text, metadata=None):
        """Add a document to the store"""
        embedding = self.embeddings_model.embed_query(text)
        doc = GraphDocument(
            id=doc_id,
            text=text,
            metadata=metadata or {},
            embedding=embedding
        )
        self.docs[doc_id] = doc
        return doc

    def add_document_with_embedding(self, doc_id, text, embedding, metadata=None):
        """
        Add a document with a pre-computed embedding.
        Use this for batch processing where embeddings are generated separately.

        Args:
            doc_id: Unique document identifier
            text: Document text content
            embedding: Pre-computed embedding vector
            metadata: Optional metadata dictionary
        """
        doc = GraphDocument(
            id=doc_id,
            text=text,
            metadata=metadata or {},
            embedding=embedding
        )
        self.docs[doc_id] = doc
        return doc

    def add_edge(self, doc_id1, doc_id2, edge_type, weight=1.0):
        """Add an edge between two documents with weight"""
        if doc_id1 in self.docs and doc_id2 in self.docs:
            self.docs[doc_id1].add_neighbor(self.docs[doc_id2], edge_type, weight)
            self.docs[doc_id2].add_neighbor(self.docs[doc_id1], edge_type, weight)
    
    def rebuild_vector_store(self):
        """Build/rebuild the vector store for initial retrieval using pre-computed embeddings"""
        # Check if documents have pre-computed embeddings
        docs_with_embeddings = [(doc_id, doc) for doc_id, doc in self.docs.items() if doc.embedding is not None]
        docs_without_embeddings = [(doc_id, doc) for doc_id, doc in self.docs.items() if doc.embedding is None]

        if docs_without_embeddings:
            logger.warning(f"{len(docs_without_embeddings)} documents missing embeddings, will generate them")

        # Use pre-computed embeddings when available (avoids redundant API calls)
        if docs_with_embeddings:
            text_embeddings = []
            metadatas = []

            for doc_id, graph_doc in docs_with_embeddings:
                text_embeddings.append((graph_doc.text, graph_doc.embedding))
                metadatas.append({**graph_doc.metadata, "doc_id": doc_id})

            logger.info(f"Building vector store with {len(text_embeddings)} pre-computed embeddings (no API calls)")
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=text_embeddings,
                embedding=self.embeddings_model,
                metadatas=metadatas
            )

            # Add any documents without embeddings (will generate embeddings via API)
            if docs_without_embeddings:
                docs_for_faiss = []
                for doc_id, graph_doc in docs_without_embeddings:
                    doc = Document(
                        page_content=graph_doc.text,
                        metadata={**graph_doc.metadata, "doc_id": doc_id}
                    )
                    docs_for_faiss.append(doc)
                logger.info(f"Adding {len(docs_for_faiss)} documents without pre-computed embeddings")
                self.vector_store.add_documents(docs_for_faiss)
        else:
            # Fallback: no pre-computed embeddings, use original method
            docs_for_faiss = []
            for doc_id, graph_doc in self.docs.items():
                doc = Document(
                    page_content=graph_doc.text,
                    metadata={**graph_doc.metadata, "doc_id": doc_id}
                )
                docs_for_faiss.append(doc)

            logger.info(f"Building vector store with {len(docs_for_faiss)} documents (generating embeddings)")
            self.vector_store = FAISS.from_documents(docs_for_faiss, self.embeddings_model)

    def save_document_graph(self, path='document_graph.pkl'):
        """Save document graph to disk"""
        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.docs, f)
            logger.info(f"Document graph saved to {path} with {len(self.docs)} documents")
            return True
        except Exception as e:
            logger.error(f"Error saving document graph: {str(e)}")
            return False

    def load_document_graph(self, path='document_graph.pkl'):
        """Load document graph from disk"""
        if not os.path.exists(path):
            logger.error(f"Document graph file not found at {path}")
            return False
            
        try:
            with open(path, 'rb') as f:
                self.docs = pickle.load(f)
            logger.info(f"Document graph loaded from {path} with {len(self.docs)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading document graph: {str(e)}")
            return False
        
    def retrieve(self, query, k=10):
        """First-stage retrieval using vector similarity.

        Returns:
            List of (GraphDocument, score) tuples, where score is the FAISS
            similarity score (higher = more similar). FAISS returns L2 distances
            (lower = better), so we convert to similarity via 1/(1+distance).
        """
        if not self.vector_store:
            self.rebuild_vector_store()

        logger.info(f"Retrieving documents for query: '{query}'")
        results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        retrieved = []

        for doc, distance in results_with_scores:
            doc_id = doc.metadata.get("doc_id")
            if doc_id in self.docs:
                similarity = 1.0 / (1.0 + distance)
                retrieved.append((self.docs[doc_id], similarity))

        logger.info(f"Retrieved {len(retrieved)} documents")
        return retrieved

    def create_adjacency_matrix(self, doc_ids, triples_index, weight_fn, max_hops=2):
        """Create an adjacency matrix with virtual 2-hop edges through the full RDF graph.

        Uses the triples index (built from edges.parquet) rather than in-memory
        document neighbor lists.  This ensures the adjacency matrix reflects the
        full knowledge-graph topology — including entities removed by thin-doc
        chaining that still serve as valid intermediaries for 2-hop paths
        (e.g., Artist -> E12_Production -> Artwork).

        Args:
            doc_ids: List of candidate document IDs (entity URIs)
            triples_index: Dict mapping entity_uri -> [triple_dicts] from Parquet.
                Each triple dict has keys: subject, predicate, object (+ labels).
            weight_fn: Callable(predicate_uri) -> float for CIDOC-CRM semantic weights.
            max_hops: Maximum hops (used for virtual edge discount factor)
        """
        doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        n = len(doc_ids)
        adj_matrix = np.zeros((n, n), dtype=np.float64)

        # 1-hop: direct edges between candidates (from full RDF graph)
        direct_edges = 0
        # Virtual 2-hop: intermediate_node -> [(candidate_idx, weight)]
        intermediate_index = {}

        for i, doc_id in enumerate(doc_ids):
            for triple in triples_index.get(doc_id, []):
                # Determine the other endpoint
                if triple["subject"] == doc_id:
                    other = triple["object"]
                else:
                    other = triple["subject"]

                weight = weight_fn(triple["predicate"])

                if other in doc_to_idx:
                    # 1-hop: both endpoints are candidates
                    j = doc_to_idx[other]
                    if weight > adj_matrix[i, j]:
                        adj_matrix[i, j] = weight
                    direct_edges += 1
                else:
                    # Intermediate outside the candidate pool — collect for 2-hop
                    if other not in intermediate_index:
                        intermediate_index[other] = []
                    intermediate_index[other].append((i, weight))

        # For each intermediate connecting 2+ candidates, add virtual edges
        virtual_edges = 0
        for intermediate_id, connections in intermediate_index.items():
            if len(connections) < 2:
                continue
            for ci in range(len(connections)):
                idx_a, weight_a = connections[ci]
                for cj in range(ci + 1, len(connections)):
                    idx_b, weight_b = connections[cj]
                    virtual_weight = (weight_a * weight_b) * (1.0 / max_hops)
                    if virtual_weight > adj_matrix[idx_a, idx_b]:
                        adj_matrix[idx_a, idx_b] = virtual_weight
                        adj_matrix[idx_b, idx_a] = virtual_weight
                        virtual_edges += 1

        logger.info(f"Adjacency: {direct_edges} direct edges, {virtual_edges} virtual 2-hop edges "
                    f"(via {sum(1 for c in intermediate_index.values() if len(c) >= 2)} intermediates)")

        # Add self-loops to prevent isolated nodes
        adj_matrix = adj_matrix + np.eye(n)

        # Log adjacency matrix statistics before normalization
        non_zero_count = np.count_nonzero(adj_matrix - np.eye(n))  # Exclude self-loops
        total_possible = n * n - n  # Exclude diagonal
        density = (non_zero_count / total_possible) if total_possible > 0 else 0.0

        logger.info(f"=== Adjacency Matrix Statistics (before normalization) ===")
        logger.info(f"  Size: {n}x{n} nodes")
        logger.info(f"  Non-zero edges: {non_zero_count} ({density*100:.1f}% density)")
        logger.info(f"  Value range: [{np.min(adj_matrix):.3f}, {np.max(adj_matrix):.3f}]")
        logger.info(f"  Mean weight: {np.mean(adj_matrix):.3f}")
        logger.info(f"  Std weight: {np.std(adj_matrix):.3f}")

        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        rowsum = np.array(adj_matrix.sum(1))
        d_inv_sqrt = np.zeros_like(rowsum)
        non_zero_mask = rowsum > 1e-10

        if np.any(non_zero_mask):
            d_inv_sqrt[non_zero_mask] = np.power(rowsum[non_zero_mask], -0.5)
            d_inv_sqrt = np.nan_to_num(d_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
            d_mat_inv_sqrt = np.diag(d_inv_sqrt)

            with np.errstate(invalid='ignore', divide='ignore'):
                adj_normalized = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
                adj_normalized = np.nan_to_num(adj_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            logger.warning("All rows in adjacency matrix are zero, using identity matrix")
            adj_normalized = np.eye(n)

        # Log adjacency matrix statistics after normalization
        logger.info(f"=== Adjacency Matrix Statistics (after symmetric normalization) ===")
        logger.info(f"  Value range: [{np.min(adj_normalized):.3f}, {np.max(adj_normalized):.3f}]")
        logger.info(f"  Mean: {np.mean(adj_normalized):.3f}")
        logger.info(f"  Std: {np.std(adj_normalized):.3f}")
        logger.info(f"  Row sum range: [{np.min(np.sum(adj_normalized, axis=1)):.3f}, {np.max(np.sum(adj_normalized, axis=1)):.3f}]")

        return adj_normalized

    # ==================== BM25 Sparse Retrieval ====================

    def build_bm25_index(self) -> bool:
        """Build a BM25 index over all documents in the store.

        Uses bm25s for fast sparse retrieval. Tokenization uses simple
        whitespace splitting (language-agnostic for European languages).

        Returns:
            True if index was built successfully, False otherwise.
        """
        if not HAS_BM25S:
            logger.warning("bm25s not installed — BM25 retrieval disabled. Install with: pip install bm25s")
            return False

        if not self.docs:
            logger.warning("No documents to index for BM25")
            return False

        # Build ordered corpus aligned with doc IDs
        self._bm25_doc_ids = list(self.docs.keys())
        corpus = [self.docs[did].text for did in self._bm25_doc_ids]

        logger.info(f"Building BM25 index over {len(corpus)} documents...")
        corpus_tokens = bm25s.tokenize(corpus, stopwords=None)

        self.bm25_retriever = bm25s.BM25()
        self.bm25_retriever.index(corpus_tokens)

        logger.info(f"BM25 index built: {len(corpus)} docs, vocab={len(self.bm25_retriever.vocab_dict)} terms")
        return True

    def retrieve_bm25(self, query: str, k: int = 10) -> List[Tuple['GraphDocument', float]]:
        """Retrieve documents using BM25 sparse scoring.

        Args:
            query: Query string.
            k: Number of results to return.

        Returns:
            List of (GraphDocument, score) tuples sorted by BM25 score descending.
        """
        if not HAS_BM25S or self.bm25_retriever is None:
            return []

        query_tokens = bm25s.tokenize(query, stopwords=None)
        results, scores = self.bm25_retriever.retrieve(query_tokens, k=min(k, len(self._bm25_doc_ids)))

        retrieved = []
        for idx, score in zip(results[0], scores[0]):
            if score <= 0:
                continue
            doc_id = self._bm25_doc_ids[idx]
            if doc_id in self.docs:
                retrieved.append((self.docs[doc_id], float(score)))

        logger.info(f"BM25 retrieved {len(retrieved)} documents (top score: {retrieved[0][1]:.3f})" if retrieved else "BM25 retrieved 0 documents")
        return retrieved

    # ==================== FC Type Index ====================

    def build_fc_type_index(self, fc_class_mapping: Dict[str, List[str]]) -> None:
        """Build an inverted index from FC category to document IDs.

        Scans all documents once and maps each FC name to the set of doc IDs
        whose ``metadata["all_types"]`` overlaps with that FC's class list.

        Args:
            fc_class_mapping: FC name → list of CRM class names (E-coded and
                human-readable labels, already expanded).
        """
        self._fc_doc_ids: Dict[str, set] = {fc: set() for fc in fc_class_mapping}

        # Build lookup: class_name → set of FC names it belongs to
        class_to_fc: Dict[str, set] = {}
        for fc_name, class_list in fc_class_mapping.items():
            for cls_name in class_list:
                class_to_fc.setdefault(cls_name, set()).add(fc_name)

        for doc_id, doc in self.docs.items():
            all_types = doc.metadata.get('all_types', [])
            for t in all_types:
                for fc_name in class_to_fc.get(t, ()):
                    self._fc_doc_ids[fc_name].add(doc_id)

        summary = ", ".join(f"{fc}: {len(ids)}" for fc, ids in sorted(self._fc_doc_ids.items()))
        logger.info(f"FC type index built — {summary}")

    def retrieve_faiss_typed(self, query: str, k: int,
                             allowed_doc_ids: set,
                             fetch_k: int = 10000) -> List[Tuple['GraphDocument', float]]:
        """FAISS retrieval restricted to a set of allowed document IDs.

        Uses the native ``filter`` parameter of LangChain FAISS to pre-screen
        candidates before ranking.

        Args:
            query: Query string.
            k: Number of results to return.
            allowed_doc_ids: Set of doc IDs to consider.
            fetch_k: Raw FAISS results to scan before applying the filter.

        Returns:
            List of (GraphDocument, similarity_score) sorted descending.
        """
        if not self.vector_store:
            return []

        results_with_scores = self.vector_store.similarity_search_with_score(
            query, k=k,
            filter=lambda meta: meta.get("doc_id") in allowed_doc_ids,
            fetch_k=fetch_k,
        )

        retrieved = []
        for doc, distance in results_with_scores:
            doc_id = doc.metadata.get("doc_id")
            if doc_id in self.docs:
                similarity = 1.0 / (1.0 + distance)
                retrieved.append((self.docs[doc_id], similarity))

        logger.info(f"Type-filtered FAISS: {len(retrieved)} results "
                    f"(from {len(allowed_doc_ids)} allowed docs, fetch_k={fetch_k})")
        return retrieved

    def retrieve_bm25_typed(self, query: str, k: int,
                            allowed_doc_ids: set) -> List[Tuple['GraphDocument', float]]:
        """BM25 retrieval restricted to a set of allowed document IDs.

        Retrieves an over-sized BM25 pool and post-filters to ``allowed_doc_ids``.

        Args:
            query: Query string.
            k: Number of results to return.
            allowed_doc_ids: Set of doc IDs to consider.

        Returns:
            List of (GraphDocument, score) sorted descending, at most k items.
        """
        if not HAS_BM25S or self.bm25_retriever is None:
            return []

        # Retrieve a large pool and post-filter
        oversized_k = min(k * 5, len(self._bm25_doc_ids))
        query_tokens = bm25s.tokenize(query, stopwords=None)
        results, scores = self.bm25_retriever.retrieve(query_tokens, k=oversized_k)

        retrieved = []
        for idx, score in zip(results[0], scores[0]):
            if score <= 0:
                continue
            doc_id = self._bm25_doc_ids[idx]
            if doc_id in allowed_doc_ids and doc_id in self.docs:
                retrieved.append((self.docs[doc_id], float(score)))
                if len(retrieved) >= k:
                    break

        logger.info(f"Type-filtered BM25: {len(retrieved)} results "
                    f"(from {len(allowed_doc_ids)} allowed docs)")
        return retrieved

    def save_bm25_index(self, path: str) -> bool:
        """Save BM25 index and doc ID mapping to disk.

        Args:
            path: Directory to save the BM25 index.

        Returns:
            True if saved successfully.
        """
        if not HAS_BM25S or self.bm25_retriever is None:
            return False

        os.makedirs(path, exist_ok=True)
        self.bm25_retriever.save(path)

        # Save doc ID ordering separately
        ids_path = os.path.join(path, "bm25_doc_ids.pkl")
        with open(ids_path, 'wb') as f:
            pickle.dump(self._bm25_doc_ids, f)

        logger.info(f"BM25 index saved to {path}")
        return True

    def load_bm25_index(self, path: str) -> bool:
        """Load BM25 index and doc ID mapping from disk.

        Args:
            path: Directory containing the saved BM25 index.

        Returns:
            True if loaded successfully.
        """
        if not HAS_BM25S:
            logger.warning("bm25s not installed — cannot load BM25 index")
            return False

        ids_path = os.path.join(path, "bm25_doc_ids.pkl")
        if not os.path.exists(ids_path):
            logger.info(f"BM25 index not found at {path}")
            return False

        try:
            self.bm25_retriever = bm25s.BM25.load(path, mmap=True)
            with open(ids_path, 'rb') as f:
                self._bm25_doc_ids = pickle.load(f)
            logger.info(f"BM25 index loaded from {path} ({len(self._bm25_doc_ids)} docs)")
            return True
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            self.bm25_retriever = None
            self._bm25_doc_ids = []
            return False
