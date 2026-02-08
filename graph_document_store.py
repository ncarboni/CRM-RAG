"""
Graph-based document store for universal RAG system.
Unified graph structure for document storage with weighted edges and vector retrieval.
"""

import logging
import os
import pickle
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

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

    def create_adjacency_matrix(self, doc_ids, max_hops=2):
        """Create an adjacency matrix with virtual 2-hop edges through the full graph.

        Unlike the previous approach (A^2 within pool only), this discovers
        connections between candidates through intermediate nodes anywhere in
        the full document graph. This is critical for event-based ontologies
        like CIDOC-CRM where entities connect through event intermediaries
        (e.g., Artist -> E12_Production -> Artwork).

        Args:
            doc_ids: List of candidate document IDs
            max_hops: Maximum hops (used for virtual edge discount factor)
        """
        doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        n = len(doc_ids)
        adj_matrix = np.zeros((n, n), dtype=np.float64)

        # 1-hop: direct edges between candidates
        direct_edges = 0
        for i, doc_id in enumerate(doc_ids):
            if doc_id not in self.docs:
                continue
            doc = self.docs[doc_id]
            for neighbor in doc.neighbors:
                neighbor_id = neighbor["doc_id"]
                if neighbor_id in doc_to_idx:
                    j = doc_to_idx[neighbor_id]
                    adj_matrix[i, j] = neighbor["weight"]
                    direct_edges += 1

        # Virtual 2-hop: edges through intermediate nodes in the FULL graph
        # Build inverted index: intermediate_node -> [(candidate_idx, weight)]
        intermediate_index = {}
        for i, doc_id in enumerate(doc_ids):
            if doc_id not in self.docs:
                continue
            doc = self.docs[doc_id]
            for neighbor in doc.neighbors:
                neighbor_id = neighbor["doc_id"]
                # Only consider intermediates OUTSIDE the candidate pool
                if neighbor_id not in doc_to_idx:
                    if neighbor_id not in intermediate_index:
                        intermediate_index[neighbor_id] = []
                    intermediate_index[neighbor_id].append((i, neighbor.get("weight", 0.5)))

        # For each intermediate connecting 2+ candidates, add virtual edges
        virtual_edges = 0
        for intermediate_id, connections in intermediate_index.items():
            if len(connections) < 2:
                continue
            # Add virtual edge between each pair of candidates through this intermediate
            for ci in range(len(connections)):
                idx_a, weight_a = connections[ci]
                for cj in range(ci + 1, len(connections)):
                    idx_b, weight_b = connections[cj]
                    # Virtual edge weight: product of path weights, discounted by hop count
                    virtual_weight = (weight_a * weight_b) * (1.0 / max_hops)
                    # Keep the stronger connection if multiple paths exist
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
    
