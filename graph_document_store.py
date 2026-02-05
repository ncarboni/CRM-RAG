"""
Graph-based document store for universal RAG system.
This replaces context-specific indices with a unified graph structure.
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
        """First-stage retrieval using vector similarity"""
        if not self.vector_store:
            self.rebuild_vector_store()
            
        logger.info(f"Retrieving documents for query: '{query}'")
        results = self.vector_store.similarity_search(query, k=k)
        retrieved_docs = []
        
        for doc in results:
            doc_id = doc.metadata.get("doc_id")
            if doc_id in self.docs:
                retrieved_docs.append(self.docs[doc_id])
                
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs

    def create_adjacency_matrix(self, doc_ids, max_hops=3):
        """Create an adjacency matrix for a subgraph with multi-hop connections"""
        # Create a mapping of doc IDs to indices
        doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        
        # Create an empty adjacency matrix
        n = len(doc_ids)
        adj_matrix = np.zeros((n, n), dtype=np.float64)
        
        # Fill the adjacency matrix with direct connections (1-hop)
        for i, doc_id in enumerate(doc_ids):
            if doc_id not in self.docs:
                continue
                
            doc = self.docs[doc_id]
            for neighbor in doc.neighbors:
                neighbor_id = neighbor["doc_id"]
                if neighbor_id in doc_to_idx:
                    j = doc_to_idx[neighbor_id]
                    adj_matrix[i, j] = neighbor["weight"]
        
        # Store the original 1-hop adjacency matrix
        original_adj = adj_matrix.copy()
        
        # Compute multi-hop connections up to max_hops with numerical stability
        current_power = original_adj.copy()
        for hop in range(2, max_hops + 1):
            # Compute next power by multiplying current power by original
            # This is more stable than using np.linalg.matrix_power
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                current_power = np.matmul(current_power, original_adj)
                
                # Clip values to prevent overflow
                current_power = np.clip(current_power, -1e10, 1e10)
                
                # Replace any NaN or Inf values with 0
                current_power = np.nan_to_num(current_power, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Add these connections to the adjacency matrix with reduced weight
                # Scale by 1/hop to give closer connections more weight
                adj_matrix += current_power * (1.0 / hop)
        
        # Add self-loops to prevent isolated nodes
        adj_matrix = adj_matrix + np.eye(n)

        # Log adjacency matrix statistics before normalization
        non_zero_count = np.count_nonzero(adj_matrix - np.eye(n))  # Exclude self-loops
        total_possible = n * n - n  # Exclude diagonal
        sparsity = 1.0 - (non_zero_count / total_possible) if total_possible > 0 else 0.0

        logger.info(f"=== Adjacency Matrix Statistics (before normalization) ===")
        logger.info(f"  Size: {n}x{n} nodes")
        logger.info(f"  Non-zero edges: {non_zero_count} ({(1-sparsity)*100:.1f}% density)")
        logger.info(f"  Value range: [{np.min(adj_matrix):.3f}, {np.max(adj_matrix):.3f}]")
        logger.info(f"  Mean weight: {np.mean(adj_matrix):.3f}")
        logger.info(f"  Std weight: {np.std(adj_matrix):.3f}")

        # Normalize the adjacency matrix with numerical stability
        rowsum = np.array(adj_matrix.sum(1))
        
        # Handle zero or very small rows
        d_inv_sqrt = np.zeros_like(rowsum)
        non_zero_mask = rowsum > 1e-10  # Use small threshold instead of exact zero
        
        if np.any(non_zero_mask):
            d_inv_sqrt[non_zero_mask] = np.power(rowsum[non_zero_mask], -0.5)
            
            # Replace any remaining inf or nan values
            d_inv_sqrt = np.nan_to_num(d_inv_sqrt, nan=0.0, posinf=0.0, neginf=0.0)
            
            d_mat_inv_sqrt = np.diag(d_inv_sqrt)
            
            with np.errstate(invalid='ignore', divide='ignore'):
                adj_normalized = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
                
                # Clean up any remaining numerical issues
                adj_normalized = np.nan_to_num(adj_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # If all rows are zero (no connections), return identity matrix
            logger.warning("All rows in adjacency matrix are zero, using identity matrix")
            adj_normalized = np.eye(n)

        # Log adjacency matrix statistics after normalization
        logger.info(f"=== Adjacency Matrix Statistics (after symmetric normalization) ===")
        logger.info(f"  Value range: [{np.min(adj_normalized):.3f}, {np.max(adj_normalized):.3f}]")
        logger.info(f"  Mean: {np.mean(adj_normalized):.3f}")
        logger.info(f"  Std: {np.std(adj_normalized):.3f}")
        logger.info(f"  Row sum range: [{np.min(np.sum(adj_normalized, axis=1)):.3f}, {np.max(np.sum(adj_normalized, axis=1)):.3f}]")

        return adj_normalized
    
