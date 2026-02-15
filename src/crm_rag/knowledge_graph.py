"""Persistent igraph-backed knowledge graph for CRM_RAG.

Replaces six separate data structures (edges.parquet, _triples_index,
GraphDocument.neighbors, nx.DiGraph, _actor_work_counts, aggregation_index.json)
with a single igraph graph stored as a pickle.

Vertex attributes:
    name   – entity URI (used as primary key)
    label  – human-readable label
    fc     – Fundamental Category (Thing/Actor/Place/Event/Concept/Time)
    is_doc – whether this vertex has a document in the store
    doc_type – primary type string from doc.metadata["type"] (empty for non-docs)
    pagerank – float, computed after build

Edge attributes:
    predicate       – predicate URI (RDF) or FR id (FR)
    predicate_label – human label
    weight          – CIDOC-CRM semantic weight (RDF) or 1.0 (FR)
    edge_type       – "rdf" or "fr"
"""

import logging
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

import igraph as ig
import numpy as np

logger = logging.getLogger(__name__)

# Predicates used to trace the Actor→Work production chain
_ACTOR_CHAIN_PREDICATES = {
    "P14_carried_out_by", "P14i_performed",
    "P108i_was_produced_by",
    "P16i_was_used_for",
}

_DATE_PREDICATES = {
    "P82a_begin_of_the_begin", "P82b_end_of_the_end",
    "P82_at_some_time_within", "P81a_end_of_the_begin", "P81b_begin_of_the_end",
}


class KnowledgeGraph:
    """Persistent igraph-backed knowledge graph."""

    def __init__(self):
        self._graph: ig.Graph = ig.Graph(directed=True)
        self._uri_to_vid: Dict[str, int] = {}
        self._seen_hashes: Set[int] = set()

    # ── Vertex helper ──

    def _get_or_create_vertex(self, uri: str, label: str = "") -> int:
        vid = self._uri_to_vid.get(uri)
        if vid is not None:
            # Update label if previously empty and now provided
            if label and not self._graph.vs[vid]["label"]:
                self._graph.vs[vid]["label"] = label
            return vid
        vid = self._graph.vcount()
        self._graph.add_vertex(
            name=uri, label=label, fc="", is_doc=False,
            doc_type="", pagerank=0.0,
        )
        self._uri_to_vid[uri] = vid
        return vid

    # ── Incremental build ──

    def add_triples(self, triples: List[Dict], weight_fn: Callable) -> None:
        """Add RDF triples as edges, deduplicating by (s, p, o) hash."""
        new_edges = []
        new_attrs = {"predicate": [], "predicate_label": [], "weight": [], "edge_type": []}
        for t in triples:
            h = hash((t["subject"], t["predicate"], t["object"]))
            if h in self._seen_hashes:
                continue
            self._seen_hashes.add(h)
            s_vid = self._get_or_create_vertex(t["subject"], t.get("subject_label", ""))
            o_vid = self._get_or_create_vertex(t["object"], t.get("object_label", ""))
            new_edges.append((s_vid, o_vid))
            new_attrs["predicate"].append(t["predicate"])
            new_attrs["predicate_label"].append(t.get("predicate_label", ""))
            new_attrs["weight"].append(weight_fn(t["predicate"]))
            new_attrs["edge_type"].append("rdf")
        if new_edges:
            self._graph.add_edges(new_edges, new_attrs)

    def add_fr_edges(self, all_fr_stats: List[Tuple]) -> None:
        """Add FR shortcut edges + set FC on source vertices.

        Args:
            all_fr_stats: List of (entity_uri, entity_label, fr_stats_dict).
                fr_stats_dict has "fc" and "fr_results" keys.
        """
        new_edges = []
        new_attrs = {"predicate": [], "predicate_label": [], "weight": [], "edge_type": []}
        for entity_uri, entity_label, stats in all_fr_stats:
            fc = stats.get("fc", "")
            s_vid = self._get_or_create_vertex(entity_uri, entity_label)
            if fc:
                self._graph.vs[s_vid]["fc"] = fc
            for fr in stats.get("fr_results", []):
                fr_id = fr["fr_id"]
                fr_label = fr.get("fr_label", fr_id)
                for target_uri, target_label in fr["targets"]:
                    o_vid = self._get_or_create_vertex(target_uri, target_label)
                    new_edges.append((s_vid, o_vid))
                    new_attrs["predicate"].append(fr_id)
                    new_attrs["predicate_label"].append(fr_label)
                    new_attrs["weight"].append(1.0)
                    new_attrs["edge_type"].append("fr")
        if new_edges:
            self._graph.add_edges(new_edges, new_attrs)
        logger.info(f"Added {len(new_edges)} FR edges from {len(all_fr_stats)} entities")

    def mark_doc_vertices(self, doc_uris: Set[str],
                          doc_types: Optional[Dict[str, str]] = None) -> None:
        """Set is_doc=True (and optionally doc_type) for vertices that have documents."""
        marked = 0
        for uri in doc_uris:
            vid = self._uri_to_vid.get(uri)
            if vid is not None:
                self._graph.vs[vid]["is_doc"] = True
                if doc_types and uri in doc_types:
                    self._graph.vs[vid]["doc_type"] = doc_types[uri]
                marked += 1
        logger.info(f"Marked {marked}/{len(doc_uris)} vertices as doc vertices")

    def compute_pagerank(self, alpha: float = 0.85) -> None:
        """Run PageRank on FR edges, store scores as vertex attribute."""
        fr_eids = [e.index for e in self._graph.es if e["edge_type"] == "fr"]
        if not fr_eids:
            logger.info("PageRank: no FR edges, skipping")
            return
        subgraph = self._graph.subgraph_edges(fr_eids)
        scores = subgraph.pagerank(damping=alpha, weights="weight")
        # Map back to main graph
        scored = 0
        for i, v in enumerate(subgraph.vs):
            main_vid = self._uri_to_vid.get(v["name"])
            if main_vid is not None:
                self._graph.vs[main_vid]["pagerank"] = scores[i]
                scored += 1
        logger.info(f"PageRank computed on {subgraph.vcount()} nodes / "
                    f"{subgraph.ecount()} FR edges, {scored} scores stored")

    # ── Persistence ──

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._graph.write_pickle(path)
        self._seen_hashes.clear()
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"KnowledgeGraph saved to {path} ({size_mb:.1f} MB, "
                    f"{self._graph.vcount()} vertices, {self._graph.ecount()} edges)")

    def load(self, path: str) -> None:
        self._graph = ig.Graph.Read_Pickle(path)
        self._uri_to_vid = {v["name"]: v.index for v in self._graph.vs}
        logger.info(f"KnowledgeGraph loaded from {path} "
                    f"({self._graph.vcount()} vertices, {self._graph.ecount()} edges)")

    # ── Triple queries (replaces _triples_index) ──

    def get_triples(self, entity_uri: str, edge_type: str = "rdf") -> List[Dict]:
        """Return triples incident on entity_uri as list of dicts.

        Each dict has: subject, subject_label, predicate, predicate_label,
                       object, object_label.
        """
        vid = self._uri_to_vid.get(entity_uri)
        if vid is None:
            return []
        result = []
        for e in self._graph.es[self._graph.incident(vid, mode="all")]:
            if e["edge_type"] != edge_type:
                continue
            src = self._graph.vs[e.source]
            tgt = self._graph.vs[e.target]
            result.append({
                "subject": src["name"],
                "subject_label": src["label"],
                "predicate": e["predicate"],
                "predicate_label": e["predicate_label"],
                "object": tgt["name"],
                "object_label": tgt["label"],
            })
        return result

    def triple_count(self, entity_uri: str) -> int:
        """Count of RDF triples incident on entity_uri."""
        vid = self._uri_to_vid.get(entity_uri)
        if vid is None:
            return 0
        return sum(
            1 for e in self._graph.es[self._graph.incident(vid, mode="all")]
            if e["edge_type"] == "rdf"
        )

    def get_rdf_neighbors(self, entity_uri: str,
                          filter_uris: Optional[Set[str]] = None
                          ) -> List[Tuple[str, str, float]]:
        """Return (neighbor_uri, predicate_local_name, weight) for RDF edges.

        If filter_uris is given, only return neighbors whose URI is in the set.
        """
        vid = self._uri_to_vid.get(entity_uri)
        if vid is None:
            return []
        result = []
        for e in self._graph.es[self._graph.incident(vid, mode="all")]:
            if e["edge_type"] != "rdf":
                continue
            other_vid = e.target if e.source == vid else e.source
            other_uri = self._graph.vs[other_vid]["name"]
            if filter_uris is not None and other_uri not in filter_uris:
                continue
            pred = e["predicate"]
            local_name = pred.split("/")[-1].split("#")[-1]
            result.append((other_uri, local_name, e["weight"]))
        return result

    def resolve_time_span(self, ts_uri: str) -> Dict[str, str]:
        """Follow a time-span URI to extract begin/end date values."""
        vid = self._uri_to_vid.get(ts_uri)
        if vid is None:
            return {}
        dates = {}
        for e in self._graph.es[self._graph.incident(vid, mode="out")]:
            if e["edge_type"] != "rdf":
                continue
            if e.source != vid:
                continue
            pred_local = e["predicate"].split("/")[-1].split("#")[-1]
            for dp in _DATE_PREDICATES:
                if dp in pred_local:
                    tgt = self._graph.vs[e.target]
                    obj_val = tgt["label"] or tgt["name"]
                    if obj_val and not obj_val.startswith("http"):
                        if "begin" in dp:
                            dates["began"] = obj_val
                        elif "end" in dp:
                            dates["ended"] = obj_val
                        else:
                            dates["date"] = obj_val
        return dates

    # ── Adjacency matrix (replaces document_store.create_adjacency_matrix) ──

    def build_adjacency_matrix(self, doc_ids: List[str],
                               weight_fn: Callable,
                               max_hops: int = 2) -> np.ndarray:
        """Build weighted adjacency matrix with virtual 2-hop edges.

        Uses RDF edges from the full graph (including non-doc intermediaries)
        for both 1-hop direct edges and 2-hop virtual edges.

        Args:
            doc_ids: Candidate document URIs.
            weight_fn: Callable(predicate_uri) -> float for CIDOC-CRM weights.
            max_hops: Discount factor denominator for virtual edges.

        Returns:
            Symmetrically-normalized adjacency matrix (n×n).
        """
        doc_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
        n = len(doc_ids)
        adj_matrix = np.zeros((n, n), dtype=np.float64)

        direct_edges = 0
        intermediate_index: Dict[str, List[Tuple[int, float]]] = {}

        for i, doc_id in enumerate(doc_ids):
            vid = self._uri_to_vid.get(doc_id)
            if vid is None:
                continue
            for e in self._graph.es[self._graph.incident(vid, mode="all")]:
                if e["edge_type"] != "rdf":
                    continue
                other_vid = e.target if e.source == vid else e.source
                other_uri = self._graph.vs[other_vid]["name"]
                weight = weight_fn(e["predicate"])

                if other_uri in doc_to_idx:
                    j = doc_to_idx[other_uri]
                    if weight > adj_matrix[i, j]:
                        adj_matrix[i, j] = weight
                    direct_edges += 1
                else:
                    if other_uri not in intermediate_index:
                        intermediate_index[other_uri] = []
                    intermediate_index[other_uri].append((i, weight))

        # Virtual 2-hop edges through intermediaries
        virtual_edges = 0
        for connections in intermediate_index.values():
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

        intermediaries = sum(1 for c in intermediate_index.values() if len(c) >= 2)
        logger.info(f"Adjacency: {direct_edges} direct edges, {virtual_edges} virtual 2-hop edges "
                    f"(via {intermediaries} intermediates)")

        # Self-loops
        adj_matrix += np.eye(n)

        # Stats before normalization
        non_zero = np.count_nonzero(adj_matrix - np.eye(n))
        total_possible = n * n - n
        density = (non_zero / total_possible) if total_possible > 0 else 0.0
        logger.info(f"=== Adjacency Matrix Statistics (before normalization) ===")
        logger.info(f"  Size: {n}x{n} nodes")
        logger.info(f"  Non-zero edges: {non_zero} ({density*100:.1f}% density)")
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

        logger.info(f"=== Adjacency Matrix Statistics (after symmetric normalization) ===")
        logger.info(f"  Value range: [{np.min(adj_normalized):.3f}, {np.max(adj_normalized):.3f}]")
        logger.info(f"  Mean: {np.mean(adj_normalized):.3f}")
        logger.info(f"  Std: {np.std(adj_normalized):.3f}")
        logger.info(f"  Row sum range: [{np.min(np.sum(adj_normalized, axis=1)):.3f}, "
                    f"{np.max(np.sum(adj_normalized, axis=1)):.3f}]")

        return adj_normalized

    # ── Dataset statistics (replaces aggregation_index.json) ──

    def get_pagerank_top(self, fc: Optional[str] = None, top_n: int = 500) -> List[Dict]:
        """Top entities by PageRank (doc vertices only), optionally filtered by FC."""
        entries = []
        for v in self._graph.vs:
            if not v["is_doc"]:
                continue
            if v["pagerank"] <= 0:
                continue
            if fc and v["fc"] != fc:
                continue
            entries.append({
                "uri": v["name"],
                "label": v["label"],
                "score": v["pagerank"],
            })
        entries.sort(key=lambda x: x["score"], reverse=True)
        return entries[:top_n]

    def get_entity_type_counts(self) -> Dict[str, int]:
        """Count doc entities by their doc_type vertex attribute."""
        counts: Dict[str, int] = {}
        for v in self._graph.vs:
            if not v["is_doc"]:
                continue
            dt = v["doc_type"] or "Unknown"
            counts[dt] = counts.get(dt, 0) + 1
        return counts

    def get_fc_counts(self) -> Dict[str, int]:
        """Count entities per FC from vertex attributes."""
        counts: Dict[str, int] = {}
        for v in self._graph.vs:
            fc = v["fc"]
            if fc:
                counts[fc] = counts.get(fc, 0) + 1
        return counts

    def get_fr_summaries(self, fr_meta: Optional[Dict] = None) -> Dict[str, Dict]:
        """Compute FR statistics from FR edges.

        Args:
            fr_meta: Optional dict fr_id -> {"label", "domain_fc", "range_fc"}.

        Returns:
            Dict of fr_id -> summary dict matching the old aggregation_index format.
        """
        if fr_meta is None:
            fr_meta = {}

        # Accumulate per-FR
        source_counts: Dict[str, Dict[str, int]] = {}
        source_labels: Dict[str, Dict[str, str]] = {}
        target_counts: Dict[str, Dict[str, int]] = {}
        target_labels: Dict[str, Dict[str, str]] = {}

        for e in self._graph.es:
            if e["edge_type"] != "fr":
                continue
            fr_id = e["predicate"]
            src = self._graph.vs[e.source]
            tgt = self._graph.vs[e.target]

            if fr_id not in source_counts:
                source_counts[fr_id] = {}
                source_labels[fr_id] = {}
                target_counts[fr_id] = {}
                target_labels[fr_id] = {}

            source_counts[fr_id][src["name"]] = source_counts[fr_id].get(src["name"], 0) + 1
            source_labels[fr_id][src["name"]] = src["label"]
            target_counts[fr_id][tgt["name"]] = target_counts[fr_id].get(tgt["name"], 0) + 1
            target_labels[fr_id][tgt["name"]] = tgt["label"]

        def _top_n(counts_dict, labels_dict, n=50):
            sorted_items = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)[:n]
            return [
                {"uri": uri, "label": labels_dict.get(uri, uri.rsplit("/", 1)[-1]), "count": count}
                for uri, count in sorted_items
            ]

        summaries = {}
        for fr_id in source_counts:
            meta = fr_meta.get(fr_id, {})
            total_connections = sum(source_counts[fr_id].values())
            summaries[fr_id] = {
                "label": meta.get("label", fr_id),
                "domain_fc": meta.get("domain_fc", ""),
                "range_fc": meta.get("range_fc", ""),
                "total_connections": total_connections,
                "unique_sources": len(source_counts[fr_id]),
                "unique_targets": len(target_counts[fr_id]),
                "top_sources": _top_n(source_counts[fr_id], source_labels[fr_id]),
                "top_targets": _top_n(target_counts[fr_id], target_labels[fr_id]),
            }
        return summaries

    def actor_work_counts(self) -> Dict[str, int]:
        """Compute actor→work counts from production chain predicates.

        Traces: Work -[P108i_was_produced_by/P16i_was_used_for]-> Event
                -[P14_carried_out_by]-> Actor.

        Returns:
            Dict of actor_uri -> work count.
        """
        creation_to_actors: Dict[str, List[str]] = {}
        work_to_events: Dict[str, List[str]] = {}

        for e in self._graph.es:
            if e["edge_type"] != "rdf":
                continue
            pred = e["predicate"]
            local = pred.split("/")[-1].split("#")[-1]
            src_uri = self._graph.vs[e.source]["name"]
            tgt_uri = self._graph.vs[e.target]["name"]

            if "P14_carried_out_by" in local or "P14i_performed" in local:
                if local.endswith("_carried_out_by"):
                    creation_to_actors.setdefault(src_uri, []).append(tgt_uri)
                else:
                    creation_to_actors.setdefault(tgt_uri, []).append(src_uri)
            elif "P108i_was_produced_by" in local:
                work_to_events.setdefault(src_uri, []).append(tgt_uri)
            elif "P16i_was_used_for" in local:
                if any(pat in tgt_uri for pat in ("/creation", "/edition", "/impression")):
                    work_to_events.setdefault(src_uri, []).append(tgt_uri)

        counts: Dict[str, int] = {}
        for work_uri, event_uris in work_to_events.items():
            counted = set()
            for event_uri in event_uris:
                for actor_uri in creation_to_actors.get(event_uri, []):
                    counted.add(actor_uri)
            for actor_uri in counted:
                counts[actor_uri] = counts.get(actor_uri, 0) + 1
        return counts

    def get_label(self, uri: str) -> str:
        """Get label for a URI, falling back to local name."""
        vid = self._uri_to_vid.get(uri)
        if vid is not None:
            lbl = self._graph.vs[vid]["label"]
            if lbl:
                return lbl
        return uri.split("/")[-1]

    @property
    def vertex_count(self) -> int:
        return self._graph.vcount()

    @property
    def edge_count(self) -> int:
        return self._graph.ecount()

    def total_doc_entities(self) -> int:
        """Count vertices with is_doc=True."""
        return sum(1 for v in self._graph.vs if v["is_doc"])
