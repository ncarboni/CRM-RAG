"""igraph-native FR walker and bulk materializer.

Replaces the dict-based FR traversal in fr_traversal.py with direct
igraph edge traversal.  Predicate-first: for each FR path, looks up
step-0 predicate in a predicate index, walks only from vertices that
have that predicate.  Skips ~99% of vertices per path.

Adapted from the validated IGraphPathWalker in test_fr_igraph_vs_sparql.py
which matched SPARQL results 5/5 on the MAH dataset.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import igraph as ig

from crm_rag.fundamental_relationships import (
    FundamentalRelationship,
    Step,
    build_fully_expanded,
)

logger = logging.getLogger(__name__)


def _local_name(uri: str) -> str:
    """Extract local name from a full URI."""
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


def _build_property_families(
    property_children: Dict[str, List[str]],
) -> Dict[str, frozenset]:
    """Build transitive descendant sets for subPropertyOf resolution.

    Args:
        property_children: parent_local_name -> [direct child local names]
            from crm_taxonomy.json "propertyChildren" section.

    Returns:
        Dict mapping each property -> frozenset({self} | all descendants).
        Properties not in the hierarchy map to frozenset({self}).
    """
    cache: Dict[str, Set[str]] = {}

    def _descendants(prop: str) -> Set[str]:
        if prop in cache:
            return cache[prop]
        children = property_children.get(prop, [])
        desc: Set[str] = set(children)
        for child in children:
            desc |= _descendants(child)
        cache[prop] = desc
        return desc

    # Collect every property mentioned (as parent or child)
    all_props: Set[str] = set(property_children.keys())
    for children_list in property_children.values():
        all_props.update(children_list)

    result: Dict[str, frozenset] = {}
    for prop in all_props:
        result[prop] = frozenset({prop} | _descendants(prop))
    return result


class StepTrieNode:
    """Trie node representing a single FR step (property + recursive flag).

    Children are keyed by (property, recursive) tuples.  Terminal nodes
    mark the end of at least one complete FR path.
    """

    __slots__ = ("property", "recursive", "children", "is_terminal")

    def __init__(self, prop: str, recursive: bool):
        self.property = prop
        self.recursive = recursive
        self.children: Dict[Tuple[str, bool], "StepTrieNode"] = {}
        self.is_terminal = False


class StepTrieRoot:
    """Sentinel root of a step trie (has no step of its own)."""

    __slots__ = ("children", "is_terminal")

    def __init__(self):
        self.children: Dict[Tuple[str, bool], StepTrieNode] = {}
        self.is_terminal = False


class IGraphFRWalker:
    """Walk FR property paths on an igraph knowledge graph.

    Builds two indexes at init time:
      - _pred_to_eids: predicate local name -> [edge IDs]
      - _pred_to_sources: predicate local name -> set of source vertex IDs
      - _inv_pred_to_targets: inverse predicate -> set of target vertex IDs

    Supports subPropertyOf resolution: when matching a predicate P,
    also matches any sub-property of P (transitively).
    """

    def __init__(
        self,
        graph: ig.Graph,
        inverse_map: Dict[str, str],
        property_families: Optional[Dict[str, frozenset]] = None,
    ):
        self.g = graph
        self.inv = inverse_map
        # property -> frozenset({property, sub1, sub2, ...})
        self._prop_family = property_families or {}

        logger.info("Building predicate index for FR walker ...")
        self._pred_to_eids: Dict[str, List[int]] = {}
        self._pred_to_sources: Dict[str, Set[int]] = {}
        self._inv_pred_to_targets: Dict[str, Set[int]] = {}

        # Per-vertex predicate adjacency: O(1) lookups instead of O(degree) scans
        self._out_by_pred: Dict[int, Dict[str, Set[int]]] = {}
        self._in_by_pred: Dict[int, Dict[str, Set[int]]] = {}

        for e in self.g.es:
            pred_local = _local_name(e["predicate"])
            self._pred_to_eids.setdefault(pred_local, []).append(e.index)
            self._pred_to_sources.setdefault(pred_local, set()).add(e.source)

            # Outgoing: source → pred → {targets}
            src_map = self._out_by_pred.get(e.source)
            if src_map is None:
                src_map = {}
                self._out_by_pred[e.source] = src_map
            tgt_set = src_map.get(pred_local)
            if tgt_set is None:
                tgt_set = set()
                src_map[pred_local] = tgt_set
            tgt_set.add(e.target)

            # Incoming: target → pred → {sources}
            tgt_map = self._in_by_pred.get(e.target)
            if tgt_map is None:
                tgt_map = {}
                self._in_by_pred[e.target] = tgt_map
            src_set = tgt_map.get(pred_local)
            if src_set is None:
                src_set = set()
                tgt_map[pred_local] = src_set
            src_set.add(e.source)

        # Build inverse target index: for each predicate P with inverse P_i,
        # record target vertices of P as "sources reachable via P_i"
        for pred_local, eids in self._pred_to_eids.items():
            inv_local = self.inv.get(pred_local)
            if inv_local:
                targets = set()
                for eid in eids:
                    targets.add(self.g.es[eid].target)
                if targets:
                    existing = self._inv_pred_to_targets.get(inv_local, set())
                    self._inv_pred_to_targets[inv_local] = existing | targets

        n_families = sum(1 for f in self._prop_family.values() if len(f) > 1)
        logger.info(
            f"Predicate index: {len(self._pred_to_eids)} unique predicates, "
            f"{self.g.ecount()} edges total, "
            f"{len(self._out_by_pred)} vertices with outgoing pred index, "
            f"{n_families} properties with sub-property expansion"
        )

    def follow_predicate(self, vid: int, prop_local: str) -> Set[int]:
        """Follow a single predicate from a vertex.

        Checks outgoing edges for prop_local (+ sub-properties) AND incoming
        edges for the inverse of prop_local (+ sub-properties of the inverse).

        Uses per-vertex predicate dicts for O(|family|) lookups instead of
        O(degree) edge scans.
        """
        targets: Set[int] = set()

        # Expand prop_local to include all sub-properties
        prop_family = self._prop_family.get(prop_local, frozenset({prop_local}))

        # Expand inverse to include all sub-properties of the inverse
        inv_local = self.inv.get(prop_local)
        inv_family = (
            self._prop_family.get(inv_local, frozenset({inv_local}))
            if inv_local
            else frozenset()
        )

        # Outgoing: dict lookup per family member
        out_map = self._out_by_pred.get(vid)
        if out_map:
            for p in prop_family:
                t = out_map.get(p)
                if t:
                    targets |= t

        # Incoming: dict lookup per inverse family member
        if inv_family:
            in_map = self._in_by_pred.get(vid)
            if in_map:
                for p in inv_family:
                    s = in_map.get(p)
                    if s:
                        targets |= s

        return targets

    def walk_path(self, start_vid: int, steps: List[Step]) -> Set[int]:
        """Walk a sequence of Steps from a starting vertex.

        For recursive steps (*), zero-or-more semantics: the start nodes
        are always included in the result (zero case).
        """
        current = {start_vid}

        for step in steps:
            prop_local = step.property

            if step.recursive:
                # Zero-or-more: BFS, current nodes ARE valid results (zero case)
                visited = set(current)
                frontier = set(current)
                while frontier:
                    next_frontier = set()
                    for vid in frontier:
                        reached = self.follow_predicate(vid, prop_local)
                        for r in reached:
                            if r not in visited:
                                visited.add(r)
                                next_frontier.add(r)
                    frontier = next_frontier
                current = visited
            else:
                # Single hop
                next_set = set()
                for vid in current:
                    next_set.update(self.follow_predicate(vid, prop_local))
                current = next_set

            if not current:
                return set()

        return current

    def get_step0_sources(self, prop_local: str) -> Set[int]:
        """Get vertex IDs that have step-0 predicate (outgoing or incoming inverse).

        Expands prop_local to include all sub-properties, then returns the
        union of:
        - Vertices with outgoing edges matching any member of the family
        - Vertices with incoming edges matching inverse of any family member
        """
        prop_family = self._prop_family.get(prop_local, frozenset({prop_local}))

        sources: Set[int] = set()
        for p in prop_family:
            sources |= self._pred_to_sources.get(p, set())
            sources |= self._inv_pred_to_targets.get(p, set())
        return sources


# ── Trie construction & walking ─────────────────────────────────────


def _build_step_trie(paths) -> StepTrieRoot:
    """Build a step trie from a list of PropertyPath objects.

    Each path's steps are inserted into the trie keyed by (property, recursive).
    Paths sharing a common prefix share trie nodes, so the expensive BFS at each
    node is computed once and reused across all children.
    """
    root = StepTrieRoot()
    for path in paths:
        if not path.steps:
            root.is_terminal = True
            continue
        node = root
        for step in path.steps:
            key = (step.property, step.recursive)
            if key not in node.children:
                node.children[key] = StepTrieNode(step.property, step.recursive)
            node = node.children[key]
        node.is_terminal = True
    return root


def _count_trie_nodes(root: StepTrieRoot) -> int:
    """Count total nodes in a step trie (excluding root sentinel)."""
    count = 0
    stack = list(root.children.values())
    while stack:
        node = stack.pop()
        count += 1
        stack.extend(node.children.values())
    return count


def _apply_step(walker: IGraphFRWalker, frontier: Set[int],
                prop: str, recursive: bool) -> Set[int]:
    """Apply a single FR step to a frontier of vertex IDs.

    For recursive steps, performs BFS closure (zero-or-more semantics).
    For non-recursive steps, performs a single hop.
    """
    if recursive:
        visited = set(frontier)
        bfs_front = set(frontier)
        while bfs_front:
            next_front: Set[int] = set()
            for vid in bfs_front:
                for r in walker.follow_predicate(vid, prop):
                    if r not in visited:
                        visited.add(r)
                        next_front.add(r)
            bfs_front = next_front
        return visited
    else:
        result: Set[int] = set()
        for vid in frontier:
            result.update(walker.follow_predicate(vid, prop))
        return result


def _walk_trie(walker: IGraphFRWalker, start_vid: int,
               applicable_children: Dict[Tuple[str, bool], StepTrieNode]
               ) -> Set[int]:
    """Walk a step trie from a single source vertex, collecting all targets.

    At each trie node, the frontier is computed once and shared across all
    children.  Terminal nodes contribute their frontier to the result set.
    """
    targets: Set[int] = set()

    # Iterative DFS: stack of (children_dict, frontier)
    stack: list = [(applicable_children, frozenset({start_vid}))]

    while stack:
        children, frontier = stack.pop()
        for _key, child in children.items():
            next_frontier = _apply_step(walker, frontier, child.property,
                                        child.recursive)
            if not next_frontier:
                continue
            if child.is_terminal:
                targets.update(next_frontier)
            if child.children:
                stack.append((child.children, frozenset(next_frontier)))

    return targets


def materialize_fr_edges(
    kg,
    inverse_map: Dict[str, str],
    fr_definitions: Optional[List[FundamentalRelationship]] = None,
    property_children: Optional[Dict[str, List[str]]] = None,
) -> List[Tuple]:
    """Materialize FR edges on the full igraph using predicate-first traversal.

    For each FR path:
    1. Look up step-0 predicate in the predicate index
    2. Get source vertices that have that predicate (+ sub-properties)
    3. Walk the path only from those vertices

    Args:
        kg: KnowledgeGraph instance (must have RDF triples loaded)
        inverse_map: local_name -> inverse_local_name dict
        fr_definitions: List of FundamentalRelationship objects.
            If None, builds from fundamental_relationships.build_fully_expanded()
        property_children: parent_local_name -> [child local names] dict
            from crm_taxonomy.json for subPropertyOf resolution.
            If None, no sub-property expansion is performed.

    Returns:
        List of (entity_uri, entity_label, fr_stats_dict) tuples
        compatible with kg.add_fr_edges()
    """
    if fr_definitions is None:
        fr_definitions = build_fully_expanded()

    # Build transitive property families for subPropertyOf resolution
    prop_families = (
        _build_property_families(property_children)
        if property_children
        else {}
    )

    walker = IGraphFRWalker(kg._graph, inverse_map, prop_families)

    total_fr_edges = 0
    total_paths = sum(len(fr.paths) for fr in fr_definitions)
    logger.info(f"Materializing FR edges: {len(fr_definitions)} FRs, {total_paths} paths")

    # Accumulate results per source entity: uri -> {fr_id -> set of (target_uri, target_label)}
    entity_fr_targets: Dict[str, Dict[str, Set[Tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(set)
    )
    # Track FR metadata for stats
    fr_meta: Dict[str, Tuple[str, str, str]] = {}  # fr_id -> (label, domain_fc, range_fc)

    # Pre-fetch vertex attributes for batch access (avoids per-vertex dict lookup)
    vs_name = walker.g.vs["name"]
    vs_label = walker.g.vs["label"]
    vs_fc = walker.g.vs["fc"]

    overall_start = time.time()

    for fr in fr_definitions:
        fr_start = time.time()
        fr_id = fr.id
        fr_label = fr.label
        fr_meta[fr_id] = (fr_label, fr.domain_fc, fr.range_fc)
        domain_fc = fr.domain_fc

        # Build step trie for this FR — shared prefixes are traversed once
        trie_root = _build_step_trie(fr.paths)
        n_trie = _count_trie_nodes(trie_root)
        n_steps = sum(len(p.steps) for p in fr.paths)

        # Compute source sets per root child (step-0 predicate routing)
        step0_sources: Dict[Tuple[str, bool], Set[int]] = {}
        for key, child in trie_root.children.items():
            step0_sources[key] = walker.get_step0_sources(child.property)

        # Union all source sets, filter by domain FC
        all_sources: Set[int] = set()
        for s in step0_sources.values():
            all_sources |= s
        domain_sources = {vid for vid in all_sources if vs_fc[vid] == domain_fc}

        fr_pairs = 0

        for vid in domain_sources:
            # Build applicable root children for this vertex
            applicable: Dict[Tuple[str, bool], StepTrieNode] = {}
            for key, child in trie_root.children.items():
                if vid in step0_sources[key]:
                    applicable[key] = child

            if not applicable:
                continue

            targets = _walk_trie(walker, vid, applicable)
            # Also collect targets if root is terminal (empty-path FR)
            if trie_root.is_terminal:
                targets.add(vid)

            if not targets:
                continue

            # Remove self-references
            targets.discard(vid)
            if not targets:
                continue

            source_uri = vs_name[vid]
            source_label = vs_label[vid] or _local_name(source_uri)

            for tvid in targets:
                target_uri = vs_name[tvid]
                target_label = vs_label[tvid] or _local_name(target_uri)
                entity_fr_targets[source_uri][fr_id].add((target_uri, target_label))
                fr_pairs += 1

        elapsed = time.time() - fr_start
        total_fr_edges += fr_pairs
        if fr_pairs > 0:
            logger.info(
                f"  {fr_id} ({fr_label}): {fr_pairs} pairs in {elapsed:.1f}s "
                f"[{len(fr.paths)} paths, {n_steps} steps -> {n_trie} trie nodes]"
            )

    # Convert to the format expected by kg.add_fr_edges()
    all_fr_stats = []
    for entity_uri, fr_dict in entity_fr_targets.items():
        fr_results = []
        for fr_id, target_set in fr_dict.items():
            fr_label_meta = fr_meta[fr_id][0]
            targets = list(target_set)
            fr_results.append({
                "fr_id": fr_id,
                "fr_label": fr_label_meta,
                "targets": targets,
                "total_count": len(targets),
            })

        entity_label = kg._graph.vs[kg._uri_to_vid[entity_uri]]["label"] or _local_name(entity_uri)
        # Determine FC from vertex attribute (set during Phase 2 of build)
        fc = kg._graph.vs[kg._uri_to_vid[entity_uri]]["fc"]
        stats = {
            "fc": fc,
            "fr_results": fr_results,
        }
        all_fr_stats.append((entity_uri, entity_label, stats))

    total_elapsed = time.time() - overall_start
    logger.info(
        f"FR materialization complete: {total_fr_edges} edges from "
        f"{len(all_fr_stats)} entities in {total_elapsed:.1f}s"
    )

    return all_fr_stats


def get_step0_predicates(
    fr_definitions: Optional[List[FundamentalRelationship]] = None,
    property_children: Optional[Dict[str, List[str]]] = None,
) -> Set[str]:
    """Extract the set of all step-0 predicate local names from FR definitions.

    Used by knowledge_graph.get_direct_rdf_predicates() to exclude FR-covered
    predicates from the direct predicate list.

    If property_children is provided, expands each step-0 predicate to include
    all sub-properties (transitively), since those are also matched by the
    walker during materialization.
    """
    if fr_definitions is None:
        fr_definitions = build_fully_expanded()

    step0 = set()
    for fr in fr_definitions:
        for path in fr.paths:
            if path.steps:
                step0.add(path.steps[0].property)

    # Expand with sub-properties so direct predicates section doesn't
    # redundantly show predicates already covered by FR materialization
    if property_children:
        prop_families = _build_property_families(property_children)
        expanded = set()
        for pred in step0:
            expanded |= prop_families.get(pred, frozenset({pred}))
        return expanded

    return step0
