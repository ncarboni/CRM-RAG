"""
Test FR path equivalence: igraph traversal vs SPARQL property paths.

Compares the igraph walker (which resolves inverse properties) against the
actual SPARQL queries from fr_sparql_by_entity.sparql.

Usage:
    uv run python scripts/FR/test_fr_igraph_vs_sparql.py [--dataset asinou|mah] [--endpoint URL]

Requires:
    - data/cache/{dataset}/knowledge_graph.pkl (igraph with RDF triples)
    - data/labels/inverse_properties.json
    - Running SPARQL endpoint (Fuseki)
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import igraph as ig
from SPARQLWrapper import JSON, SPARQLWrapper

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "FR"))

from fundamental_relationships import Step, PropertyPath, FundamentalRelationship, S, P, FR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inverse property lookup
# ---------------------------------------------------------------------------


def load_inverse_map(path: Path) -> Dict[str, str]:
    """Load inverse_properties.json -> {local_name: inverse_local_name}."""
    with open(path) as f:
        raw = json.load(f)
    inv = {}
    for uri_a, uri_b in raw.items():
        local_a = uri_a.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        local_b = uri_b.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
        inv[local_a] = local_b
        inv[local_b] = local_a
    return inv


def _local_name(uri: str) -> str:
    """Extract local name from a full URI."""
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# igraph FR path walker
# ---------------------------------------------------------------------------


class IGraphPathWalker:
    """Walk FR property paths on an igraph knowledge graph."""

    def __init__(self, graph: ig.Graph, inverse_map: Dict[str, str]):
        self.g = graph
        self.inv = inverse_map
        log.info("Building predicate index ...")
        self._pred_to_eids: Dict[str, List[int]] = {}
        for e in self.g.es:
            pred_local = _local_name(e["predicate"])
            self._pred_to_eids.setdefault(pred_local, []).append(e.index)
        log.info(
            f"Predicate index: {len(self._pred_to_eids)} unique predicates, "
            f"{self.g.ecount()} edges total"
        )
        self._uri_to_vid = {v["name"]: v.index for v in self.g.vs}

    def follow_predicate(self, vid: int, prop_local: str) -> Set[int]:
        """Follow a single predicate from a vertex.

        Checks outgoing edges for prop_local AND incoming edges for the
        inverse of prop_local (to handle data that stores only one direction).
        """
        targets = set()
        inv_local = self.inv.get(prop_local)

        # Outgoing: edges where vid is source, predicate matches prop_local
        for eid in self.g.incident(vid, mode="out"):
            e = self.g.es[eid]
            if _local_name(e["predicate"]) == prop_local:
                targets.add(e.target)

        # Incoming: edges where vid is target, predicate matches inverse
        if inv_local:
            for eid in self.g.incident(vid, mode="in"):
                e = self.g.es[eid]
                if _local_name(e["predicate"]) == inv_local:
                    targets.add(e.source)

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

    def run_fr(
        self, fr: FundamentalRelationship, sample_sources: Optional[List[str]] = None
    ) -> Set[Tuple[str, str]]:
        """Run all paths of an FR, return {(source_uri, target_uri)} pairs."""
        results = set()

        if sample_sources is not None:
            source_vids = []
            for uri in sample_sources:
                vid = self._uri_to_vid.get(uri)
                if vid is not None:
                    source_vids.append(vid)
        else:
            source_vids = list(range(self.g.vcount()))

        log.info(
            f"  igraph: running {fr.id} ({len(fr.paths)} paths) "
            f"on {len(source_vids)} source vertices ..."
        )

        for path in fr.paths:
            for vid in source_vids:
                targets = self.walk_path(vid, path.steps)
                source_uri = self.g.vs[vid]["name"]
                for tvid in targets:
                    target_uri = self.g.vs[tvid]["name"]
                    if source_uri != target_uri:
                        results.add((source_uri, target_uri))

        return results

    def uri_to_vid(self, uri: str) -> Optional[int]:
        return self._uri_to_vid.get(uri)


# ---------------------------------------------------------------------------
# SPARQL runner — uses actual queries from fr_sparql_by_entity.sparql
# ---------------------------------------------------------------------------

PREFIX = "PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>\n"


def run_sparql_raw(
    endpoint: str,
    fr_id: str,
    sparql_body: str,
    sample_sources: Optional[List[str]] = None,
    timeout: int = 300,
) -> Set[Tuple[str, str]]:
    """Execute a raw SPARQL query body and return {(s, o)} pairs.

    The sparql_body is the actual UNION content from fr_sparql_by_entity.sparql.
    A VALUES clause is prepended if sample_sources is provided.
    """
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)

    values_clause = ""
    if sample_sources:
        values_list = " ".join(f"<{uri}>" for uri in sample_sources)
        values_clause = f"  VALUES ?s {{ {values_list} }}\n"

    query = f"{PREFIX}SELECT DISTINCT ?s ?o WHERE {{\n{values_clause}{sparql_body}\n}}"

    log.info(
        f"  SPARQL: running {fr_id}"
        f"{f' ({len(sample_sources)} sources)' if sample_sources else ''} ..."
    )

    try:
        sparql.setQuery(query)
        raw = sparql.query().convert()
        results = set()
        for binding in raw["results"]["bindings"]:
            s = binding["s"]["value"]
            o = binding["o"]["value"]
            if s != o:
                results.add((s, o))
        return results
    except Exception as exc:
        log.error(f"  SPARQL query failed for {fr_id}: {exc}")
        return set()


# ---------------------------------------------------------------------------
# Test cases: actual SPARQL from fr_sparql_by_entity.sparql + igraph Steps
# ---------------------------------------------------------------------------
#
# Each test has:
#   - fr_id, label: identification
#   - igraph_fr: FR object with Step definitions (for igraph walker)
#   - sparql_body: the actual UNION block copied from fr_sparql_by_entity.sparql
#   - step0_predicate: for sampling source entities

TEST_CASES = [
    # ── Test 1: Simple ─────────────────────────────────────────────────────
    # FR: thing_concept_a -- has type (Thing -> Concept)
    # 4 paths from fr_sparql_by_entity.sparql lines 1883-1892
    {
        "fr_id": "thing_concept_a",
        "label": "has type (Thing -> Concept)",
        "step0_predicate": "P2_has_type",
        "igraph_fr": FR(
            id="thing_concept_a",
            label="has type",
            domain_fc="Thing",
            range_fc="Concept",
            domain_class="E70",
            range_class="E55",
            paths=[
                P("tca_01", "P2/P127*", S("P2_has_type", "E1", "E55"), S("P127_has_broader_term", "E55", "E55", rec=True)),
                P("tca_02", "P45/P127*", S("P45_consists_of", "E18", "E57"), S("P127_has_broader_term", "E55", "E55", rec=True)),
                P("tca_03", "P92i/P9i*/P33/P68", S("P92i_was_brought_into_existence_by", "E77", "E63"), S("P9i_forms_part_of", "E5", "E5", rec=True), S("P33_used_specific_technique", "E29", "E55"), S("P68_foresees_use_of", "E29", "E57")),
                P("tca_04", "P92i/P9i*/P126", S("P92i_was_brought_into_existence_by", "E77", "E63"), S("P9i_forms_part_of", "E5", "E5", rec=True), S("P126_employed", "E11", "E57")),
            ],
        ),
        "sparql_body": """\
  { ?s crm:P2_has_type/crm:P127_has_broader_term* ?o . }
  UNION
  { ?s crm:P45_consists_of/crm:P127_has_broader_term* ?o . }
  UNION
  { ?s crm:P92i_was_brought_into_existence_by/crm:P9i_forms_part_of*/crm:P33_used_specific_technique/crm:P68_foresees_use_of ?o . }
  UNION
  { ?s crm:P92i_was_brought_into_existence_by/crm:P9i_forms_part_of*/crm:P126_employed ?o . }""",
    },

    # ── Test 2: Single recursion ───────────────────────────────────────────
    # FR: place_place_b -- is part of (Place -> Place)
    # 1 path from fr_sparql_by_entity.sparql line 2229-2231
    {
        "fr_id": "place_place_b",
        "label": "is part of (Place -> Place)",
        "step0_predicate": "P89_falls_within",
        "igraph_fr": FR(
            id="place_place_b",
            label="is part of",
            domain_fc="Place",
            range_fc="Place",
            domain_class="E53",
            range_class="E53",
            paths=[
                P("ppb_01", "P89*", S("P89_falls_within", "E53", "E53", rec=True)),
            ],
        ),
        "sparql_body": """\
  { ?s crm:P89_falls_within* ?o . }""",
    },

    # ── Test 3: Full FR with big UNION ─────────────────────────────────────
    # FR: thing_place_a -- refers to or is about (Thing -> Place)
    # First 12 paths only (from fr_sparql_by_entity.sparql lines 494-517)
    # to keep SPARQL manageable while covering the key patterns:
    # plain refs, refs with P89i_contains*, refs via P53, refs via P128
    {
        "fr_id": "thing_place_a_subset",
        "label": "refers to or is about (Thing -> Place) [12 paths]",
        "step0_predicate": "P62_depicts",
        "igraph_fr": FR(
            id="thing_place_a_subset",
            label="refers to or is about",
            domain_fc="Thing",
            range_fc="Place",
            domain_class="E70",
            range_class="E53",
            paths=[
                P("tpa_01", "P62", S("P62_depicts", "E24", "E53")),
                P("tpa_02", "P62/P89i*", S("P62_depicts", "E24", "E53"), S("P89i_contains", "E53", "E53", rec=True)),
                P("tpa_03", "P67", S("P67_refers_to", "E73", "E53")),
                P("tpa_04", "P67/P89i*", S("P67_refers_to", "E73", "E53"), S("P89i_contains", "E53", "E53", rec=True)),
                P("tpa_05", "P128/P67", S("P128_carries", "E18", "E73"), S("P67_refers_to", "E73", "E53")),
                P("tpa_06", "P128/P67/P89i*", S("P128_carries", "E18", "E73"), S("P67_refers_to", "E73", "E53"), S("P89i_contains", "E53", "E53", rec=True)),
                P("tpa_07", "P62/P53", S("P62_depicts", "E24", "E26"), S("P53_has_former_or_current_location", "E18", "E53")),
                P("tpa_08", "P62/P53/P89i*", S("P62_depicts", "E24", "E26"), S("P53_has_former_or_current_location", "E18", "E53"), S("P89i_contains", "E53", "E53", rec=True)),
                P("tpa_09", "P67/P53", S("P67_refers_to", "E73", "E26"), S("P53_has_former_or_current_location", "E18", "E53")),
                P("tpa_10", "P67/P53/P89i*", S("P67_refers_to", "E73", "E26"), S("P53_has_former_or_current_location", "E18", "E53"), S("P89i_contains", "E53", "E53", rec=True)),
                P("tpa_11", "P128/P67/P53", S("P128_carries", "E18", "E73"), S("P67_refers_to", "E73", "E26"), S("P53_has_former_or_current_location", "E18", "E53")),
                P("tpa_12", "P128/P67/P53/P89i*", S("P128_carries", "E18", "E73"), S("P67_refers_to", "E73", "E26"), S("P53_has_former_or_current_location", "E18", "E53"), S("P89i_contains", "E53", "E53", rec=True)),
            ],
        ),
        "sparql_body": """\
  { ?s crm:P62_depicts ?o . }
  UNION
  { ?s crm:P62_depicts/crm:P89i_contains* ?o . }
  UNION
  { ?s crm:P67_refers_to ?o . }
  UNION
  { ?s crm:P67_refers_to/crm:P89i_contains* ?o . }
  UNION
  { ?s crm:P128_carries/crm:P67_refers_to ?o . }
  UNION
  { ?s crm:P128_carries/crm:P67_refers_to/crm:P89i_contains* ?o . }
  UNION
  { ?s crm:P62_depicts/crm:P53_has_former_or_current_location ?o . }
  UNION
  { ?s crm:P62_depicts/crm:P53_has_former_or_current_location/crm:P89i_contains* ?o . }
  UNION
  { ?s crm:P67_refers_to/crm:P53_has_former_or_current_location ?o . }
  UNION
  { ?s crm:P67_refers_to/crm:P53_has_former_or_current_location/crm:P89i_contains* ?o . }
  UNION
  { ?s crm:P128_carries/crm:P67_refers_to/crm:P53_has_former_or_current_location ?o . }
  UNION
  { ?s crm:P128_carries/crm:P67_refers_to/crm:P53_has_former_or_current_location/crm:P89i_contains* ?o . }""",
    },

    # ── Test 4: Multi-hop through event ────────────────────────────────────
    # FR: thing_actor_d_created_by -- created by (Thing -> Actor)
    # 1 path from fr_sparql_by_entity.sparql lines 1534-1537
    {
        "fr_id": "thing_actor_d_created_by",
        "label": "created by (Thing -> Actor)",
        "step0_predicate": "P92i_was_brought_into_existence_by",
        "igraph_fr": FR(
            id="thing_actor_d_created_by",
            label="created by",
            domain_fc="Thing",
            range_fc="Actor",
            domain_class="E24",
            range_class="E39",
            paths=[
                P(
                    "tadc_01", "P92i/P9i*/P14/P107i*",
                    S("P92i_was_brought_into_existence_by", "E77", "E63"),
                    S("P9i_forms_part_of", "E5", "E5", rec=True),
                    S("P14_carried_out_by", "E7", "E39"),
                    S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
                ),
            ],
        ),
        "sparql_body": """\
  { ?s crm:P92i_was_brought_into_existence_by/crm:P9i_forms_part_of*/crm:P14_carried_out_by/crm:P107i_is_current_or_former_member_of* ?o . }""",
    },

    # ── Test 5: Full thing_actor_d (the "by" FR) ───────────────────────────
    # FR: thing_actor_d -- by (Thing -> Actor)
    # 6 paths from fr_sparql_by_entity.sparql lines 1517-1530
    {
        "fr_id": "thing_actor_d",
        "label": "by (Thing -> Actor)",
        "step0_predicate": "P92i_was_brought_into_existence_by",
        "igraph_fr": FR(
            id="thing_actor_d",
            label="by",
            domain_fc="Thing",
            range_fc="Actor",
            domain_class="E24",
            range_class="E39",
            paths=[
                P("tad_01", "P92i/P9i*/P14/P107i*", S("P92i_was_brought_into_existence_by", "E77", "E63"), S("P9i_forms_part_of", "E5", "E5", rec=True), S("P14_carried_out_by", "E7", "E39"), S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
                P("tad_02", "P16i/P9i*/P14/P107i*", S("P16i_was_used_for", "E70", "E7"), S("P9i_forms_part_of", "E5", "E5", rec=True), S("P14_carried_out_by", "E7", "E39"), S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
                P("tad_03", "P31i/P9i*/P14/P107i*", S("P31i_was_modified_by", "E24", "E11"), S("P9i_forms_part_of", "E5", "E5", rec=True), S("P14_carried_out_by", "E7", "E39"), S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
                P("tad_04", "P12i/P9i*/P11/P107i*", S("P12i_was_present_at", "E77", "E5"), S("P9i_forms_part_of", "E5", "E5", rec=True), S("P11_had_participant", "E5", "E39"), S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
                P("tad_05", "P24i/P22/P107i*", S("P24i_changed_ownership_through", "E18", "E8"), S("P22_transferred_title_to", "E8", "E39"), S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
                P("tad_06", "P51/P107i*", S("P51_has_former_or_current_owner", "E18", "E39"), S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
            ],
        ),
        "sparql_body": """\
  { ?s crm:P92i_was_brought_into_existence_by/crm:P9i_forms_part_of*/crm:P14_carried_out_by/crm:P107i_is_current_or_former_member_of* ?o . }
  UNION
  { ?s crm:P16i_was_used_for/crm:P9i_forms_part_of*/crm:P14_carried_out_by/crm:P107i_is_current_or_former_member_of* ?o . }
  UNION
  { ?s crm:P31i_was_modified_by/crm:P9i_forms_part_of*/crm:P14_carried_out_by/crm:P107i_is_current_or_former_member_of* ?o . }
  UNION
  { ?s crm:P12i_was_present_at/crm:P9i_forms_part_of*/crm:P11_had_participant/crm:P107i_is_current_or_former_member_of* ?o . }
  UNION
  { ?s crm:P24i_changed_ownership_through/crm:P22_transferred_title_to/crm:P107i_is_current_or_former_member_of* ?o . }
  UNION
  { ?s crm:P51_has_former_or_current_owner/crm:P107i_is_current_or_former_member_of* ?o . }""",
    },
]


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_results(
    fr_id: str,
    igraph_results: Set[Tuple[str, str]],
    sparql_results: Set[Tuple[str, str]],
) -> dict:
    """Compare igraph vs SPARQL result sets, return comparison report."""
    both = igraph_results & sparql_results
    igraph_only = igraph_results - sparql_results
    sparql_only = sparql_results - igraph_results

    report = {
        "fr_id": fr_id,
        "igraph_count": len(igraph_results),
        "sparql_count": len(sparql_results),
        "both_count": len(both),
        "igraph_only_count": len(igraph_only),
        "sparql_only_count": len(sparql_only),
        "match": igraph_only == set() and sparql_only == set(),
    }

    if igraph_only:
        report["igraph_only_sample"] = sorted(igraph_only)[:10]
    if sparql_only:
        report["sparql_only_sample"] = sorted(sparql_only)[:10]

    return report


def pick_sample_sources(
    walker: IGraphPathWalker, step0_predicate: str, n: int = 50
) -> List[str]:
    """Pick N sample source URIs that have edges matching step0_predicate.

    Checks both outgoing for the predicate and incoming for its inverse,
    to find entities regardless of which direction the data stores.
    """
    inv_local = walker.inv.get(step0_predicate)
    candidates = set()

    for e in walker.g.es:
        pred_local = _local_name(e["predicate"])
        # Outgoing match: source has this predicate
        if pred_local == step0_predicate:
            candidates.add(walker.g.vs[e.source]["name"])
        # Incoming inverse match: target would be the "source" in FR terms
        elif inv_local and pred_local == inv_local:
            candidates.add(walker.g.vs[e.target]["name"])
        if len(candidates) >= n * 3:
            break

    candidates = sorted(candidates)
    if len(candidates) > n:
        step = len(candidates) // n
        candidates = candidates[::step][:n]

    log.info(f"  Sampled {len(candidates)} source URIs (step0: {step0_predicate})")
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Test FR path equivalence: igraph vs SPARQL"
    )
    parser.add_argument(
        "--dataset", default="asinou", help="Dataset name (default: asinou)",
    )
    parser.add_argument(
        "--endpoint", default=None, help="SPARQL endpoint URL (overrides datasets.yaml)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Number of source entities to sample per test (default: 50)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run on ALL source entities (slow, use for final validation)",
    )
    args = parser.parse_args()

    # Resolve paths
    kg_path = PROJECT_ROOT / "data" / "cache" / args.dataset / "knowledge_graph.pkl"
    inv_path = PROJECT_ROOT / "data" / "labels" / "inverse_properties.json"

    if not kg_path.exists():
        log.error(f"Knowledge graph not found: {kg_path}")
        sys.exit(1)

    # Resolve SPARQL endpoint
    endpoint = args.endpoint
    if not endpoint:
        import yaml
        ds_config = PROJECT_ROOT / "config" / "datasets.yaml"
        with open(ds_config) as f:
            cfg = yaml.safe_load(f)
        ds = cfg.get("datasets", {}).get(args.dataset)
        if not ds:
            log.error(f"Dataset '{args.dataset}' not found in datasets.yaml")
            sys.exit(1)
        endpoint = ds["endpoint"]

    log.info(f"Dataset:  {args.dataset}")
    log.info(f"Graph:    {kg_path}")
    log.info(f"Endpoint: {endpoint}")

    # Load igraph
    log.info("Loading knowledge graph ...")
    g = ig.Graph.Read_Pickle(str(kg_path))
    log.info(f"  {g.vcount()} vertices, {g.ecount()} edges")

    edge_types = {}
    for e in g.es:
        et = e["edge_type"]
        edge_types[et] = edge_types.get(et, 0) + 1
    log.info(f"  Edge types: {edge_types}")

    # Load inverse map
    log.info("Loading inverse properties ...")
    inv_map = load_inverse_map(inv_path)
    log.info(f"  {len(inv_map)} inverse mappings")

    # Build walker
    walker = IGraphPathWalker(g, inv_map)

    # Run tests
    print("\n" + "=" * 72)
    print("FR PATH EQUIVALENCE TEST: igraph walker vs actual SPARQL")
    print("=" * 72)

    all_reports = []
    for tc in TEST_CASES:
        fr_id = tc["fr_id"]
        igraph_fr = tc["igraph_fr"]

        print(f"\n{'─' * 72}")
        print(f"Test: {fr_id}")
        print(f"  Label: {tc['label']}")
        print(f"  igraph paths: {len(igraph_fr.paths)}")
        print(f"  SPARQL (from fr_sparql_by_entity.sparql):")
        for line in tc["sparql_body"].strip().split("\n")[:6]:
            print(f"    {line.strip()}")
        if tc["sparql_body"].strip().count("\n") > 5:
            print(f"    ... ({tc['sparql_body'].count('UNION')} UNION clauses total)")
        print(f"{'─' * 72}")

        # Pick sample sources
        if args.full:
            sample = None
            log.info("  Running on ALL vertices (--full mode)")
        else:
            sample = pick_sample_sources(
                walker, tc["step0_predicate"], n=args.sample_size
            )
            if not sample:
                print("  SKIP: no matching source entities found")
                continue

        # Run igraph traversal
        t0 = time.time()
        igraph_results = walker.run_fr(igraph_fr, sample_sources=sample)
        t_igraph = time.time() - t0

        # Run actual SPARQL from fr_sparql_by_entity.sparql
        t0 = time.time()
        sparql_results = run_sparql_raw(
            endpoint, fr_id, tc["sparql_body"], sample_sources=sample
        )
        t_sparql = time.time() - t0

        # Compare
        report = compare_results(fr_id, igraph_results, sparql_results)
        report["igraph_time_s"] = round(t_igraph, 2)
        report["sparql_time_s"] = round(t_sparql, 2)
        all_reports.append(report)

        # Print results
        status = "MATCH" if report["match"] else "MISMATCH"
        print(f"\n  Result:       {status}")
        print(f"  igraph pairs: {report['igraph_count']:,}  ({t_igraph:.2f}s)")
        print(f"  SPARQL pairs: {report['sparql_count']:,}  ({t_sparql:.2f}s)")
        print(f"  Both:         {report['both_count']:,}")
        print(f"  igraph only:  {report['igraph_only_count']:,}")
        print(f"  SPARQL only:  {report['sparql_only_count']:,}")

        if report.get("igraph_only_sample"):
            print(f"\n  igraph-only sample:")
            for s, o in report["igraph_only_sample"][:5]:
                s_label = g.vs[walker.uri_to_vid(s)]["label"] if walker.uri_to_vid(s) is not None else "?"
                o_label = g.vs[walker.uri_to_vid(o)]["label"] if walker.uri_to_vid(o) is not None else "?"
                print(f"    {s_label}  -->  {o_label}")
                print(f"      s: {s}")
                print(f"      o: {o}")

        if report.get("sparql_only_sample"):
            print(f"\n  SPARQL-only sample:")
            for s, o in report["sparql_only_sample"][:5]:
                s_vid = walker.uri_to_vid(s)
                o_vid = walker.uri_to_vid(o)
                s_label = g.vs[s_vid]["label"] if s_vid is not None else "?"
                o_label = g.vs[o_vid]["label"] if o_vid is not None else "?"
                s_in_graph = "in graph" if s_vid is not None else "NOT in graph"
                o_in_graph = "in graph" if o_vid is not None else "NOT in graph"
                print(f"    {s_label}  -->  {o_label}")
                print(f"      s: {s} ({s_in_graph})")
                print(f"      o: {o} ({o_in_graph})")

    # Summary
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    match_count = sum(1 for r in all_reports if r["match"])
    print(f"\n  {match_count}/{len(all_reports)} tests matched exactly\n")
    for r in all_reports:
        status = "MATCH" if r["match"] else "MISMATCH"
        print(
            f"  {r['fr_id']:35s}  {status:10s}  "
            f"ig={r['igraph_count']:>6,}  sq={r['sparql_count']:>6,}  "
            f"ig_only={r['igraph_only_count']:>4}  sq_only={r['sparql_only_count']:>4}  "
            f"ig_t={r['igraph_time_s']:.1f}s  sq_t={r['sparql_time_s']:.1f}s"
        )

    # Save detailed report
    report_path = (
        PROJECT_ROOT / "scripts" / "FR" / f"equivalence_report_{args.dataset}.json"
    )
    with open(report_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n  Detailed report saved to {report_path}")


if __name__ == "__main__":
    main()
