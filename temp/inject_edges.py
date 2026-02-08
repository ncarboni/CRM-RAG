"""
Inject edges from edges.parquet into an existing document_graph.pkl.

Usage:
    uv run python3 temp/inject_edges.py --dataset mah
    uv run python3 temp/inject_edges.py --dataset mah --dry-run
"""

import argparse
import os
import pickle
import sys

# Add parent directory to path so pickle can find graph_document_store.GraphDocument
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.parquet as pq


# CIDOC-CRM relationship weights (same as universal_rag_system.get_relationship_weight)
WEIGHTS = {
    "P89_falls_within": 0.9, "P89i_contains": 0.9,
    "P55_has_current_location": 0.9, "P55i_currently_holds": 0.9,
    "P56_bears_feature": 0.8, "P56i_is_found_on": 0.8,
    "P46_is_composed_of": 0.8, "P46i_forms_part_of": 0.8,
    "P108_has_produced": 0.85, "P108i_was_produced_by": 0.85,
    "P14_carried_out_by": 0.85, "P14i_performed": 0.85,
    "P94_has_created": 0.85, "P94i_was_created_by": 0.85,
    "K24_portray": 0.7, "K24i_is_portrayed_in": 0.7,
    "K34_illustrates": 0.7, "K34i_is_illustrated_by": 0.7,
    "P2_has_type": 0.6, "P2i_is_type_of": 0.6,
    "P67_refers_to": 0.5, "P67i_is_referred_to_by": 0.5,
    "P70_documents": 0.5, "P70i_is_documented_in": 0.5,
    "P4_has_time-span": 0.6, "P4i_is_time-span_of": 0.6,
    "P1_is_identified_by": 0.4, "P1i_identifies": 0.4,
}


def get_weight(predicate_uri):
    local_name = predicate_uri.split('/')[-1].split('#')[-1]
    return WEIGHTS.get(predicate_uri) or WEIGHTS.get(local_name, 0.5)


def main():
    parser = argparse.ArgumentParser(description="Inject edges from parquet into document_graph.pkl")
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g., mah, asinou)")
    parser.add_argument("--dry-run", action="store_true", help="Count edges without modifying pickle")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkl_path = os.path.join(base_dir, "data", "cache", args.dataset, "document_graph.pkl")
    parquet_path = os.path.join(base_dir, "data", "documents", args.dataset, "edges.parquet")

    if not os.path.exists(pkl_path):
        print(f"ERROR: {pkl_path} not found")
        sys.exit(1)
    if not os.path.exists(parquet_path):
        print(f"ERROR: {parquet_path} not found")
        sys.exit(1)

    # Load pickle
    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        docs = pickle.load(f)
    print(f"  {len(docs)} documents loaded")

    # Count existing edges
    existing_edges = sum(len(doc.neighbors) for doc in docs.values())
    print(f"  {existing_edges} existing neighbor entries")

    # Clear existing edges to avoid duplicates
    if existing_edges > 0 and not args.dry_run:
        print("  Clearing existing edges...")
        for doc in docs.values():
            doc.neighbors = []

    # Load parquet
    print(f"Loading {parquet_path}...")
    table = pq.read_table(parquet_path)
    print(f"  {len(table)} triples")

    # Build edges
    doc_uris = set(docs.keys())
    edges_added = 0
    skipped = 0

    for s, p, o in zip(table.column("s"), table.column("p"), table.column("o")):
        s_str, p_str, o_str = s.as_py(), p.as_py(), o.as_py()
        if s_str in doc_uris and o_str in doc_uris and s_str != o_str:
            if not args.dry_run:
                weight = get_weight(p_str)
                pred_name = p_str.split('/')[-1].split('#')[-1]
                docs[s_str].neighbors.append({
                    "doc_id": o_str, "edge_type": pred_name, "weight": weight
                })
                docs[o_str].neighbors.append({
                    "doc_id": s_str, "edge_type": pred_name, "weight": weight
                })
            edges_added += 1
        else:
            skipped += 1

    print(f"  {edges_added} edges ({'would be ' if args.dry_run else ''}added)")
    print(f"  {skipped} triples skipped (endpoints not in document store)")

    if args.dry_run:
        print("Dry run — no changes made.")
        return

    # Save
    print(f"Saving {pkl_path}...")
    with open(pkl_path, "wb") as f:
        pickle.dump(docs, f)

    # Verify
    total_neighbors = sum(len(doc.neighbors) for doc in docs.values())
    print(f"  Done. {total_neighbors} total neighbor entries ({edges_added} edges × 2 bidirectional)")


if __name__ == "__main__":
    main()
