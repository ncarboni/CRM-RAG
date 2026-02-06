#!/usr/bin/env python3
"""
Benchmark Script for Document Generation Approaches

Compares approaches for generating entity documents:
1. Individual SPARQL queries (universal_rag_system.py legacy path)
2. Bulk TTL export + rdflib parsing (bulk_generate_documents.py)
3. Batch SPARQL queries with VALUES clause (universal_rag_system.py batch path)
4. Response format comparison: JSON vs TSV for batch queries

Usage:
    # Run all benchmarks on asinou (small dataset for testing)
    python benchmarks/benchmark_document_generation.py --dataset asinou

    # Run specific benchmark
    python benchmarks/benchmark_document_generation.py --dataset asinou --approach individual
    python benchmarks/benchmark_document_generation.py --dataset asinou --approach bulk
    python benchmarks/benchmark_document_generation.py --dataset asinou --approach batch

    # Limit number of entities (for quick tests)
    python benchmarks/benchmark_document_generation.py --dataset asinou --limit 100

    # Test different batch sizes for batch approach
    python benchmarks/benchmark_document_generation.py --dataset asinou --approach batch --batch-sizes 100,500,1000

Output:
    benchmarks/results/<dataset>_<timestamp>/
        - report.json       : Machine-readable detailed results
        - report.md         : Human-readable summary
        - individual.json   : Raw timing data for individual queries
        - bulk.json         : Raw timing data for bulk approach
        - batch.json        : Raw timing data for batch approach
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from SPARQLWrapper import SPARQLWrapper, JSON, TSV, POST
from config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentGenerationBenchmark:
    """Benchmark different document generation approaches."""

    def __init__(self, dataset_id: str, endpoint: str, output_dir: Path):
        self.dataset_id = dataset_id
        self.endpoint = endpoint
        self.output_dir = output_dir
        self.sparql = SPARQLWrapper(endpoint)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)

        # Results storage
        self.results = {
            "metadata": {
                "dataset": dataset_id,
                "endpoint": endpoint,
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
            },
            "entity_count": 0,
            "approaches": {}
        }

    def _load_ontology_classes(self) -> set:
        """Load ontology class local names from data/labels/ontology_classes.json"""
        classes_file = Path(__file__).parent.parent / "data" / "labels" / "ontology_classes.json"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                return set(json.load(f))
        logger.warning(f"Ontology classes file not found: {classes_file}")
        return set()

    def _is_ontology_class(self, uri: str, ontology_classes: set) -> bool:
        """
        Check if URI is an ontology class by extracting local name.
        Matches logic in universal_rag_system.py:is_technical_class_name()
        """
        # Extract local name from URI (after last / or #)
        local_name = uri.split('/')[-1].split('#')[-1]
        return local_name in ontology_classes

    def get_all_entities(self, limit: Optional[int] = None) -> List[str]:
        """
        Get list of entity URIs that have literal values.

        This matches the logic in both:
        - universal_rag_system.py:get_all_entities()
        - bulk_generate_documents.py:build_indexes()

        Entities are subjects that have literal properties, excluding
        ontology schema elements (classes, properties).
        """
        # Load ontology classes from data/labels/ontology_classes.json
        ontology_classes = self._load_ontology_classes()
        logger.info(f"Loaded {len(ontology_classes)} ontology class names for filtering")

        query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT DISTINCT ?entity WHERE {
            ?entity ?p ?o .
            FILTER(isLiteral(?o))

            # Exclude ontology schema elements by type
            FILTER NOT EXISTS {
                ?entity rdf:type ?type .
                VALUES ?type {
                    rdfs:Class
                    owl:Class
                    rdf:Property
                    owl:ObjectProperty
                    owl:DatatypeProperty
                    owl:AnnotationProperty
                }
            }
        }
        """
        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Fetching entities with literals (limit={limit})...")
        start = time.time()

        self.sparql.setQuery(query)
        results = self.sparql.query().convert()

        # Filter out ontology classes by checking local name against ontology_classes.json
        all_entities = [r["entity"]["value"] for r in results["results"]["bindings"]]
        entities = []
        skipped = 0

        for uri in all_entities:
            if self._is_ontology_class(uri, ontology_classes):
                skipped += 1
                continue
            entities.append(uri)

        elapsed = time.time() - start

        logger.info(f"Found {len(entities)} entities in {elapsed:.2f}s (skipped {skipped} ontology classes)")
        self.results["entity_count"] = len(entities)
        self.results["metadata"]["entity_list_time_sec"] = elapsed
        self.results["metadata"]["ontology_classes_skipped"] = skipped

        return entities

    def benchmark_individual_queries(self, entities: List[str], depth: int = 2) -> Dict[str, Any]:
        """
        Benchmark Approach 1: Individual SPARQL queries per entity.

        This simulates what universal_rag_system.py:get_entity_context() actually does:
        - Bidirectional queries (outgoing AND incoming)
        - Multi-hop traversal through related entities

        For each entity, queries:
        1. Outgoing: <entity> ?p ?o
        2. Incoming: ?s ?p <entity>
        3. Recursively for connected URIs up to depth
        """
        logger.info(f"=== Benchmarking INDIVIDUAL QUERIES ({len(entities)} entities, depth={depth}) ===")

        total_queries = 0
        total_triples = 0
        entity_timings = []
        errors = []

        for i, entity_uri in enumerate(entities):
            entity_start = time.time()
            entity_queries = 0
            entity_triples = 0
            visited = set()

            # BFS traversal with depth limit
            to_visit = [(entity_uri, 0)]  # (uri, current_depth)

            while to_visit:
                uri, current_depth = to_visit.pop(0)

                if uri in visited or current_depth > depth:
                    continue
                visited.add(uri)

                # Query 1: Outgoing relationships
                outgoing_query = f"""
                SELECT ?p ?o WHERE {{
                    <{uri}> ?p ?o .
                    FILTER(isURI(?o))
                }}
                """
                try:
                    self.sparql.setQuery(outgoing_query)
                    results = self.sparql.query().convert()
                    entity_queries += 1

                    for binding in results["results"]["bindings"]:
                        entity_triples += 1
                        obj = binding["o"]["value"]
                        if current_depth < depth and obj not in visited:
                            to_visit.append((obj, current_depth + 1))
                except Exception as e:
                    errors.append({"entity": uri, "query": "outgoing", "error": str(e)})

                # Query 2: Incoming relationships
                incoming_query = f"""
                SELECT ?s ?p WHERE {{
                    ?s ?p <{uri}> .
                    FILTER(isURI(?s))
                }}
                """
                try:
                    self.sparql.setQuery(incoming_query)
                    results = self.sparql.query().convert()
                    entity_queries += 1

                    for binding in results["results"]["bindings"]:
                        entity_triples += 1
                        subj = binding["s"]["value"]
                        if current_depth < depth and subj not in visited:
                            to_visit.append((subj, current_depth + 1))
                except Exception as e:
                    errors.append({"entity": uri, "query": "incoming", "error": str(e)})

            entity_elapsed = time.time() - entity_start
            entity_timings.append(entity_elapsed)
            total_queries += entity_queries
            total_triples += entity_triples

            if (i + 1) % 10 == 0:
                avg_so_far = statistics.mean(entity_timings)
                logger.info(f"  Progress: {i+1}/{len(entities)} entities, "
                           f"avg={avg_so_far*1000:.1f}ms/entity, "
                           f"queries={total_queries}, triples={total_triples}")

        total_time = sum(entity_timings)
        result = {
            "approach": "individual",
            "entity_count": len(entities),
            "depth": depth,
            "total_time_sec": total_time,
            "total_queries": total_queries,
            "avg_queries_per_entity": total_queries / len(entities),
            "avg_time_per_entity_ms": (total_time / len(entities)) * 1000,
            "median_time_ms": statistics.median(entity_timings) * 1000,
            "min_time_ms": min(entity_timings) * 1000,
            "max_time_ms": max(entity_timings) * 1000,
            "stddev_ms": statistics.stdev(entity_timings) * 1000 if len(entity_timings) > 1 else 0,
            "total_triples": total_triples,
            "avg_triples_per_entity": total_triples / len(entities),
            "error_count": len(errors),
            "errors": errors[:10],
            "throughput_entities_per_sec": len(entities) / total_time if total_time > 0 else 0,
        }

        logger.info(f"  Total: {total_time:.2f}s, {total_queries} queries, "
                   f"Avg: {result['avg_time_per_entity_ms']:.1f}ms/entity")
        return result

    def benchmark_bulk_export(self, entities: List[str]) -> Dict[str, Any]:
        """
        Benchmark Approach 2: Bulk CONSTRUCT export.
        This simulates what bulk_generate_documents.py:export_from_sparql() does.
        """
        logger.info(f"=== Benchmarking BULK EXPORT ===")

        # Step 1: Export all triples using requests.post (same as bulk_generate_documents.py)
        export_query = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"

        start_export = time.time()
        try:
            response = requests.post(
                self.endpoint,
                data={"query": export_query},
                headers={"Accept": "text/turtle"},
                timeout=3600  # 1 hour timeout for large datasets
            )
            response.raise_for_status()
            raw_results = response.content
            export_time = time.time() - start_export

            export_size = len(raw_results)

            logger.info(f"  Export: {export_time:.2f}s, size={export_size/1024/1024:.1f}MB")

        except Exception as e:
            export_time = time.time() - start_export
            logger.error(f"  Export failed: {e}")
            return {
                "approach": "bulk",
                "error": str(e),
                "export_time_sec": export_time,
            }

        # Step 2: Parse with rdflib
        start_parse = time.time()
        try:
            from rdflib import Graph
            g = Graph()

            # Suppress rdflib warnings for invalid date formats (BCE dates, "Unknown-Format", etc.)
            # These are common in cultural heritage data and rdflib still loads them as strings
            rdflib_logger = logging.getLogger('rdflib.term')
            original_level = rdflib_logger.level
            rdflib_logger.setLevel(logging.ERROR)

            try:
                g.parse(data=raw_results, format="turtle")
            finally:
                rdflib_logger.setLevel(original_level)

            parse_time = time.time() - start_parse
            triple_count = len(g)

            logger.info(f"  Parse: {parse_time:.2f}s, triples={triple_count}")

        except Exception as e:
            parse_time = time.time() - start_parse
            logger.error(f"  Parse failed: {e}")
            return {
                "approach": "bulk",
                "error": str(e),
                "export_time_sec": export_time,
                "parse_time_sec": parse_time,
            }

        # Step 3: Query for each entity (in-memory)
        start_query = time.time()
        entity_triple_counts = []

        for entity_uri in entities:
            from rdflib import URIRef
            triples = list(g.triples((URIRef(entity_uri), None, None)))
            entity_triple_counts.append(len(triples))

        query_time = time.time() - start_query
        logger.info(f"  In-memory query: {query_time:.2f}s for {len(entities)} entities")

        total_time = export_time + parse_time + query_time

        result = {
            "approach": "bulk",
            "entity_count": len(entities),
            "total_time_sec": total_time,
            "export_time_sec": export_time,
            "parse_time_sec": parse_time,
            "query_time_sec": query_time,
            "export_size_bytes": export_size,
            "export_size_mb": export_size / 1024 / 1024,
            "total_triples": triple_count,
            "avg_triples_per_entity": statistics.mean(entity_triple_counts) if entity_triple_counts else 0,
            "throughput_entities_per_sec": len(entities) / total_time if total_time > 0 else 0,
        }

        logger.info(f"  Total: {total_time:.2f}s")
        return result

    def benchmark_batch_queries(self, entities: List[str], batch_sizes: List[int] = None,
                                 depth: int = 2) -> Dict[str, Any]:
        """
        Benchmark Approach 3: Batch SPARQL queries using VALUES clause.

        This tests batch queries with:
        - Bidirectional queries (outgoing AND incoming)
        - Multi-hop traversal (depth parameter)

        Strategy:
        - Depth 0: Batch query outgoing+incoming for all input entities
        - Collect connected URIs
        - Depth 1: Batch query for those URIs
        - Repeat until depth reached
        """
        if batch_sizes is None:
            batch_sizes = [100, 500, 1000]

        logger.info(f"=== Benchmarking BATCH QUERIES ({len(entities)} entities, depth={depth}) ===")

        all_batch_results = {}

        for batch_size in batch_sizes:
            logger.info(f"  Testing batch_size={batch_size}...")

            total_time = 0
            total_queries = 0
            total_triples = 0
            errors = []

            visited = set()
            current_level_uris = set(entities)

            for current_depth in range(depth + 1):
                # Filter out already visited
                uris_to_query = [u for u in current_level_uris if u not in visited]
                if not uris_to_query:
                    break

                visited.update(uris_to_query)
                next_level_uris = set()

                logger.info(f"    Depth {current_depth}: {len(uris_to_query)} URIs to query")

                # Process in batches
                for i in range(0, len(uris_to_query), batch_size):
                    batch = uris_to_query[i:i + batch_size]
                    values_clause = " ".join(f"<{uri}>" for uri in batch)

                    # Query 1: Outgoing relationships
                    outgoing_query = f"""
                    SELECT ?entity ?p ?o WHERE {{
                        VALUES ?entity {{ {values_clause} }}
                        ?entity ?p ?o .
                        FILTER(isURI(?o))
                    }}
                    """

                    start = time.time()
                    try:
                        self.sparql.setQuery(outgoing_query)
                        results = self.sparql.query().convert()
                        elapsed = time.time() - start
                        total_time += elapsed
                        total_queries += 1

                        for binding in results["results"]["bindings"]:
                            total_triples += 1
                            obj = binding["o"]["value"]
                            if obj not in visited:
                                next_level_uris.add(obj)

                    except Exception as e:
                        elapsed = time.time() - start
                        total_time += elapsed
                        errors.append({"depth": current_depth, "batch": i, "query": "outgoing", "error": str(e)})

                    # Query 2: Incoming relationships
                    incoming_query = f"""
                    SELECT ?s ?p ?entity WHERE {{
                        VALUES ?entity {{ {values_clause} }}
                        ?s ?p ?entity .
                        FILTER(isURI(?s))
                    }}
                    """

                    start = time.time()
                    try:
                        self.sparql.setQuery(incoming_query)
                        results = self.sparql.query().convert()
                        elapsed = time.time() - start
                        total_time += elapsed
                        total_queries += 1

                        for binding in results["results"]["bindings"]:
                            total_triples += 1
                            subj = binding["s"]["value"]
                            if subj not in visited:
                                next_level_uris.add(subj)

                    except Exception as e:
                        elapsed = time.time() - start
                        total_time += elapsed
                        errors.append({"depth": current_depth, "batch": i, "query": "incoming", "error": str(e)})

                current_level_uris = next_level_uris

            batch_result = {
                "batch_size": batch_size,
                "depth": depth,
                "total_time_sec": total_time,
                "total_queries": total_queries,
                "total_triples": total_triples,
                "total_uris_visited": len(visited),
                "avg_time_per_entity_ms": (total_time / len(entities)) * 1000,
                "error_count": len(errors),
                "errors": errors[:5],
                "throughput_entities_per_sec": len(entities) / total_time if total_time > 0 else 0,
            }

            all_batch_results[f"batch_{batch_size}"] = batch_result
            logger.info(f"    Total: {total_time:.2f}s, {total_queries} queries, "
                       f"{total_triples} triples, {len(visited)} URIs visited")

        # Find best batch size: prefer error-free results, then fastest time
        error_free = {k: v for k, v in all_batch_results.items() if v["error_count"] == 0}
        candidates = error_free if error_free else all_batch_results
        best_batch = min(candidates.items(), key=lambda x: x[1]["total_time_sec"])

        result = {
            "approach": "batch",
            "entity_count": len(entities),
            "depth": depth,
            "batch_results": all_batch_results,
            "best_batch_size": best_batch[1]["batch_size"],
            "best_total_time_sec": best_batch[1]["total_time_sec"],
            "best_throughput": best_batch[1]["throughput_entities_per_sec"],
        }

        return result

    @staticmethod
    def _parse_tsv_uri(value: str) -> Optional[str]:
        """Extract URI from TSV value like <http://...>. Returns None for literals."""
        if value.startswith('<') and value.endswith('>'):
            return value[1:-1]
        return None

    def _query_tsv(self, query: str) -> bytes:
        """Execute a SPARQL query and return raw TSV bytes."""
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(TSV)
        return self.sparql.query().convert()

    def _query_json(self, query: str) -> dict:
        """Execute a SPARQL query and return parsed JSON dict."""
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        return self.sparql.query().convert()

    def benchmark_format_comparison(self, entities: List[str], batch_size: int = 1000,
                                     depth: int = 2) -> Dict[str, Any]:
        """
        Benchmark JSON vs TSV response formats for batch SPARQL queries.

        Runs the same BFS traversal twice — once with JSON, once with TSV —
        to measure the impact of response format on total time.
        """
        logger.info(f"=== Benchmarking FORMAT COMPARISON: JSON vs TSV ({len(entities)} entities, "
                     f"batch_size={batch_size}, depth={depth}) ===")

        format_results = {}

        for fmt in ["json", "tsv"]:
            logger.info(f"  Testing format={fmt}...")

            total_time = 0
            total_queries = 0
            total_triples = 0
            total_bytes = 0
            errors = []

            visited = set()
            current_level_uris = set(entities)

            for current_depth in range(depth + 1):
                uris_to_query = [u for u in current_level_uris if u not in visited]
                if not uris_to_query:
                    break

                visited.update(uris_to_query)
                next_level_uris = set()

                logger.info(f"    Depth {current_depth}: {len(uris_to_query)} URIs to query")

                for i in range(0, len(uris_to_query), batch_size):
                    batch = uris_to_query[i:i + batch_size]
                    values_clause = " ".join(f"<{uri}>" for uri in batch)

                    for direction in ["outgoing", "incoming"]:
                        if direction == "outgoing":
                            query = f"""
                            SELECT ?entity ?p ?o WHERE {{
                                VALUES ?entity {{ {values_clause} }}
                                ?entity ?p ?o .
                                FILTER(isURI(?o))
                            }}
                            """
                            uri_col = 2  # ?o is 3rd column (index 2)
                            json_key = "o"
                        else:
                            query = f"""
                            SELECT ?s ?p ?entity WHERE {{
                                VALUES ?entity {{ {values_clause} }}
                                ?s ?p ?entity .
                                FILTER(isURI(?s))
                            }}
                            """
                            uri_col = 0  # ?s is 1st column (index 0)
                            json_key = "s"

                        start = time.time()
                        try:
                            if fmt == "json":
                                results = self._query_json(query)
                                elapsed = time.time() - start
                                for binding in results["results"]["bindings"]:
                                    total_triples += 1
                                    uri = binding[json_key]["value"]
                                    if uri not in visited:
                                        next_level_uris.add(uri)
                            else:
                                raw = self._query_tsv(query)
                                elapsed = time.time() - start
                                total_bytes += len(raw)
                                lines = raw.decode('utf-8').split('\n')
                                for line in lines[1:]:  # skip header
                                    if not line.strip():
                                        continue
                                    total_triples += 1
                                    cols = line.split('\t')
                                    uri = self._parse_tsv_uri(cols[uri_col])
                                    if uri and uri not in visited:
                                        next_level_uris.add(uri)

                            total_time += elapsed
                            total_queries += 1

                        except Exception as e:
                            elapsed = time.time() - start
                            total_time += elapsed
                            errors.append({"depth": current_depth, "batch": i,
                                          "query": direction, "error": str(e)})

                current_level_uris = next_level_uris

            format_results[fmt] = {
                "format": fmt,
                "batch_size": batch_size,
                "depth": depth,
                "total_time_sec": total_time,
                "total_queries": total_queries,
                "total_triples": total_triples,
                "total_uris_visited": len(visited),
                "error_count": len(errors),
                "errors": errors[:5],
                "avg_time_per_entity_ms": (total_time / len(entities)) * 1000,
                "throughput_entities_per_sec": len(entities) / total_time if total_time > 0 else 0,
            }
            if fmt == "tsv":
                format_results[fmt]["total_response_bytes"] = total_bytes

            logger.info(f"    Total: {total_time:.2f}s, {total_queries} queries, "
                        f"{total_triples} triples, {len(visited)} URIs visited")

        # Compute speedup
        json_time = format_results["json"]["total_time_sec"]
        tsv_time = format_results["tsv"]["total_time_sec"]
        if tsv_time > 0:
            format_results["speedup"] = json_time / tsv_time
        else:
            format_results["speedup"] = 0

        logger.info(f"  JSON: {json_time:.2f}s, TSV: {tsv_time:.2f}s, "
                     f"speedup: {format_results['speedup']:.2f}x")

        return format_results

    def run_all_benchmarks(self, entities: List[str], batch_sizes: List[int] = None,
                           approaches: List[str] = None, depth: int = 2) -> Dict[str, Any]:
        """Run all benchmarks and compile results."""

        if approaches is None:
            approaches = ["individual", "bulk", "batch"]

        self.results["metadata"]["depth"] = depth

        if "individual" in approaches:
            self.results["approaches"]["individual"] = self.benchmark_individual_queries(entities, depth=depth)

        if "bulk" in approaches:
            self.results["approaches"]["bulk"] = self.benchmark_bulk_export(entities)

        if "batch" in approaches:
            self.results["approaches"]["batch"] = self.benchmark_batch_queries(entities, batch_sizes, depth=depth)

        if "format" in approaches:
            best_batch_size = 1000
            if "batch" in self.results.get("approaches", {}):
                best_batch_size = self.results["approaches"]["batch"].get("best_batch_size", 1000)
            self.results["approaches"]["format"] = self.benchmark_format_comparison(
                entities, batch_size=best_batch_size, depth=depth)

        return self.results

    def generate_report(self) -> str:
        """Generate a human-readable markdown report."""
        r = self.results
        depth = r['metadata'].get('depth', 2)
        lines = [
            f"# Document Generation Benchmark Report",
            f"",
            f"## Metadata",
            f"- **Dataset**: {r['metadata']['dataset']}",
            f"- **Endpoint**: {r['metadata']['endpoint']}",
            f"- **Timestamp**: {r['metadata']['timestamp']}",
            f"- **Entity Count**: {r['entity_count']}",
            f"- **Traversal Depth**: {depth} (bidirectional: outgoing + incoming)",
            f"",
            f"## Summary",
            f"",
        ]

        # Build comparison table
        lines.append("| Approach | Total Time | Queries | Throughput | Notes |")
        lines.append("|----------|------------|---------|------------|-------|")

        if "individual" in r["approaches"]:
            ind = r["approaches"]["individual"]
            lines.append(f"| Individual Queries | {ind['total_time_sec']:.2f}s | {ind['total_queries']} | {ind['throughput_entities_per_sec']:.1f} ent/s | {ind['error_count']} errors |")

        if "bulk" in r["approaches"]:
            bulk = r["approaches"]["bulk"]
            if "error" not in bulk:
                lines.append(f"| Bulk Export + rdflib | {bulk['total_time_sec']:.2f}s | 1 | {bulk['throughput_entities_per_sec']:.1f} ent/s | {bulk['export_size_mb']:.1f}MB export |")
            else:
                lines.append(f"| Bulk Export + rdflib | FAILED | - | - | {bulk['error']} |")

        if "batch" in r["approaches"]:
            batch = r["approaches"]["batch"]
            best = batch["batch_results"][f"batch_{batch['best_batch_size']}"]
            lines.append(f"| Batch Queries (best) | {batch['best_total_time_sec']:.2f}s | {best['total_queries']} | {batch['best_throughput']:.1f} ent/s | batch_size={batch['best_batch_size']} |")

        lines.append("")

        # Detailed results
        if "individual" in r["approaches"]:
            ind = r["approaches"]["individual"]
            lines.extend([
                f"## Individual Queries (Approach 1)",
                f"",
                f"Simulates `universal_rag_system.py:get_entity_context()` with bidirectional traversal.",
                f"",
                f"- **Total Time**: {ind['total_time_sec']:.2f}s",
                f"- **Total Queries**: {ind['total_queries']} ({ind['avg_queries_per_entity']:.1f} per entity)",
                f"- **Avg per Entity**: {ind['avg_time_per_entity_ms']:.2f}ms",
                f"- **Median**: {ind['median_time_ms']:.2f}ms",
                f"- **Min/Max**: {ind['min_time_ms']:.2f}ms / {ind['max_time_ms']:.2f}ms",
                f"- **Std Dev**: {ind['stddev_ms']:.2f}ms",
                f"- **Total Triples**: {ind['total_triples']} ({ind['avg_triples_per_entity']:.1f} per entity)",
                f"- **Errors**: {ind['error_count']}",
                f"",
            ])

        if "bulk" in r["approaches"]:
            bulk = r["approaches"]["bulk"]
            if "error" not in bulk:
                lines.extend([
                    f"## Bulk Export + rdflib (Approach 2)",
                    f"",
                    f"- **Export Time**: {bulk['export_time_sec']:.2f}s",
                    f"- **Parse Time**: {bulk['parse_time_sec']:.2f}s",
                    f"- **Query Time**: {bulk['query_time_sec']:.2f}s",
                    f"- **Total Time**: {bulk['total_time_sec']:.2f}s",
                    f"- **Export Size**: {bulk['export_size_mb']:.2f}MB",
                    f"- **Total Triples**: {bulk['total_triples']}",
                    f"",
                ])

        if "batch" in r["approaches"]:
            batch = r["approaches"]["batch"]
            lines.extend([
                f"## Batch Queries (Approach 3 - Proposed)",
                f"",
                f"Uses VALUES clause to query multiple entities per request, with bidirectional traversal.",
                f"",
                f"| Batch Size | Time | Queries | URIs Visited | Throughput | Errors |",
                f"|------------|------|---------|--------------|------------|--------|",
            ])
            for name, br in batch["batch_results"].items():
                lines.append(f"| {br['batch_size']} | {br['total_time_sec']:.2f}s | {br['total_queries']} | {br['total_uris_visited']} | {br['throughput_entities_per_sec']:.1f} ent/s | {br['error_count']} |")

            lines.extend([
                f"",
                f"**Best batch size**: {batch['best_batch_size']}",
                f"",
            ])

        if "format" in r["approaches"]:
            fmt = r["approaches"]["format"]
            json_r = fmt["json"]
            tsv_r = fmt["tsv"]
            speedup = fmt.get("speedup", 0)
            lines.extend([
                f"## Response Format Comparison (JSON vs TSV)",
                f"",
                f"Same batch traversal (batch_size={json_r['batch_size']}, depth={json_r['depth']}), different response formats.",
                f"",
                f"| Format | Time | Queries | Triples | URIs Visited | Throughput | Errors |",
                f"|--------|------|---------|---------|--------------|------------|--------|",
                f"| JSON | {json_r['total_time_sec']:.2f}s | {json_r['total_queries']} | {json_r['total_triples']} | {json_r['total_uris_visited']} | {json_r['throughput_entities_per_sec']:.1f} ent/s | {json_r['error_count']} |",
                f"| TSV | {tsv_r['total_time_sec']:.2f}s | {tsv_r['total_queries']} | {tsv_r['total_triples']} | {tsv_r['total_uris_visited']} | {tsv_r['throughput_entities_per_sec']:.1f} ent/s | {tsv_r['error_count']} |",
                f"",
                f"**TSV speedup**: {speedup:.2f}x",
                f"",
            ])

        # Recommendations
        lines.extend([
            f"## Recommendations",
            f"",
        ])

        approaches_data = r["approaches"]
        if len(approaches_data) >= 2:
            # Find fastest
            times = {}
            if "individual" in approaches_data:
                times["Individual"] = approaches_data["individual"]["total_time_sec"]
            if "bulk" in approaches_data and "error" not in approaches_data["bulk"]:
                times["Bulk"] = approaches_data["bulk"]["total_time_sec"]
            if "batch" in approaches_data:
                times["Batch"] = approaches_data["batch"]["best_total_time_sec"]

            if times:
                fastest = min(times, key=times.get)
                slowest = max(times, key=times.get)
                speedup = times[slowest] / times[fastest] if times[fastest] > 0 else 0

                lines.append(f"- **Fastest approach**: {fastest} ({times[fastest]:.2f}s)")
                lines.append(f"- **Speedup vs {slowest}**: {speedup:.1f}x")

        lines.append("")
        lines.append(f"---")
        lines.append(f"*Generated by benchmark_document_generation.py*")

        return "\n".join(lines)

    def save_results(self):
        """Save results to output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON results
        json_path = self.output_dir / "report.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved JSON report to {json_path}")

        # Save markdown report
        md_path = self.output_dir / "report.md"
        with open(md_path, "w") as f:
            f.write(self.generate_report())
        logger.info(f"Saved markdown report to {md_path}")

        return json_path, md_path


def generate_comparison_report(all_results: Dict[str, Dict], dataset: str, entity_count: int, depth: int) -> str:
    """Generate a markdown report comparing results across multiple servers."""
    lines = [
        "# Multi-Server Benchmark Comparison",
        "",
        "## Metadata",
        f"- **Dataset**: {dataset}",
        f"- **Timestamp**: {datetime.now().isoformat()}",
        f"- **Entity Count**: {entity_count}",
        f"- **Traversal Depth**: {depth} (bidirectional: outgoing + incoming)",
        f"- **Servers tested**: {', '.join(all_results.keys())}",
        "",
        "## Comparison",
        "",
        "| Server | Approach | Total Time | Queries | Throughput | Notes |",
        "|--------|----------|------------|---------|------------|-------|",
    ]

    for server_name, results in all_results.items():
        for approach_name, data in results.get("approaches", {}).items():
            if approach_name == "individual":
                lines.append(
                    f"| {server_name} | Individual | {data['total_time_sec']:.2f}s | "
                    f"{data['total_queries']} | {data['throughput_entities_per_sec']:.1f} ent/s | "
                    f"{data['error_count']} errors |"
                )
            elif approach_name == "bulk":
                if "error" not in data:
                    lines.append(
                        f"| {server_name} | Bulk Export | {data['total_time_sec']:.2f}s | "
                        f"1 | {data['throughput_entities_per_sec']:.1f} ent/s | "
                        f"{data['export_size_mb']:.1f}MB export |"
                    )
                else:
                    lines.append(f"| {server_name} | Bulk Export | FAILED | - | - | {data['error']} |")
            elif approach_name == "batch":
                best = data["batch_results"][f"batch_{data['best_batch_size']}"]
                lines.append(
                    f"| {server_name} | Batch (best) | {data['best_total_time_sec']:.2f}s | "
                    f"{best['total_queries']} | {data['best_throughput']:.1f} ent/s | "
                    f"batch_size={data['best_batch_size']} |"
                )

    # Find overall fastest
    lines.extend(["", "## Fastest per Approach", ""])
    approach_times = {}  # approach -> [(server, time)]
    for server_name, results in all_results.items():
        for approach_name, data in results.get("approaches", {}).items():
            if approach_name == "individual":
                t = data["total_time_sec"]
            elif approach_name == "bulk" and "error" not in data:
                t = data["total_time_sec"]
            elif approach_name == "batch":
                t = data["best_total_time_sec"]
            else:
                continue
            if approach_name not in approach_times:
                approach_times[approach_name] = []
            approach_times[approach_name].append((server_name, t))

    for approach, entries in sorted(approach_times.items()):
        if len(entries) > 1:
            entries.sort(key=lambda x: x[1])
            fastest_name, fastest_time = entries[0]
            slowest_name, slowest_time = entries[-1]
            speedup = slowest_time / fastest_time if fastest_time > 0 else 0
            lines.append(f"- **{approach}**: {fastest_name} ({fastest_time:.2f}s) is {speedup:.1f}x faster than {slowest_name} ({slowest_time:.2f}s)")
        else:
            lines.append(f"- **{approach}**: {entries[0][0]} ({entries[0][1]:.2f}s)")

    lines.extend(["", "---", "*Generated by benchmark_document_generation.py*"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark document generation approaches",
        epilog="""
Examples:
  # Single server (from datasets.yaml):
  %(prog)s --dataset mah --approach batch --limit 100

  # Compare two servers:
  %(prog)s --dataset mah --servers fuseki=http://localhost:3030/MAH/sparql qlever=http://localhost:7001

  # Compare servers with specific approach:
  %(prog)s --dataset mah --servers fuseki=http://localhost:3030/MAH/sparql qlever=http://localhost:7001 --approach batch --limit 100
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g., asinou, mah)")
    parser.add_argument("--endpoint", help="SPARQL endpoint URL (overrides datasets.yaml, single server mode)")
    parser.add_argument("--servers", nargs="+", metavar="NAME=URL",
                        help="Named servers to compare (e.g., fuseki=http://localhost:3030/MAH/sparql qlever=http://localhost:7001)")
    parser.add_argument("--limit", type=int, help="Limit number of entities to test")
    parser.add_argument("--depth", type=int, default=2,
                        help="Traversal depth for multi-hop queries (default: 2)")
    parser.add_argument("--approach", choices=["individual", "bulk", "batch", "format", "all"],
                        default="all", help="Which approach to benchmark (format = JSON vs TSV comparison)")
    parser.add_argument("--batch-sizes", default="100,500,1000",
                        help="Comma-separated batch sizes to test (default: 100,500,1000)")
    parser.add_argument("--output-dir", help="Output directory for results")

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Determine approaches to run
    if args.approach == "all":
        approaches = ["individual", "bulk", "batch"]
    else:
        approaches = [args.approach]

    # Build server list: name -> endpoint URL
    servers = {}
    if args.servers:
        for spec in args.servers:
            if "=" not in spec:
                parser.error(f"Invalid server spec '{spec}'. Use NAME=URL format (e.g., fuseki=http://localhost:3030/MAH/sparql)")
            name, url = spec.split("=", 1)
            servers[name] = url
    elif args.endpoint:
        servers["default"] = args.endpoint
    else:
        datasets_config = ConfigLoader.load_datasets_config()
        if args.dataset not in datasets_config.get("datasets", {}):
            logger.error(f"Dataset '{args.dataset}' not found in config/datasets.yaml")
            sys.exit(1)
        endpoint = datasets_config["datasets"][args.dataset].get("endpoint")
        if not endpoint:
            logger.error(f"No endpoint configured for dataset '{args.dataset}'")
            sys.exit(1)
        servers["default"] = endpoint

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results" / f"{args.dataset}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    multi_server = len(servers) > 1

    logger.info(f"Starting benchmark for dataset '{args.dataset}'")
    logger.info(f"Servers: {servers}")
    logger.info(f"Approaches: {approaches}")
    logger.info(f"Traversal depth: {args.depth}")
    logger.info(f"Output: {output_dir}")

    # Use the first server to get the entity list (shared across all servers)
    first_endpoint = next(iter(servers.values()))
    first_name = next(iter(servers.keys()))
    entity_fetcher = DocumentGenerationBenchmark(args.dataset, first_endpoint, output_dir)
    entities = entity_fetcher.get_all_entities(limit=args.limit)

    if not entities:
        logger.error("No entities found!")
        sys.exit(1)

    # Run benchmarks for each server
    all_results = {}
    for server_name, endpoint in servers.items():
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"BENCHMARKING SERVER: {server_name} ({endpoint})")
        logger.info("=" * 60)

        server_dir = output_dir / server_name if multi_server else output_dir
        benchmark = DocumentGenerationBenchmark(args.dataset, endpoint, server_dir)
        benchmark.results["entity_count"] = len(entities)
        benchmark.results["metadata"]["entity_list_time_sec"] = entity_fetcher.results["metadata"].get("entity_list_time_sec", 0)

        benchmark.run_all_benchmarks(entities, batch_sizes=batch_sizes, approaches=approaches, depth=args.depth)
        benchmark.save_results()

        all_results[server_name] = benchmark.results

        # Print individual report
        print(f"\n--- {server_name} ---")
        print(benchmark.generate_report())

    # If multi-server, generate and save comparison report
    if multi_server:
        comparison = generate_comparison_report(all_results, args.dataset, len(entities), args.depth)

        comparison_path = output_dir / "comparison.md"
        with open(comparison_path, "w") as f:
            f.write(comparison)

        comparison_json_path = output_dir / "comparison.json"
        with open(comparison_json_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print("\n" + "=" * 60)
        print(comparison)
        print("=" * 60)
        print(f"\nComparison saved to: {comparison_path}")
    else:
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
