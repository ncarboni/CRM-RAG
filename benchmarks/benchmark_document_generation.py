#!/usr/bin/env python3
"""
Benchmark Script for Document Generation Approaches

Compares three approaches for generating entity documents:
1. Individual SPARQL queries (current: universal_rag_system.py)
2. Bulk TTL export + rdflib parsing (current: bulk_generate_documents.py)
3. Batch SPARQL queries with VALUES clause (proposed)

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
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from SPARQLWrapper import SPARQLWrapper, JSON, N3
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

    def get_all_entities(self, limit: Optional[int] = None) -> List[str]:
        """Get list of all entity URIs from the dataset."""
        query = """
        SELECT DISTINCT ?entity WHERE {
            ?entity ?p ?o .
            FILTER(isURI(?entity))
            FILTER(!isBlank(?entity))
        }
        """
        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Fetching entity list (limit={limit})...")
        start = time.time()

        self.sparql.setQuery(query)
        results = self.sparql.query().convert()

        entities = [r["entity"]["value"] for r in results["results"]["bindings"]]
        elapsed = time.time() - start

        logger.info(f"Found {len(entities)} entities in {elapsed:.2f}s")
        self.results["entity_count"] = len(entities)
        self.results["metadata"]["entity_list_time_sec"] = elapsed

        return entities

    def benchmark_individual_queries(self, entities: List[str]) -> Dict[str, Any]:
        """
        Benchmark Approach 1: Individual SPARQL query per entity.
        This simulates what universal_rag_system.py:get_entity_context() does.
        """
        logger.info(f"=== Benchmarking INDIVIDUAL QUERIES ({len(entities)} entities) ===")

        timings = []
        triple_counts = []
        errors = []

        for i, entity_uri in enumerate(entities):
            query = f"""
            SELECT ?p ?o WHERE {{
                <{entity_uri}> ?p ?o .
            }}
            """

            start = time.time()
            try:
                self.sparql.setQuery(query)
                results = self.sparql.query().convert()
                elapsed = time.time() - start

                triple_count = len(results["results"]["bindings"])
                timings.append(elapsed)
                triple_counts.append(triple_count)

            except Exception as e:
                elapsed = time.time() - start
                timings.append(elapsed)
                triple_counts.append(0)
                errors.append({"entity": entity_uri, "error": str(e)})

            if (i + 1) % 50 == 0:
                avg_so_far = statistics.mean(timings)
                logger.info(f"  Progress: {i+1}/{len(entities)} entities, avg={avg_so_far*1000:.1f}ms/entity")

        total_time = sum(timings)
        result = {
            "approach": "individual",
            "entity_count": len(entities),
            "total_time_sec": total_time,
            "avg_time_per_entity_ms": (total_time / len(entities)) * 1000,
            "median_time_ms": statistics.median(timings) * 1000,
            "min_time_ms": min(timings) * 1000,
            "max_time_ms": max(timings) * 1000,
            "stddev_ms": statistics.stdev(timings) * 1000 if len(timings) > 1 else 0,
            "total_triples": sum(triple_counts),
            "avg_triples_per_entity": statistics.mean(triple_counts),
            "error_count": len(errors),
            "errors": errors[:10],  # First 10 errors only
            "throughput_entities_per_sec": len(entities) / total_time if total_time > 0 else 0,
        }

        logger.info(f"  Total: {total_time:.2f}s, Avg: {result['avg_time_per_entity_ms']:.1f}ms/entity")
        return result

    def benchmark_bulk_export(self, entities: List[str]) -> Dict[str, Any]:
        """
        Benchmark Approach 2: Bulk CONSTRUCT export.
        This simulates what bulk_generate_documents.py:export_from_sparql() does.
        """
        logger.info(f"=== Benchmarking BULK EXPORT ===")

        # Step 1: Export all triples
        export_query = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"

        start_export = time.time()
        try:
            self.sparql.setQuery(export_query)
            self.sparql.setReturnFormat(N3)
            raw_results = self.sparql.query().convert()
            export_time = time.time() - start_export

            # Get size of exported data
            if isinstance(raw_results, bytes):
                export_size = len(raw_results)
            else:
                export_size = len(str(raw_results))

            logger.info(f"  Export: {export_time:.2f}s, size={export_size/1024/1024:.1f}MB")

        except Exception as e:
            export_time = time.time() - start_export
            logger.error(f"  Export failed: {e}")
            return {
                "approach": "bulk",
                "error": str(e),
                "export_time_sec": export_time,
            }

        # Step 2: Parse with rdflib (simulate)
        start_parse = time.time()
        try:
            from rdflib import Graph
            g = Graph()

            if isinstance(raw_results, bytes):
                g.parse(data=raw_results, format="n3")
            else:
                g.parse(data=raw_results.serialize(format="n3"), format="n3")

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

    def benchmark_batch_queries(self, entities: List[str], batch_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark Approach 3: Batch SPARQL queries using VALUES clause.
        This is the proposed new approach.
        """
        if batch_sizes is None:
            batch_sizes = [100, 500, 1000]

        logger.info(f"=== Benchmarking BATCH QUERIES ({len(entities)} entities) ===")

        all_batch_results = {}

        for batch_size in batch_sizes:
            logger.info(f"  Testing batch_size={batch_size}...")

            timings = []
            triple_counts = []
            errors = []

            # Process in batches
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]

                # Build VALUES clause
                values_clause = " ".join(f"<{uri}>" for uri in batch)
                query = f"""
                SELECT ?entity ?p ?o WHERE {{
                    VALUES ?entity {{ {values_clause} }}
                    ?entity ?p ?o .
                }}
                """

                start = time.time()
                try:
                    self.sparql.setQuery(query)
                    self.sparql.setReturnFormat(JSON)
                    results = self.sparql.query().convert()
                    elapsed = time.time() - start

                    triple_count = len(results["results"]["bindings"])
                    timings.append(elapsed)
                    triple_counts.append(triple_count)

                except Exception as e:
                    elapsed = time.time() - start
                    timings.append(elapsed)
                    triple_counts.append(0)
                    errors.append({"batch_start": i, "error": str(e)})
                    logger.warning(f"    Batch {i//batch_size + 1} failed: {e}")

            total_time = sum(timings)
            num_batches = len(timings)

            batch_result = {
                "batch_size": batch_size,
                "num_batches": num_batches,
                "total_time_sec": total_time,
                "avg_time_per_batch_ms": (total_time / num_batches) * 1000 if num_batches > 0 else 0,
                "avg_time_per_entity_ms": (total_time / len(entities)) * 1000,
                "median_batch_time_ms": statistics.median(timings) * 1000 if timings else 0,
                "min_batch_time_ms": min(timings) * 1000 if timings else 0,
                "max_batch_time_ms": max(timings) * 1000 if timings else 0,
                "total_triples": sum(triple_counts),
                "error_count": len(errors),
                "errors": errors[:5],
                "throughput_entities_per_sec": len(entities) / total_time if total_time > 0 else 0,
            }

            all_batch_results[f"batch_{batch_size}"] = batch_result
            logger.info(f"    Total: {total_time:.2f}s, {batch_result['throughput_entities_per_sec']:.1f} entities/sec")

        # Find best batch size
        best_batch = min(all_batch_results.items(), key=lambda x: x[1]["total_time_sec"])

        result = {
            "approach": "batch",
            "entity_count": len(entities),
            "batch_results": all_batch_results,
            "best_batch_size": best_batch[1]["batch_size"],
            "best_total_time_sec": best_batch[1]["total_time_sec"],
            "best_throughput": best_batch[1]["throughput_entities_per_sec"],
        }

        return result

    def run_all_benchmarks(self, entities: List[str], batch_sizes: List[int] = None,
                           approaches: List[str] = None) -> Dict[str, Any]:
        """Run all benchmarks and compile results."""

        if approaches is None:
            approaches = ["individual", "bulk", "batch"]

        if "individual" in approaches:
            self.results["approaches"]["individual"] = self.benchmark_individual_queries(entities)

        if "bulk" in approaches:
            self.results["approaches"]["bulk"] = self.benchmark_bulk_export(entities)

        if "batch" in approaches:
            self.results["approaches"]["batch"] = self.benchmark_batch_queries(entities, batch_sizes)

        return self.results

    def generate_report(self) -> str:
        """Generate a human-readable markdown report."""
        r = self.results
        lines = [
            f"# Document Generation Benchmark Report",
            f"",
            f"## Metadata",
            f"- **Dataset**: {r['metadata']['dataset']}",
            f"- **Endpoint**: {r['metadata']['endpoint']}",
            f"- **Timestamp**: {r['metadata']['timestamp']}",
            f"- **Entity Count**: {r['entity_count']}",
            f"",
            f"## Summary",
            f"",
        ]

        # Build comparison table
        lines.append("| Approach | Total Time | Throughput | Notes |")
        lines.append("|----------|------------|------------|-------|")

        if "individual" in r["approaches"]:
            ind = r["approaches"]["individual"]
            lines.append(f"| Individual Queries | {ind['total_time_sec']:.2f}s | {ind['throughput_entities_per_sec']:.1f} ent/s | {ind['error_count']} errors |")

        if "bulk" in r["approaches"]:
            bulk = r["approaches"]["bulk"]
            if "error" not in bulk:
                lines.append(f"| Bulk Export + rdflib | {bulk['total_time_sec']:.2f}s | {bulk['throughput_entities_per_sec']:.1f} ent/s | {bulk['export_size_mb']:.1f}MB export |")
            else:
                lines.append(f"| Bulk Export + rdflib | FAILED | - | {bulk['error']} |")

        if "batch" in r["approaches"]:
            batch = r["approaches"]["batch"]
            lines.append(f"| Batch Queries (best) | {batch['best_total_time_sec']:.2f}s | {batch['best_throughput']:.1f} ent/s | batch_size={batch['best_batch_size']} |")

        lines.append("")

        # Detailed results
        if "individual" in r["approaches"]:
            ind = r["approaches"]["individual"]
            lines.extend([
                f"## Individual Queries (Approach 1)",
                f"",
                f"- **Total Time**: {ind['total_time_sec']:.2f}s",
                f"- **Avg per Entity**: {ind['avg_time_per_entity_ms']:.2f}ms",
                f"- **Median**: {ind['median_time_ms']:.2f}ms",
                f"- **Min/Max**: {ind['min_time_ms']:.2f}ms / {ind['max_time_ms']:.2f}ms",
                f"- **Std Dev**: {ind['stddev_ms']:.2f}ms",
                f"- **Total Triples**: {ind['total_triples']}",
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
                f"| Batch Size | Time | Throughput | Errors |",
                f"|------------|------|------------|--------|",
            ])
            for name, br in batch["batch_results"].items():
                lines.append(f"| {br['batch_size']} | {br['total_time_sec']:.2f}s | {br['throughput_entities_per_sec']:.1f} ent/s | {br['error_count']} |")

            lines.extend([
                f"",
                f"**Best batch size**: {batch['best_batch_size']}",
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


def main():
    parser = argparse.ArgumentParser(description="Benchmark document generation approaches")
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g., asinou, mah)")
    parser.add_argument("--endpoint", help="SPARQL endpoint URL (overrides datasets.yaml)")
    parser.add_argument("--limit", type=int, help="Limit number of entities to test")
    parser.add_argument("--approach", choices=["individual", "bulk", "batch", "all"],
                        default="all", help="Which approach to benchmark")
    parser.add_argument("--batch-sizes", default="100,500,1000",
                        help="Comma-separated batch sizes to test (default: 100,500,1000)")
    parser.add_argument("--output-dir", help="Output directory for results")

    args = parser.parse_args()

    # Load endpoint from config if not provided
    if args.endpoint:
        endpoint = args.endpoint
    else:
        datasets_config = ConfigLoader.load_datasets_config()
        if args.dataset not in datasets_config.get("datasets", {}):
            logger.error(f"Dataset '{args.dataset}' not found in config/datasets.yaml")
            sys.exit(1)
        endpoint = datasets_config["datasets"][args.dataset].get("endpoint")
        if not endpoint:
            logger.error(f"No endpoint configured for dataset '{args.dataset}'")
            sys.exit(1)

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "results" / f"{args.dataset}_{timestamp}"

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    # Determine approaches to run
    if args.approach == "all":
        approaches = ["individual", "bulk", "batch"]
    else:
        approaches = [args.approach]

    # Run benchmark
    logger.info(f"Starting benchmark for dataset '{args.dataset}'")
    logger.info(f"Endpoint: {endpoint}")
    logger.info(f"Approaches: {approaches}")
    logger.info(f"Output: {output_dir}")

    benchmark = DocumentGenerationBenchmark(args.dataset, endpoint, output_dir)

    # Get entities
    entities = benchmark.get_all_entities(limit=args.limit)

    if not entities:
        logger.error("No entities found!")
        sys.exit(1)

    # Run benchmarks
    benchmark.run_all_benchmarks(entities, batch_sizes=batch_sizes, approaches=approaches)

    # Save results
    json_path, md_path = benchmark.save_results()

    # Print summary
    print("\n" + "=" * 60)
    print(benchmark.generate_report())
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
