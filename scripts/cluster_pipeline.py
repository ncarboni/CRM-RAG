#!/usr/bin/env python3
"""
Cluster Pipeline for CRM_RAG

Orchestrates the full processing pipeline for running on clusters:
1. RDF export from SPARQL endpoint
2. Document generation (with multiprocessing)
3. Embedding computation (GPU-accelerated)

This script simplifies the workflow for processing large datasets across
local machines and GPU clusters.

Usage:
    # Full pipeline
    python scripts/cluster_pipeline.py --dataset mah --all

    # Individual steps
    python scripts/cluster_pipeline.py --dataset mah --export           # SPARQL -> TTL
    python scripts/cluster_pipeline.py --dataset mah --generate-docs    # TTL -> Documents
    python scripts/cluster_pipeline.py --dataset mah --embed            # Documents -> Embeddings

    # Combined steps
    python scripts/cluster_pipeline.py --dataset mah --export --generate-docs    # Steps 1+2
    python scripts/cluster_pipeline.py --dataset mah --generate-docs --embed     # Steps 2+3

    # Options
    --workers N           # Multiprocessing (default: 1)
    --context-depth 0|1|2 # Relationship depth (default: 2)
    --env FILE            # Config file (.env.local, .env.cluster)
    --from-file PATH      # Use existing TTL export
    --batch-size N        # Embedding batch size (default: 64)

    # Utilities
    --status              # Show pipeline status for dataset
    --clean               # Clean intermediate files
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import ConfigLoader
from scripts.bulk_generate_documents import BulkDocumentGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClusterPipeline:
    """
    Orchestrates the full processing pipeline for cluster deployment.

    The pipeline has three steps:
    1. Export: SPARQL -> TTL file
    2. Generate: TTL -> Entity documents (markdown)
    3. Embed: Documents -> Embeddings + Graph

    Each step can be run independently or combined.
    """

    def __init__(
        self,
        dataset_id: str,
        env_file: str = None,
        workers: int = 1,
        context_depth: int = 2,
        batch_size: int = 64
    ):
        """
        Initialize the cluster pipeline.

        Args:
            dataset_id: Dataset identifier (from datasets.yaml)
            env_file: Path to environment config file
            workers: Number of parallel workers for document generation
            context_depth: Relationship traversal depth (0, 1, or 2)
            batch_size: Batch size for embedding computation
        """
        self.dataset_id = dataset_id
        self.env_file = env_file
        self.workers = workers
        self.context_depth = context_depth
        self.batch_size = batch_size

        # Base directory
        self.base_dir = Path(__file__).parent.parent

        # Load configuration
        self.config = ConfigLoader.load_config(env_file)

        # Load datasets configuration
        self.datasets_config = ConfigLoader.load_datasets_config()
        self.dataset_config = self.datasets_config.get("datasets", {}).get(dataset_id)

        if not self.dataset_config:
            available = list(self.datasets_config.get("datasets", {}).keys())
            raise ValueError(
                f"Dataset '{dataset_id}' not found in datasets.yaml. "
                f"Available: {available}"
            )

        # Initialize bulk document generator
        self.generator = BulkDocumentGenerator(dataset_id, str(self.base_dir))

        # Paths
        self.export_dir = self.base_dir / "data" / "exports"
        self.export_file = self.export_dir / f"{dataset_id}_dump.ttl"
        self.docs_dir = self.base_dir / "data" / "documents" / dataset_id / "entity_documents"
        self.metadata_file = self.base_dir / "data" / "documents" / dataset_id / "documents_metadata.json"
        self.cache_dir = self.base_dir / "data" / "cache" / dataset_id
        self.graph_file = self.cache_dir / "document_graph.pkl"
        self.vector_dir = self.cache_dir / "vector_index"

    def run(self, steps: List[str], from_file: str = None) -> bool:
        """
        Execute specified pipeline steps in order.

        Args:
            steps: List of steps to execute ('export', 'generate', 'embed')
            from_file: Optional path to existing TTL file (skips export)

        Returns:
            True if all steps completed successfully
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"CLUSTER PIPELINE: {self.dataset_id}")
        logger.info("=" * 60)
        logger.info(f"Steps: {', '.join(steps)}")
        logger.info(f"Workers: {self.workers}")
        logger.info(f"Context depth: {self.context_depth}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info("=" * 60)

        export_path = from_file or str(self.export_file)

        try:
            # Step 1: Export
            if 'export' in steps:
                export_path = self.export()
                if not export_path:
                    logger.error("Export step failed")
                    return False

            # Step 2: Generate documents
            if 'generate' in steps:
                # Use from_file if provided, otherwise use export path
                input_file = from_file or export_path
                doc_count = self.generate_docs(input_file)
                if doc_count == 0:
                    logger.error("Document generation step failed")
                    return False

            # Step 3: Embed
            if 'embed' in steps:
                success = self.embed()
                if not success:
                    logger.error("Embedding step failed")
                    return False

            elapsed = time.time() - start_time
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Total time: {elapsed/60:.1f} minutes")
            self._print_next_steps(steps)
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def export(self) -> Optional[str]:
        """
        Step 1: Export RDF data from SPARQL endpoint.

        Returns:
            Path to exported TTL file, or None on failure
        """
        logger.info("-" * 60)
        logger.info("STEP 1: EXPORT RDF DATA")
        logger.info("-" * 60)

        endpoint = self.dataset_config.get("endpoint")
        if not endpoint:
            logger.error(f"No endpoint configured for dataset '{self.dataset_id}'")
            return None

        try:
            export_path = self.generator.export_from_sparql(endpoint)
            logger.info(f"Export complete: {export_path}")
            return export_path
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None

    def generate_docs(self, from_file: str = None) -> int:
        """
        Step 2: Generate entity documents from RDF data.

        Args:
            from_file: Path to TTL/RDF file (uses default export if None)

        Returns:
            Number of documents generated, or 0 on failure
        """
        logger.info("-" * 60)
        logger.info("STEP 2: GENERATE DOCUMENTS")
        logger.info("-" * 60)

        # Determine input file
        input_file = from_file
        if not input_file:
            if self.export_file.exists():
                input_file = str(self.export_file)
                logger.info(f"Using existing export: {input_file}")
            else:
                logger.error(f"No input file specified and export not found at {self.export_file}")
                logger.error("Run with --export first or specify --from-file")
                return 0

        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return 0

        try:
            # Load and process the graph
            self.generator.load_graph(input_file)
            self.generator.build_indexes()

            # Generate documents
            doc_count = self.generator.generate_all_documents(
                context_depth=self.context_depth,
                workers=self.workers
            )

            logger.info(f"Generated {doc_count} documents")
            return doc_count

        except Exception as e:
            logger.error(f"Document generation failed: {e}")
            import traceback
            traceback.print_exc()
            return 0

    def embed(self) -> bool:
        """
        Step 3: Compute embeddings and build document graph.

        Returns:
            True on success, False on failure
        """
        logger.info("-" * 60)
        logger.info("STEP 3: COMPUTE EMBEDDINGS")
        logger.info("-" * 60)

        # Check for documents
        if not self.metadata_file.exists():
            logger.error(f"Metadata file not found: {self.metadata_file}")
            logger.error("Run --generate-docs first")
            return False

        if not self.docs_dir.exists():
            logger.error(f"Documents directory not found: {self.docs_dir}")
            return False

        # Import here to avoid loading heavy dependencies if not needed
        from universal_rag_system import UniversalRagSystem

        # Configure for embedding mode
        embed_config = self.config.copy()
        embed_config['embed_from_docs'] = True
        embed_config['embedding_batch_size'] = self.batch_size

        # Create RAG system instance
        # Use a dummy endpoint since we don't need SPARQL for embedding
        endpoint = self.dataset_config.get("endpoint", "http://localhost:3030/dummy/sparql")

        try:
            rag_system = UniversalRagSystem(
                endpoint_url=endpoint,
                config=embed_config,
                dataset_id=self.dataset_id
            )

            # Run embedding (embed_from_docs mode)
            success = rag_system.initialize()

            if success:
                logger.info("Embedding complete")
                logger.info(f"Document graph: {self.graph_file}")
                logger.info(f"Vector index: {self.vector_dir}")

            return success

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def status(self) -> Dict:
        """
        Get current pipeline status for the dataset.

        Returns:
            Dictionary with status information
        """
        status = {
            "dataset_id": self.dataset_id,
            "endpoint": self.dataset_config.get("endpoint"),
            "steps": {}
        }

        # Check export
        if self.export_file.exists():
            size_mb = self.export_file.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(self.export_file.stat().st_mtime)
            status["steps"]["export"] = {
                "complete": True,
                "file": str(self.export_file),
                "size_mb": round(size_mb, 1),
                "modified": mtime.isoformat()
            }
        else:
            status["steps"]["export"] = {"complete": False}

        # Check documents
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                meta = json.load(f)
            doc_count = meta.get("total_documents", 0)
            generated_at = meta.get("generated_at", "unknown")
            status["steps"]["generate"] = {
                "complete": True,
                "document_count": doc_count,
                "directory": str(self.docs_dir),
                "generated_at": generated_at
            }
        else:
            status["steps"]["generate"] = {"complete": False}

        # Check embeddings
        if self.graph_file.exists() and self.vector_dir.exists():
            graph_size_mb = self.graph_file.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(self.graph_file.stat().st_mtime)
            status["steps"]["embed"] = {
                "complete": True,
                "graph_file": str(self.graph_file),
                "graph_size_mb": round(graph_size_mb, 1),
                "vector_dir": str(self.vector_dir),
                "modified": mtime.isoformat()
            }
        else:
            status["steps"]["embed"] = {"complete": False}

        return status

    def clean(self, clean_export: bool = False, clean_docs: bool = False, clean_cache: bool = False):
        """
        Clean intermediate files.

        Args:
            clean_export: Remove export files
            clean_docs: Remove generated documents
            clean_cache: Remove embeddings and graph
        """
        logger.info(f"Cleaning files for dataset: {self.dataset_id}")

        if clean_export and self.export_file.exists():
            self.export_file.unlink()
            logger.info(f"Removed: {self.export_file}")

        if clean_docs:
            if self.docs_dir.exists():
                shutil.rmtree(self.docs_dir)
                logger.info(f"Removed: {self.docs_dir}")
            if self.metadata_file.exists():
                self.metadata_file.unlink()
                logger.info(f"Removed: {self.metadata_file}")

        if clean_cache:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                logger.info(f"Removed: {self.cache_dir}")

        logger.info("Clean complete")

    def _print_next_steps(self, completed_steps: List[str]):
        """Print next steps after pipeline completion."""
        logger.info("")
        logger.info("Next steps:")

        if 'export' in completed_steps and 'generate' not in completed_steps:
            logger.info(f"  Generate documents:")
            logger.info(f"    python scripts/cluster_pipeline.py --dataset {self.dataset_id} --generate-docs")

        if 'generate' in completed_steps and 'embed' not in completed_steps:
            logger.info(f"  Option A - Embed locally:")
            logger.info(f"    python scripts/cluster_pipeline.py --dataset {self.dataset_id} --embed --env .env.local")
            logger.info("")
            logger.info(f"  Option B - Transfer to cluster:")
            logger.info(f"    rsync -avz data/documents/{self.dataset_id}/ user@cluster:CRM_RAG/data/documents/{self.dataset_id}/")
            logger.info(f"    # On cluster:")
            logger.info(f"    python scripts/cluster_pipeline.py --dataset {self.dataset_id} --embed --env .env.cluster")

        if 'embed' in completed_steps:
            # Check if we're likely on a cluster (no export step means cluster)
            if 'export' not in completed_steps and 'generate' not in completed_steps:
                logger.info(f"  Transfer cache to local machine:")
                logger.info(f"    rsync -avz data/cache/{self.dataset_id}/ user@local:CRM_RAG/data/cache/{self.dataset_id}/")
                logger.info("")

            logger.info(f"  Start web server:")
            logger.info(f"    python main.py --env .env.local")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster pipeline for CRM_RAG processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (all steps, single machine)
  python scripts/cluster_pipeline.py --dataset mah --all --workers 8

  # Split workflow - LOCAL: export only (fast, single SPARQL query)
  python scripts/cluster_pipeline.py --dataset mah --export

  # Split workflow - CLUSTER: generate docs + embed (no SPARQL needed)
  python scripts/cluster_pipeline.py --dataset mah --generate-docs --embed --workers 16 --env .env.cluster

  # Check status
  python scripts/cluster_pipeline.py --dataset mah --status
        """
    )

    # Required
    parser.add_argument("--dataset", required=True, help="Dataset ID (from datasets.yaml)")

    # Pipeline steps
    parser.add_argument("--all", action="store_true", help="Run full pipeline (export + generate + embed)")
    parser.add_argument("--export", action="store_true", help="Step 1: Export RDF from SPARQL")
    parser.add_argument("--generate-docs", action="store_true", help="Step 2: Generate entity documents")
    parser.add_argument("--embed", action="store_true", help="Step 3: Compute embeddings and build graph")

    # Options
    parser.add_argument("--env", help="Path to environment config file (e.g., .env.local)")
    parser.add_argument("--from-file", help="Use existing TTL/RDF file instead of exporting")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--context-depth", type=int, default=2, choices=[0, 1, 2],
                        help="Relationship traversal depth (default: 2)")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size (default: 64)")

    # Utilities
    parser.add_argument("--status", action="store_true", help="Show pipeline status for dataset")
    parser.add_argument("--clean", action="store_true", help="Clean intermediate files")
    parser.add_argument("--clean-export", action="store_true", help="Clean export files")
    parser.add_argument("--clean-docs", action="store_true", help="Clean generated documents")
    parser.add_argument("--clean-cache", action="store_true", help="Clean embeddings and graph")

    args = parser.parse_args()

    try:
        pipeline = ClusterPipeline(
            dataset_id=args.dataset,
            env_file=args.env,
            workers=args.workers,
            context_depth=args.context_depth,
            batch_size=args.batch_size
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Handle status
    if args.status:
        status = pipeline.status()
        print(json.dumps(status, indent=2))
        return

    # Handle clean
    if args.clean or args.clean_export or args.clean_docs or args.clean_cache:
        pipeline.clean(
            clean_export=args.clean or args.clean_export,
            clean_docs=args.clean or args.clean_docs,
            clean_cache=args.clean or args.clean_cache
        )
        return

    # Determine steps to run
    steps = []
    if args.all:
        steps = ['export', 'generate', 'embed']
    else:
        if args.export:
            steps.append('export')
        if args.generate_docs:
            steps.append('generate')
        if args.embed:
            steps.append('embed')

    if not steps:
        parser.error("No pipeline steps specified. Use --all, --export, --generate-docs, or --embed")

    # Run pipeline
    success = pipeline.run(steps, from_file=args.from_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
