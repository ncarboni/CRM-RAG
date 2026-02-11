#!/usr/bin/env python3
"""
Pipeline evaluation script for RAG system.
Runs a sequence of questions as a conversation and logs retrieval details.

Output is written to reports/<dataset>_<YYYYMMDD_HHMMSS>.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from crm_rag import PROJECT_ROOT
from crm_rag.config_loader import ConfigLoader
from crm_rag.dataset_manager import DatasetManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

QUESTIONS = [
    "Which pieces from Swiss Artists are in the Musée d'art et d'histoire in Geneva ?",
    "Which paintings depict geneva?",
    "Tell me more about Ferdinand Hodler and its paintings",
    "Tell me more about Guerrier au morgenstern",
    "Guerrier au morgenstern and Fondation Pierre Gianadda, it has been exhibited in which exhibition?",
    "When it happened and with which other pieces?",
    "In which exhibitions and where the work of hodler has been featured?",
    "When they took place?",
    "Aside from Hodler, are there any relevant Swiss Artist?",
    "What Hans Schweizer did?",
    "Which are the top 10 Swiss Artist in the Musée d'art et d'histoire?",
]


def run_evaluation(env_file=None, dataset_id="asinou", output_file=None):
    config = ConfigLoader.load_config(env_file)

    # Load datasets config
    datasets_config = ConfigLoader.load_datasets_config()

    dm = DatasetManager(datasets_config, config)
    rag = dm.get_dataset(dataset_id)

    results = []
    chat_history = []

    for i, question in enumerate(QUESTIONS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Q{i+1}: {question}")
        logger.info(f"{'='*60}")

        start = time.time()
        result = rag.answer_question(question, chat_history=chat_history if chat_history else None)
        elapsed = time.time() - start

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # Extract retrieval details
        source_summary = []
        for s in sources:
            entry = {
                "entity_label": s.get("entity_label", ""),
                "entity_uri": s.get("entity_uri", ""),
                "entity_type": s.get("entity_type", ""),
            }
            # Count raw triples
            raw_triples = s.get("raw_triples", [])
            entry["raw_triples_count"] = len(raw_triples)
            source_summary.append(entry)

        record = {
            "question_number": i + 1,
            "question": question,
            "answer": answer,
            "elapsed_seconds": round(elapsed, 2),
            "num_sources": len(sources),
            "sources": source_summary,
        }
        results.append(record)

        # Print summary
        print(f"\nQ{i+1}: {question}")
        print(f"Answer ({elapsed:.1f}s):")
        print(answer[:1500])
        print(f"\nSources ({len(sources)}):")
        for s in source_summary:
            print(f"  - {s['entity_label']} [{s['entity_type']}] ({s['raw_triples_count']} triples) {s['entity_uri']}")

        # Update chat history
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

    # Determine output path
    if output_file:
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    else:
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = reports_dir / f"{dataset_id}_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=None, help="Env file path")
    parser.add_argument("--dataset", default="asinou", help="Dataset ID")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: reports/<dataset>_<timestamp>.json)")
    args = parser.parse_args()
    run_evaluation(args.env, args.dataset, args.output)
