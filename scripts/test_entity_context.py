#!/usr/bin/env python3
"""
Test script to debug event-aware entity context traversal.
This script tests a single entity to trace the filtering decisions.

Usage:
    python scripts/test_entity_context.py --entity <entity_uri> --endpoint <sparql_endpoint>

Example:
    python scripts/test_entity_context.py --entity "http://example.org/inscription/05b72a27" --endpoint "http://localhost:3030/asinou/sparql"
"""

import os
import argparse
import logging

from crm_rag.config_loader import ConfigLoader

# Set up debug logging BEFORE importing crm_rag.rag_system
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from crm_rag.rag_system import UniversalRagSystem


def main():
    parser = argparse.ArgumentParser(description='Test event-aware entity context traversal')
    parser.add_argument('--entity', type=str, required=True, help='Entity URI to test')
    parser.add_argument('--endpoint', type=str, required=True, help='SPARQL endpoint URL')
    parser.add_argument('--depth', type=int, default=2, help='Traversal depth (default: 2)')
    parser.add_argument('--env', type=str, default=None, help='Optional .env config file')
    args = parser.parse_args()

    # Load minimal config
    if args.env:
        config = ConfigLoader.load_config(args.env)
    else:
        config = {
            'llm_provider': 'openai',  # We won't use LLM, just traversal
            'embedding_provider': 'openai',
        }

    # Print loaded event classes
    event_classes = UniversalRagSystem.get_event_classes()
    print(f"\n=== Loaded {len(event_classes)} event classes ===")
    for ec in sorted(event_classes):
        print(f"  {ec.split('/')[-1].split('#')[-1]}")

    print(f"\n=== Testing entity: {args.entity} ===")
    print(f"Endpoint: {args.endpoint}")
    print(f"Depth: {args.depth}")

    # Create RAG system (we won't use it for retrieval, just context generation)
    try:
        rag = UniversalRagSystem(args.endpoint, config, dataset_id="test")
    except Exception as e:
        logger.error(f"Failed to create RAG system: {e}")
        return

    # Test entity types
    print(f"\n=== Entity types ===")
    types = rag.get_entity_types_cached(args.entity, {})
    for t in types:
        print(f"  {t}")

    # Test context generation with debug output
    print(f"\n=== Context traversal (depth={args.depth}) ===")
    print("(Debug output shows filtering decisions)")

    try:
        statements, triples = rag.get_entity_context(
            args.entity,
            depth=args.depth,
            return_triples=True
        )

        print(f"\n=== Generated {len(statements)} statements ===")
        for stmt in statements[:20]:  # Show first 20
            print(f"  • {stmt}")
        if len(statements) > 20:
            print(f"  ... and {len(statements) - 20} more")

        print(f"\n=== Raw triples ({len(triples)}) ===")
        for triple in triples[:10]:  # Show first 10
            print(f"  {triple['subject_label']} --{triple['predicate_label']}--> {triple['object_label']}")
        if len(triples) > 10:
            print(f"  ... and {len(triples) - 10} more")

    except Exception as e:
        logger.error(f"Error during context generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
