"""
Document formatting utilities for CIDOC-CRM predicates and classes.

Contains pure functions for predicate filtering, class name checks,
and relationship weight assignment. Extracted from UniversalRagSystem.
"""

import json
import logging
import re

from crm_rag import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Module-level cache for relationship weights (loaded once on first call)
_relationship_weights = None
_default_weight = 0.5


def is_schema_predicate(predicate):
    """Check if a predicate is a schema-level predicate that should be filtered out"""
    schema_patterns = [
        'rdf-syntax-ns#type',
        'rdf-schema#subClassOf',
        'rdf-schema#domain',
        'rdf-schema#range',
        'rdf-schema#Class',
        'rdf-schema#subPropertyOf',
        'rdf-schema#label',
        'rdf-schema#comment',
        'owl#',
        '/type',
        '/subClassOf',
        '/domain',
        '/range'
    ]

    for pattern in schema_patterns:
        if pattern in predicate:
            return True

    return False


def is_technical_class_name(class_name, ontology_classes=None):
    """
    Check if a class name is a technical ontology class that should be filtered
    from natural language output.

    Uses the ontology classes extracted from CIDOC-CRM, VIR, CRMdig, etc.

    Args:
        class_name: The class name to check (can be full URI or local name)
        ontology_classes: Set of known ontology class names. If None, falls back
                         to regex pattern matching.

    Returns:
        bool: True if it's a technical ontology class, False if it's human-readable
    """
    if ontology_classes is None:
        logger.warning("Ontology classes not loaded, falling back to regex pattern matching")
        technical_pattern = r'^[A-Z]+\d+[a-z]?_'
        return bool(re.match(technical_pattern, class_name))

    # Check direct match
    if class_name in ontology_classes:
        return True

    # Check local name (after last / or #)
    if '/' in class_name or '#' in class_name:
        local_name = class_name.split('/')[-1].split('#')[-1]
        if local_name in ontology_classes:
            return True

    return False


def _load_relationship_weights():
    """Load relationship weights from config/relationship_weights.json (once)."""
    global _relationship_weights, _default_weight
    if _relationship_weights is not None:
        return
    path = PROJECT_ROOT / "config" / "relationship_weights.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        _relationship_weights = data.get("weights", {})
        _default_weight = data.get("default_weight", 0.5)
        logger.info(f"Loaded {len(_relationship_weights)} relationship weights from {path}")
    except FileNotFoundError:
        logger.warning(f"Relationship weights file not found: {path}, using default {_default_weight}")
        _relationship_weights = {}


def get_relationship_weight(predicate_uri):
    """
    Get weight for a CIDOC-CRM relationship predicate.
    Higher weights indicate more semantically important relationships.
    Weights are loaded from config/relationship_weights.json.

    Args:
        predicate_uri: Full URI of the predicate

    Returns:
        Float weight between 0 and 1
    """
    _load_relationship_weights()
    local_name = predicate_uri.split('/')[-1].split('#')[-1]

    weight = _relationship_weights.get(predicate_uri)
    if weight is None:
        weight = _relationship_weights.get(local_name, _default_weight)

    return weight
