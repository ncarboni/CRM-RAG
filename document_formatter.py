"""
Document formatting utilities for CIDOC-CRM predicates and classes.

Contains pure functions for predicate filtering, class name checks,
and relationship weight assignment. Extracted from UniversalRagSystem.
"""

import logging
import re

logger = logging.getLogger(__name__)


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


def get_relationship_weight(predicate_uri):
    """
    Get weight for a CIDOC-CRM relationship predicate.
    Higher weights indicate more semantically important relationships.

    Args:
        predicate_uri: Full URI of the predicate

    Returns:
        Float weight between 0 and 1
    """
    local_name = predicate_uri.split('/')[-1].split('#')[-1]

    weights = {
        # Spatial relationships (high weight - location is key context)
        "P89_falls_within": 0.9,
        "P89i_contains": 0.9,
        "P55_has_current_location": 0.9,
        "P55i_currently_holds": 0.9,

        # Physical composition
        "P56_bears_feature": 0.8,
        "P56i_is_found_on": 0.8,
        "P46_is_composed_of": 0.8,
        "P46i_forms_part_of": 0.8,

        # Creation/Production (important for authorship)
        "P108_has_produced": 0.85,
        "P108i_was_produced_by": 0.85,
        "P14_carried_out_by": 0.85,
        "P14i_performed": 0.85,
        "P94_has_created": 0.85,
        "P94i_was_created_by": 0.85,

        # Visual representation (VIR)
        "K24_portray": 0.7,
        "K24i_is_portrayed_in": 0.7,
        "K34_illustrates": 0.7,
        "K34i_is_illustrated_by": 0.7,

        # Type classification
        "P2_has_type": 0.6,
        "P2i_is_type_of": 0.6,

        # Documentation/Reference
        "P67_refers_to": 0.5,
        "P67i_is_referred_to_by": 0.5,
        "P70_documents": 0.5,
        "P70i_is_documented_in": 0.5,

        # Temporal
        "P4_has_time-span": 0.6,
        "P4i_is_time-span_of": 0.6,

        # Identification
        "P1_is_identified_by": 0.4,
        "P1i_identifies": 0.4,
    }

    weight = weights.get(predicate_uri)
    if weight is None:
        weight = weights.get(local_name, 0.5)

    return weight
