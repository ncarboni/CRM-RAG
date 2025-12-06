"""
Extract property labels and class definitions from ontology files and save to JSON.
This script parses CIDOC-CRM, VIR, and CRMdig ontologies to extract:
- rdfs:label for all properties
- All class URIs (owl:Class, rdfs:Class) for technical class filtering
"""

import os
import json
import logging
from rdflib import Graph, Namespace, RDF, RDFS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_property_labels(ontology_dir='ontology'):
    """
    Extract property labels from all ontology files in the given directory.
    Returns a dictionary mapping property URIs to their English labels.
    """
    property_labels = {}

    # Define common namespaces
    CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
    VIR = Namespace("http://w3id.org/vir#")
    CRMDIG = Namespace("http://www.ics.forth.gr/isl/CRMdig/")

    # Find all ontology files
    ontology_files = []
    for filename in os.listdir(ontology_dir):
        if filename.endswith(('.ttl', '.rdf', '.owl', '.n3')):
            ontology_files.append(os.path.join(ontology_dir, filename))

    logger.info(f"Found {len(ontology_files)} ontology files to process")

    # Process each ontology file
    for filepath in ontology_files:
        logger.info(f"Processing: {filepath}")

        # Create a new graph and parse the file
        g = Graph()
        try:
            # Try to parse the file (rdflib auto-detects format)
            g.parse(filepath)
            logger.info(f"  Loaded {len(g)} triples")

            # Query for PROPERTIES ONLY (not classes!)
            # Explicitly require the entity to be a property type
            query_with_labels = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT ?property ?label ?lang
            WHERE {
                # MUST be one of these property types
                {
                    ?property a rdf:Property .
                } UNION {
                    ?property a owl:ObjectProperty .
                } UNION {
                    ?property a owl:DatatypeProperty .
                }
                # Get labels for these properties
                ?property rdfs:label ?label .
                BIND(LANG(?label) AS ?lang)
            }
            """

            # Query for properties without explicit labels (like CRMdig)
            query_without_labels = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT DISTINCT ?property
            WHERE {
                {
                    ?property a rdf:Property .
                } UNION {
                    ?property a owl:ObjectProperty .
                } UNION {
                    ?property a owl:DatatypeProperty .
                }
                FILTER NOT EXISTS { ?property rdfs:label ?anyLabel }
            }
            """

            import re

            # First, process properties with explicit labels
            results_with_labels = g.query(query_with_labels)

            for row in results_with_labels:
                prop_uri = str(row.property)
                label = str(row.label)
                lang = str(row.lang) if row.lang else None

                # Only store full URI mapping to save space
                # The system always receives full URIs from SPARQL queries
                if lang == 'en' or lang == '' or lang == 'None':
                    property_labels[prop_uri] = label
                elif prop_uri not in property_labels:
                    # If no English label exists yet, use any available label as fallback
                    property_labels[prop_uri] = label

            # Second, process properties without explicit labels (like CRMdig)
            # For these, we derive the label from the property name itself
            results_without_labels = g.query(query_without_labels)
            count_without_labels = 0

            for row in results_without_labels:
                prop_uri = str(row.property)

                # Extract local name
                if '#' in prop_uri:
                    local_name = prop_uri.split('#')[-1]
                elif '/' in prop_uri:
                    local_name = prop_uri.split('/')[-1]
                else:
                    local_name = prop_uri

                if not local_name:
                    continue

                # For properties without labels, strip the prefix code and convert to natural language
                # E.g., "L54_is_same-as" -> "is same-as"
                stripped_name = re.sub(r'^[A-Z]\d+[a-z]?_', '', local_name)
                # Convert underscores to spaces for the label
                derived_label = stripped_name.replace('_', ' ').replace('-', ' ')

                # Only store full URI mapping
                if prop_uri not in property_labels:
                    property_labels[prop_uri] = derived_label
                    count_without_labels += 1

            logger.info(f"  Extracted {len([r for r in results_with_labels])} properties with labels")
            logger.info(f"  Extracted {count_without_labels} properties without labels (derived from names)")

        except Exception as e:
            logger.error(f"  Error parsing {filepath}: {str(e)}")
            continue

    logger.info(f"\nTotal unique property labels extracted: {len(property_labels)}")

    # Log some examples
    logger.info("\nExample property labels:")
    for i, (key, value) in enumerate(list(property_labels.items())[:10]):
        logger.info(f"  {key} -> {value}")

    return property_labels


def extract_ontology_classes(ontology_dir='ontology'):
    """
    Extract all class URIs from ontology files.
    Returns a set of class URIs (as strings) that represent technical ontology classes.
    """
    ontology_classes = set()
    class_labels_map = {}  # New: map class URIs to English labels

    # Find all ontology files
    ontology_files = []
    for filename in os.listdir(ontology_dir):
        if filename.endswith(('.ttl', '.rdf', '.owl', '.n3')):
            ontology_files.append(os.path.join(ontology_dir, filename))

    logger.info(f"Extracting classes from {len(ontology_files)} ontology files")

    # Process each ontology file
    for filepath in ontology_files:
        logger.info(f"Processing classes in: {filepath}")

        # Create a new graph and parse the file
        g = Graph()
        try:
            # Try to parse the file (rdflib auto-detects format)
            g.parse(filepath)

            # Query for all classes with their labels and language tags
            # Note: VIR uses rdf:Class, CIDOC-CRM uses owl:Class, others use rdfs:Class
            query = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT DISTINCT ?class ?label ?lang
            WHERE {
                {
                    ?class a owl:Class .
                } UNION {
                    ?class a rdfs:Class .
                } UNION {
                    ?class a rdf:Class .
                }
                OPTIONAL {
                    ?class rdfs:label ?label .
                    BIND(LANG(?label) AS ?lang)
                }
            }
            """

            results = g.query(query)

            # Group labels by class URI to select English labels preferentially
            class_labels = {}
            for row in results:
                class_uri = str(row['class'])
                label = str(row['label']) if row['label'] else None
                lang = str(row['lang']) if row['lang'] else None

                if class_uri not in class_labels:
                    class_labels[class_uri] = []

                if label:
                    class_labels[class_uri].append((label, lang))

            # Process each class and add to ontology_classes
            count = 0
            for class_uri, labels in class_labels.items():
                # Extract and store ONLY the local name (e.g., E22_Human-Made_Object)
                # We don't need the full URI because is_technical_class_name() extracts
                # local names from URIs for comparison
                if '#' in class_uri:
                    local_name = class_uri.split('#')[-1]
                elif '/' in class_uri:
                    local_name = class_uri.split('/')[-1]
                else:
                    local_name = class_uri

                # Store the local name for technical class filtering
                if local_name:
                    ontology_classes.add(local_name)

                # NEW: Select English label preferentially
                english_label = None
                fallback_label = None

                for label, lang in labels:
                    if lang == 'en':
                        english_label = label
                        break
                    elif lang == '' or lang == 'None':
                        fallback_label = label

                # Use English label if available, otherwise fallback
                selected_label = english_label or fallback_label

                # Store the class URI -> English label mapping
                if selected_label:
                    class_labels_map[class_uri] = selected_label

                count += 1

            logger.info(f"  Extracted {count} classes (local names only)")
            logger.info(f"  Extracted {len(class_labels_map)} class labels (English)")

        except Exception as e:
            logger.error(f"  Error parsing {filepath}: {str(e)}")
            continue

    logger.info(f"\nTotal unique classes extracted: {len(ontology_classes)}")
    logger.info(f"Total unique class labels extracted: {len(class_labels_map)}")

    # Log some examples
    logger.info("\nExample classes:")
    for i, class_name in enumerate(list(ontology_classes)[:10]):
        logger.info(f"  {class_name}")

    logger.info("\nExample class labels:")
    for i, (class_uri, label) in enumerate(list(class_labels_map.items())[:10]):
        logger.info(f"  {class_uri} -> {label}")

    return ontology_classes, class_labels_map


def save_property_labels(property_labels, output_file='property_labels.json'):
    """Save property labels to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(property_labels, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved property labels to {output_file}")


def save_ontology_classes(ontology_classes, output_file='ontology_classes.json'):
    """Save ontology classes to JSON file"""
    # Convert set to sorted list for better readability
    classes_list = sorted(list(ontology_classes))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(classes_list, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved ontology classes to {output_file}")


def save_class_labels(class_labels, output_file='class_labels.json'):
    """Save class labels (URI -> English label mapping) to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(class_labels, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved class labels to {output_file}")


def run_extraction(ontology_dir='ontology', output_file='property_labels.json', classes_file='ontology_classes.json', class_labels_file='class_labels.json'):
    """
    Main function to extract and save property labels and ontology classes.
    Can be called from other modules.

    Args:
        ontology_dir: Directory containing ontology files
        output_file: Output file for property labels
        classes_file: Output file for ontology classes
        class_labels_file: Output file for class labels (URI -> English label)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract labels from ontology files
        logger.info("=" * 60)
        logger.info("EXTRACTING PROPERTY LABELS")
        logger.info("=" * 60)
        labels = extract_property_labels(ontology_dir)

        # Extract classes from ontology files
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTING ONTOLOGY CLASSES")
        logger.info("=" * 60)
        classes, class_labels = extract_ontology_classes(ontology_dir)

        # Save to JSON
        save_property_labels(labels, output_file)
        save_ontology_classes(classes, classes_file)
        save_class_labels(class_labels, class_labels_file)

        logger.info(f"\n✓ Successfully extracted {len(labels)} property labels")
        logger.info(f"✓ Saved to {output_file}")
        logger.info(f"✓ Successfully extracted {len(classes)} ontology classes")
        logger.info(f"✓ Saved to {classes_file}")
        logger.info(f"✓ Successfully extracted {len(class_labels)} class labels")
        logger.info(f"✓ Saved to {class_labels_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract ontology data: {str(e)}")
        return False


if __name__ == '__main__':
    # Run extraction when called directly
    success = run_extraction('ontology', 'property_labels.json', 'ontology_classes.json', 'class_labels.json')

    if success:
        print(f"\n✓ Successfully extracted property labels, ontology classes, and class labels")
        print("✓ Saved to property_labels.json, ontology_classes.json, and class_labels.json")
    else:
        print("\n✗ Failed to extract ontology data")
        import sys
        sys.exit(1)
