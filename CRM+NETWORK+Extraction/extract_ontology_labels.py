"""
Extract property labels from ontology files and save to JSON.
This script parses CIDOC-CRM, VIR, and CRMdig ontologies to extract
rdfs:label for all properties.
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

            # Query for all properties with labels
            # A property is typically an rdf:Property or owl:ObjectProperty
            query_with_labels = """
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX owl: <http://www.w3.org/2002/07/owl#>

            SELECT ?property ?label ?lang
            WHERE {
                ?property rdfs:label ?label .
                OPTIONAL { ?property a rdf:Property }
                OPTIONAL { ?property a owl:ObjectProperty }
                OPTIONAL { ?property a owl:DatatypeProperty }
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


def save_property_labels(property_labels, output_file='property_labels.json'):
    """Save property labels to JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(property_labels, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSaved property labels to {output_file}")


def run_extraction(ontology_dir='ontology', output_file='property_labels.json'):
    """
    Main function to extract and save property labels.
    Can be called from other modules.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract labels from ontology files
        labels = extract_property_labels(ontology_dir)

        # Save to JSON
        save_property_labels(labels, output_file)

        logger.info(f"\n✓ Successfully extracted {len(labels)} property labels")
        logger.info(f"✓ Saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract property labels: {str(e)}")
        return False


if __name__ == '__main__':
    # Run extraction when called directly
    success = run_extraction('ontology', 'property_labels.json')

    if success:
        print(f"\n✓ Successfully extracted property labels")
        print("✓ Saved to property_labels.json")
    else:
        print("\n✗ Failed to extract property labels")
        import sys
        sys.exit(1)
