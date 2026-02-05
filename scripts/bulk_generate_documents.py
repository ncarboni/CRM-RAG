#!/usr/bin/env python3
"""
Bulk Document Generator for CRM_RAG

This script exports all triples from a SPARQL endpoint and processes them locally
using rdflib. This is 100-1000x faster than querying per-entity because:
- Single network request to get all data
- All processing happens in memory
- No SPARQL query overhead per entity

Usage:
    python scripts/bulk_generate_documents.py --dataset mah --endpoint http://localhost:3030/mah/sparql

Or export first, then process:
    # Step 1: Export (can use curl directly if preferred)
    python scripts/bulk_generate_documents.py --dataset mah --export-only

    # Step 2: Process from file
    python scripts/bulk_generate_documents.py --dataset mah --from-file data/exports/mah_dump.ttl
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
import yaml
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL
from tqdm import tqdm

# Global variable for worker processes (set by initializer)
_worker_generator = None


def _init_worker(generator_state):
    """Initialize worker process with shared generator state."""
    global _worker_generator
    _worker_generator = BulkDocumentGenerator.__new__(BulkDocumentGenerator)
    _worker_generator.__dict__.update(generator_state)


def _process_entity_worker(args):
    """Worker function for multiprocessing. Uses global generator state."""
    entity_uri, context_depth, output_dir = args
    global _worker_generator

    try:
        text, label, types = _worker_generator.create_document(entity_uri, context_depth=context_depth)

        # Get images for this entity
        images = _worker_generator.get_entity_images(entity_uri)

        # Save document with images in frontmatter
        _worker_generator.output_dir = Path(output_dir)
        filepath = _worker_generator.save_document(entity_uri, text, label, images=images)

        # Determine primary type
        primary_type = "Unknown"
        if types:
            human_readable = [t for t in types if not _worker_generator.is_technical_class_name(t)]
            primary_type = human_readable[0] if human_readable else "Entity"

        # Get Wikidata ID if available
        wikidata_id = _worker_generator.get_wikidata_id(entity_uri)

        return {
            "uri": entity_uri,
            "label": label,
            "type": primary_type,
            "all_types": types,
            "wikidata_id": wikidata_id,
            "images": images,
            "filepath": os.path.basename(filepath) if filepath else None,
            "error": None
        }
    except Exception as e:
        return {"uri": entity_uri, "error": str(e)}

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Common namespaces
CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
VIR = Namespace("http://w3id.org/vir#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
CRMDIG = Namespace("http://www.ics.forth.gr/isl/CRMdig/")



class BulkDocumentGenerator:
    """Generate entity documents from a bulk RDF export."""

    def __init__(self, dataset_id: str, base_dir: str = None, data_dir: str = None):
        self.dataset_id = dataset_id
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent

        # Data directory (can be overridden for cluster storage)
        # Labels stay in base_dir/data/labels (small, shared files)
        # Output goes to data_dir (can be on scratch for large datasets)
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = self.base_dir / "data"

        # Output directories (use data_dir for large outputs)
        self.output_dir = self.data_dir / "documents" / dataset_id / "entity_documents"
        self.export_dir = self.data_dir / "exports"

        # Load dataset config
        self.datasets_config = self._load_datasets_config()
        self.endpoint = self._get_endpoint()

        # Load configuration files (always from base_dir/data/labels - small shared files)
        self.property_labels = self._load_json("data/labels/property_labels.json")
        self.class_labels = self._load_json("data/labels/class_labels.json")
        self.ontology_classes = set(self._load_json("data/labels/ontology_classes.json") or [])
        self.event_classes = self._load_event_classes()

        # RDF graph
        self.graph = Graph()

        # Indexes built from graph (for fast lookups)
        self.entity_types = defaultdict(set)      # uri -> set of type URIs
        self.entity_labels = {}                    # uri -> label
        self.entity_literals = defaultdict(lambda: defaultdict(list))  # uri -> prop -> [values]
        self.outgoing = defaultdict(list)          # uri -> [(pred, obj)]
        self.incoming = defaultdict(list)          # uri -> [(pred, subj)]
        self.wikidata_ids = {}                     # uri -> wikidata Q-ID
        self.entity_images = defaultdict(list)     # uri -> [image URLs]

        # Load image configuration for this dataset
        self.image_config = self._get_image_config()

    def _load_datasets_config(self) -> dict:
        """Load datasets configuration from YAML."""
        config_path = self.base_dir / "config" / "datasets.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def _get_endpoint(self) -> str:
        """Get SPARQL endpoint for this dataset from config."""
        datasets = self.datasets_config.get("datasets", {})
        if self.dataset_id in datasets:
            return datasets[self.dataset_id].get("endpoint")
        return None

    def _load_json(self, rel_path: str):
        """Load a JSON file from relative path."""
        path = self.base_dir / rel_path
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        logger.warning(f"File not found: {path}")
        return {}

    def _load_event_classes(self) -> set:
        """Load event classes from config."""
        path = self.base_dir / "config" / "event_classes.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Flatten all event class lists
                classes = set()
                for key, values in data.items():
                    if key != "_comment" and isinstance(values, list):
                        classes.update(values)
                return classes
        return set()

    def _get_image_config(self) -> dict:
        """Get image configuration for this dataset from datasets.yaml."""
        datasets = self.datasets_config.get("datasets", {})
        if self.dataset_id in datasets:
            return datasets[self.dataset_id].get("image", {})
        return {}

    def _build_image_index(self):
        """Build image index using SPARQL pattern from dataset configuration.

        The config should contain a SPARQL graph pattern with:
        - ?entity: bound to each entity URI
        - ?url: the image URL to extract
        """
        if not self.image_config:
            logger.info("No image configuration for this dataset")
            return

        sparql_pattern = self.image_config.get("sparql")
        if not sparql_pattern:
            logger.warning("No image SPARQL pattern configured")
            return

        logger.info("Indexing images via SPARQL query...")

        # Extract PREFIX declarations (must be before SELECT)
        lines = sparql_pattern.strip().split('\n')
        prefixes = []
        pattern_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith('PREFIX'):
                prefixes.append(stripped)
            elif stripped:
                pattern_lines.append(line)

        # Build full SPARQL query
        prefix_block = '\n'.join(prefixes)
        pattern_block = '\n'.join(pattern_lines)

        query = f"""
        {prefix_block}
        SELECT ?entity ?url WHERE {{
            {pattern_block}
        }}
        """

        try:
            results = self.graph.query(query)
            count = 0
            for row in results:
                entity_uri = str(row.entity)
                url = str(row.url)
                if url.startswith("http"):
                    self.entity_images[entity_uri].append(url)
                    count += 1

            logger.info(f"Found {count} images for {len(self.entity_images)} entities")

        except Exception as e:
            logger.error(f"Error executing image SPARQL query: {e}")
            logger.debug(f"Query was: {query}")

    def export_from_sparql(self, endpoint: str, output_file: str = None) -> str:
        """Export all triples from SPARQL endpoint."""
        self.export_dir.mkdir(parents=True, exist_ok=True)

        if output_file is None:
            output_file = self.export_dir / f"{self.dataset_id}_dump.ttl"
        else:
            output_file = Path(output_file)

        logger.info(f"Exporting triples from {endpoint}...")

        # Use CONSTRUCT to get all triples
        query = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"

        try:
            response = requests.post(
                endpoint,
                data={"query": query},
                headers={"Accept": "text/turtle"},
                timeout=3600  # 1 hour timeout for large datasets
            )
            response.raise_for_status()

            with open(output_file, 'wb') as f:
                f.write(response.content)

            size_mb = output_file.stat().st_size / (1024 * 1024)
            logger.info(f"Exported {size_mb:.1f} MB to {output_file}")
            return str(output_file)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to export: {e}")
            raise

    def load_graph(self, file_path: str):
        """Load RDF graph from file."""
        logger.info(f"Loading graph from {file_path}...")

        # Determine format from extension
        path = Path(file_path)
        format_map = {
            '.ttl': 'turtle',
            '.nt': 'nt',
            '.n3': 'n3',
            '.rdf': 'xml',
            '.xml': 'xml',
            '.jsonld': 'json-ld',
            '.json': 'json-ld',
        }
        fmt = format_map.get(path.suffix.lower(), 'turtle')

        # Suppress rdflib warnings for invalid date formats (BCE dates like -0075-01-01)
        # These are common in cultural heritage data and rdflib still loads them as strings
        rdflib_logger = logging.getLogger('rdflib.term')
        original_level = rdflib_logger.level
        rdflib_logger.setLevel(logging.ERROR)

        try:
            self.graph.parse(file_path, format=fmt)
        finally:
            rdflib_logger.setLevel(original_level)

        logger.info(f"Loaded {len(self.graph)} triples")

    def build_indexes(self):
        """Build in-memory indexes for fast entity lookups."""
        logger.info("Building indexes...")

        # Schema types to exclude (ontology definitions, not instances)
        schema_types = {
            RDFS.Class, OWL.Class, RDF.Property,
            OWL.ObjectProperty, OWL.DatatypeProperty,
            OWL.AnnotationProperty, OWL.FunctionalProperty,
            OWL.InverseFunctionalProperty, OWL.TransitiveProperty,
            OWL.SymmetricProperty
        }

        # First pass: identify schema entities to exclude
        schema_entities = set()
        for s, p, o in self.graph.triples((None, RDF.type, None)):
            if o in schema_types:
                schema_entities.add(s)

        logger.info(f"Found {len(schema_entities)} schema entities to exclude")

        # Second pass: build indexes
        for s, p, o in tqdm(self.graph, desc="Indexing triples", unit=" triples"):
            # Skip schema entities
            if s in schema_entities:
                continue

            s_str = str(s)
            p_str = str(p)

            if isinstance(o, Literal):
                # Literal value
                prop_name = p_str.split('/')[-1].split('#')[-1]
                self.entity_literals[s_str][prop_name].append(str(o))

                # Extract label
                if prop_name.lower() in ('label', 'preflabel', 'name', 'title'):
                    if s_str not in self.entity_labels:
                        self.entity_labels[s_str] = str(o)

            elif isinstance(o, URIRef):
                o_str = str(o)

                if p == RDF.type:
                    self.entity_types[s_str].add(o_str)
                else:
                    # Object property
                    self.outgoing[s_str].append((p_str, o_str))
                    self.incoming[o_str].append((p_str, s_str))

                    # Extract Wikidata ID (crmdig:L54_is_same-as -> wikidata.org)
                    if p_str.endswith('L54_is_same-as') and o_str.startswith('http://www.wikidata.org/entity/'):
                        wikidata_id = o_str.split('/')[-1]
                        self.wikidata_ids[s_str] = wikidata_id

        # Count entities (those with literals - same logic as original)
        entities_with_literals = set(self.entity_literals.keys())
        logger.info(f"Indexed {len(entities_with_literals)} entities with literals")
        logger.info(f"Total outgoing relationships: {sum(len(v) for v in self.outgoing.values())}")
        logger.info(f"Total incoming relationships: {sum(len(v) for v in self.incoming.values())}")
        logger.info(f"Entities with Wikidata IDs: {len(self.wikidata_ids)}")

        # Index images based on dataset configuration
        self._build_image_index()
        logger.info(f"Entities with images: {len(self.entity_images)}")

    def get_entity_label(self, uri: str) -> str:
        """Get label for an entity."""
        if uri in self.entity_labels:
            return self.entity_labels[uri]

        # Fallback to URI fragment
        return uri.rstrip('/').split('/')[-1].split('#')[-1]

    def get_wikidata_id(self, uri: str) -> str:
        """Get Wikidata Q-ID for an entity if available."""
        return self.wikidata_ids.get(uri)

    def get_entity_images(self, uri: str) -> list:
        """Get image URLs for an entity if available."""
        return self.entity_images.get(uri, [])

    def is_event(self, uri: str) -> bool:
        """Check if entity is an event class instance."""
        types = self.entity_types.get(uri, set())
        return bool(types & self.event_classes)

    def is_schema_predicate(self, predicate: str) -> bool:
        """Check if predicate should be filtered out."""
        schema_patterns = [
            'rdf-syntax-ns#type',
            'rdf-schema#subClassOf',
            'rdf-schema#domain',
            'rdf-schema#range',
            'rdf-schema#label',
            'rdf-schema#comment',
            'owl#',
        ]
        return any(p in predicate for p in schema_patterns)

    def is_technical_class_name(self, class_name: str) -> bool:
        """Check if class name is a technical ontology class."""
        if class_name in self.ontology_classes:
            return True

        if '/' in class_name or '#' in class_name:
            local_name = class_name.split('/')[-1].split('#')[-1]
            if local_name in self.ontology_classes:
                return True

        return False

    def process_relationship(self, subject_uri: str, predicate: str,
                            object_uri: str, subject_label: str = None,
                            object_label: str = None) -> str:
        """Convert RDF relationship to natural language."""
        # Get predicate local name
        simple_pred = predicate.split('/')[-1]
        if '#' in simple_pred:
            simple_pred = simple_pred.split('#')[-1]

        subject_label = subject_label or self.get_entity_label(subject_uri)
        object_label = object_label or self.get_entity_label(object_uri)

        # Look up predicate label
        predicate_label = None
        if self.property_labels:
            predicate_label = (
                self.property_labels.get(predicate) or
                self.property_labels.get(simple_pred)
            )

            if not predicate_label:
                # Try stripped name
                stripped_pred = re.sub(r'^[A-Z]\d+[a-z]?_', '', simple_pred)
                predicate_label = self.property_labels.get(stripped_pred)

        if not predicate_label:
            # Fallback
            stripped_pred = re.sub(r'^[A-Z]\d+[a-z]?_', '', simple_pred)
            predicate_label = stripped_pred.replace('_', ' ').lower()

        # Special handling for P2_has_type
        if "P2_has_type" in predicate or "P2_has_type" in simple_pred:
            return f"{subject_label} is classified as type: {object_label}"

        return f"{subject_label} {predicate_label} {object_label}"

    def get_entity_context(self, entity_uri: str, depth: int = 2, max_statements: int = 50) -> list:
        """Get context statements for an entity using event-aware traversal.

        Args:
            entity_uri: The entity to get context for
            depth: How many hops to traverse (0=none, 1=direct, 2=multi-hop)
            max_statements: Maximum statements to collect (prevents slow processing for hub entities)
        """
        if depth == 0:
            return []

        statements = []
        visited = set()

        start_is_event = self.is_event(entity_uri)
        entity_label = self.get_entity_label(entity_uri)

        def traverse(uri: str, current_depth: int = 0):
            # Early termination if we have enough statements
            if len(statements) >= max_statements:
                return
            if uri in visited or current_depth > depth:
                return
            visited.add(uri)

            current_label = self.get_entity_label(uri)
            current_is_event = self.is_event(uri)

            # Process outgoing relationships
            for pred, obj in self.outgoing.get(uri, []):
                if len(statements) >= max_statements:
                    return

                if self.is_schema_predicate(pred):
                    continue

                obj_label = self.get_entity_label(obj)

                # Skip self-references
                if uri == obj or (current_label and obj_label and
                                 current_label.lower() == obj_label.lower()):
                    continue

                # Event-aware filtering
                target_is_event = self.is_event(obj)

                if start_is_event:
                    should_include = True
                elif current_depth == 0:
                    should_include = True
                elif current_is_event:
                    should_include = True
                else:
                    should_include = False

                if should_include:
                    statement = self.process_relationship(
                        uri, pred, obj, current_label, obj_label
                    )
                    if statement not in statements:
                        statements.append(statement)

                    # Continue traversal if appropriate
                    if current_depth < depth and len(statements) < max_statements:
                        if start_is_event or target_is_event:
                            traverse(obj, current_depth + 1)

            # Process incoming relationships
            for pred, subj in self.incoming.get(uri, []):
                if len(statements) >= max_statements:
                    return

                if self.is_schema_predicate(pred):
                    continue

                subj_label = self.get_entity_label(subj)

                if uri == subj or (current_label and subj_label and
                                  current_label.lower() == subj_label.lower()):
                    continue

                source_is_event = self.is_event(subj)

                if start_is_event:
                    should_include = True
                elif current_depth == 0:
                    should_include = True
                elif current_is_event:
                    should_include = True
                else:
                    should_include = False

                if should_include:
                    statement = self.process_relationship(
                        subj, pred, uri, subj_label, current_label
                    )
                    if statement not in statements:
                        statements.append(statement)

                    if current_depth < depth and len(statements) < max_statements:
                        if start_is_event or source_is_event:
                            traverse(subj, current_depth + 1)

        traverse(entity_uri)
        return statements

    def create_document(self, entity_uri: str, context_depth: int = 2) -> tuple:
        """Create document for an entity. Returns (text, label, types)."""
        # Get label
        entity_label = self.get_entity_label(entity_uri)

        # Get types
        entity_types = []
        for type_uri in self.entity_types.get(entity_uri, []):
            type_label = self.class_labels.get(type_uri)
            if not type_label:
                type_label = type_uri.split('/')[-1].split('#')[-1]
            entity_types.append(type_label)

        # Get literals
        literals = self.entity_literals.get(entity_uri, {})

        # Get relationships
        context_statements = self.get_entity_context(entity_uri, depth=context_depth)

        # Build document text
        text = f"# {entity_label}\n\n"
        text += f"URI: {entity_uri}\n\n"

        # Add types (filter technical ones)
        if entity_types:
            human_readable = [t for t in entity_types if not self.is_technical_class_name(t)]
            if human_readable:
                text += "## Types\n\n"
                for t in human_readable:
                    text += f"- {t}\n"
                text += "\n"

        # Add literals
        if literals:
            text += "## Properties\n\n"
            for prop_name, values in sorted(literals.items()):
                display_name = prop_name.replace('_', ' ').title()
                if len(values) == 1:
                    value_str = values[0]
                    if len(value_str) > 200:
                        value_str = value_str[:200] + "... [truncated]"
                    text += f"- **{display_name}**: {value_str}\n"
                else:
                    text += f"- **{display_name}**:\n"
                    for v in values:
                        v_str = v if len(v) <= 200 else v[:200] + "... [truncated]"
                        text += f"  - {v_str}\n"
            text += "\n"

        # Add relationships
        if context_statements:
            text += "## Relationships\n\n"
            for stmt in context_statements:
                text += f"- {stmt}\n"

        return text, entity_label, entity_types

    def save_document(self, entity_uri: str, text: str, label: str, images: list = None) -> str:
        """Save document to file. Returns filepath.

        Args:
            entity_uri: The entity URI
            text: Document text content
            label: Entity label
            images: Optional list of image URLs
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create safe filename
        safe_label = re.sub(r'[^\w\s-]', '', label)
        safe_label = re.sub(r'[-\s]+', '_', safe_label)
        safe_label = safe_label[:100]

        uri_hash = hashlib.md5(entity_uri.encode()).hexdigest()[:8]
        filename = f"{safe_label}_{uri_hash}.md"
        filepath = self.output_dir / filename

        # Build metadata header with optional images
        metadata_lines = [
            "---",
            f"URI: {entity_uri}",
            f"Label: {label}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        # Add images if present
        if images:
            metadata_lines.append("Images:")
            for img_url in images:
                metadata_lines.append(f"  - {img_url}")

        metadata_lines.append("---")
        metadata_lines.append("")
        metadata = "\n".join(metadata_lines) + "\n"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(metadata)
            f.write(text)

        return str(filepath)

    def _process_single_entity(self, entity_uri: str, context_depth: int) -> dict:
        """Process a single entity and return metadata. Used for sequential processing."""
        try:
            text, label, types = self.create_document(entity_uri, context_depth=context_depth)

            # Get images for this entity
            images = self.get_entity_images(entity_uri)

            # Save document with images in frontmatter
            filepath = self.save_document(entity_uri, text, label, images=images)

            # Determine primary type
            primary_type = "Unknown"
            if types:
                human_readable = [t for t in types if not self.is_technical_class_name(t)]
                primary_type = human_readable[0] if human_readable else "Entity"

            # Get Wikidata ID if available
            wikidata_id = self.get_wikidata_id(entity_uri)

            return {
                "uri": entity_uri,
                "label": label,
                "type": primary_type,
                "all_types": types,
                "wikidata_id": wikidata_id,
                "images": images,
                "filepath": os.path.basename(filepath) if filepath else None,
                "error": None
            }
        except Exception as e:
            return {"uri": entity_uri, "error": str(e)}

    def _get_picklable_state(self):
        """Get state dict that can be pickled for multiprocessing."""
        # Only include the data needed for document generation (not the rdflib Graph)
        return {
            'dataset_id': self.dataset_id,
            'base_dir': self.base_dir,
            'output_dir': self.output_dir,
            'property_labels': self.property_labels,
            'class_labels': self.class_labels,
            'ontology_classes': self.ontology_classes,
            'event_classes': self.event_classes,
            'entity_types': dict(self.entity_types),  # Convert defaultdict
            'entity_labels': self.entity_labels,
            'entity_literals': {k: dict(v) for k, v in self.entity_literals.items()},
            'outgoing': dict(self.outgoing),
            'incoming': dict(self.incoming),
            'wikidata_ids': self.wikidata_ids,
            'entity_images': dict(self.entity_images),  # Image URLs per entity
            'image_config': self.image_config,
        }

    def generate_all_documents(self, context_depth: int = 2, workers: int = 1):
        """Generate documents for all entities.

        Args:
            context_depth: Depth for relationship traversal (default 2 for CIDOC-CRM,
                          use 1 or 0 for faster but less rich documents)
            workers: Number of parallel workers (1 = sequential, >1 = multiprocessing)
        """
        import time

        # Get entities with literals (same as original logic)
        entities = list(self.entity_literals.keys())
        total = len(entities)
        logger.info(f"Generating documents for {total} entities (context_depth={context_depth}, workers={workers})...")

        # Clean output directory
        if self.output_dir.exists():
            import shutil
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        documents_metadata = []
        errors = 0
        start_time = time.time()

        if workers > 1:
            # Multiprocessing for true parallelism
            logger.info(f"Using multiprocessing with {workers} workers...")

            # Prepare state for workers
            generator_state = self._get_picklable_state()
            output_dir_str = str(self.output_dir)

            # Prepare work items
            work_items = [(uri, context_depth, output_dir_str) for uri in entities]

            # Use 'fork' on Unix for efficiency (copy-on-write), 'spawn' on Windows
            ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')

            with ctx.Pool(
                processes=workers,
                initializer=_init_worker,
                initargs=(generator_state,)
            ) as pool:
                # Use imap_unordered for better progress tracking
                with tqdm(total=total, desc="Generating documents", unit=" entities") as pbar:
                    for result in pool.imap_unordered(_process_entity_worker, work_items, chunksize=100):
                        if result.get("error"):
                            errors += 1
                            if errors <= 10:
                                logger.error(f"Error processing {result['uri']}: {result['error']}")
                        else:
                            documents_metadata.append(result)
                        pbar.update(1)

            elapsed = time.time() - start_time
            logger.info(f"Completed in {elapsed/60:.1f} min ({total/elapsed:.1f} entities/sec)")

        else:
            # Sequential processing with progress updates
            last_log_time = start_time

            for i, entity_uri in enumerate(tqdm(entities, desc="Generating documents", unit=" entities")):
                result = self._process_single_entity(entity_uri, context_depth)
                if result.get("error"):
                    errors += 1
                    if errors <= 10:
                        logger.error(f"Error processing {entity_uri}: {result['error']}")
                else:
                    documents_metadata.append(result)

                # Log progress every 30 seconds
                current_time = time.time()
                if current_time - last_log_time > 30:
                    elapsed = current_time - start_time
                    rate = (i + 1) / elapsed
                    remaining = (total - i - 1) / rate if rate > 0 else 0
                    logger.info(f"Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%) - "
                               f"{rate:.1f} entities/sec - "
                               f"ETA: {remaining/60:.1f} min")
                    last_log_time = current_time

        # Count entities with images
        entities_with_images = sum(1 for doc in documents_metadata if doc.get("images"))
        total_images = sum(len(doc.get("images", [])) for doc in documents_metadata)

        # Save metadata
        metadata_path = self.output_dir.parent / "documents_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset_id": self.dataset_id,
                "total_documents": len(documents_metadata),
                "entities_with_images": entities_with_images,
                "total_images": total_images,
                "generated_at": datetime.now().isoformat(),
                "documents": documents_metadata
            }, f, indent=2)

        logger.info("=" * 60)
        logger.info("DOCUMENT GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Generated {len(documents_metadata)} documents")
        logger.info(f"Entities with images: {entities_with_images} ({total_images} total images)")
        logger.info(f"Documents: {self.output_dir}")
        logger.info(f"Metadata: {metadata_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  1. Transfer to cluster:")
        logger.info(f"     rsync -avz data/documents/{self.dataset_id}/ user@cluster:CRM_RAG/data/documents/{self.dataset_id}/")
        logger.info(f"  2. On cluster:")
        logger.info(f"     python main.py --env .env.cluster --dataset {self.dataset_id} --embed-from-docs --process-only")
        logger.info("=" * 60)

        return len(documents_metadata)


def get_generator(dataset_id: str, base_dir: str = None, endpoint: str = None) -> "BulkDocumentGenerator":
    """
    Factory function to create and return an initialized BulkDocumentGenerator.

    This function is useful for importing the generator from other scripts
    (e.g., cluster_pipeline.py) without duplicating initialization logic.

    Args:
        dataset_id: Dataset identifier (from datasets.yaml)
        base_dir: Base directory for the project (defaults to project root)
        endpoint: Optional SPARQL endpoint URL (overrides datasets.yaml)

    Returns:
        Initialized BulkDocumentGenerator instance

    Example:
        from scripts.bulk_generate_documents import get_generator

        generator = get_generator("mah")
        generator.export_from_sparql(generator.endpoint)
        generator.load_graph("data/exports/mah_dump.ttl")
        generator.build_indexes()
        generator.generate_all_documents(workers=8)
    """
    generator = BulkDocumentGenerator(dataset_id, base_dir)

    # Override endpoint if provided
    if endpoint:
        generator.endpoint = endpoint

    return generator


def main():
    parser = argparse.ArgumentParser(description="Bulk generate entity documents from RDF")
    parser.add_argument("--dataset", required=True, help="Dataset ID (e.g., mah)")
    parser.add_argument("--endpoint", help="SPARQL endpoint URL (overrides datasets.yaml)")
    parser.add_argument("--from-file", help="Load from existing export file instead of querying")
    parser.add_argument("--export-only", action="store_true", help="Only export, don't generate documents")
    parser.add_argument("--base-dir", help="Base directory (default: project root)")
    parser.add_argument("--context-depth", type=int, default=2, choices=[0, 1, 2],
                       help="Relationship traversal depth: 0=none, 1=direct, 2=multi-hop (default, recommended for CIDOC-CRM)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers (default: 1, try 4-8 for speedup)")

    args = parser.parse_args()

    generator = BulkDocumentGenerator(args.dataset, args.base_dir)

    # Determine input file
    if args.from_file:
        input_file = args.from_file
    else:
        # Get endpoint from args or config
        endpoint = args.endpoint or generator.endpoint
        if endpoint:
            logger.info(f"Using endpoint from {'command line' if args.endpoint else 'datasets.yaml'}: {endpoint}")
            input_file = generator.export_from_sparql(endpoint)
            if args.export_only:
                logger.info(f"Export complete: {input_file}")
                return
        else:
            # Try default export file
            default_file = generator.export_dir / f"{args.dataset}_dump.ttl"
            if default_file.exists():
                input_file = str(default_file)
                logger.info(f"Using existing export: {input_file}")
            else:
                parser.error(f"No endpoint found for dataset '{args.dataset}' in datasets.yaml. "
                           f"Use --endpoint or --from-file")

    # Load and process
    generator.load_graph(input_file)
    generator.build_indexes()
    generator.generate_all_documents(context_depth=args.context_depth, workers=args.workers)


if __name__ == "__main__":
    main()
