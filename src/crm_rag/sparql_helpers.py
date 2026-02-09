"""
Batch SPARQL query infrastructure for the CRM_RAG system.

Contains the BatchSparqlClient class that encapsulates all batch SPARQL
query methods for fetching types, relationships, literals, and other
entity data from SPARQL endpoints. Extracted from UniversalRagSystem.
"""

import logging
from typing import Dict, List, Optional, Tuple

from SPARQLWrapper import TSV, JSON, POST

logger = logging.getLogger(__name__)


class BatchSparqlClient:
    """Handles all batch SPARQL query operations against an endpoint."""

    # Default batch sizes (mirrors RetrievalConfig constants)
    DEFAULT_BATCH_SIZE = 1000
    DEFAULT_RETRY_SIZE = 100

    def __init__(self, sparql):
        """
        Args:
            sparql: SPARQLWrapper instance configured with the endpoint URL.
        """
        self.sparql = sparql

    def batch_query_tsv(self, query: str) -> List[List[str]]:
        """
        Execute a SPARQL query using TSV format for 3x faster parsing.

        TSV returns raw bytes with tab-separated values. URIs are wrapped in <...>,
        literals in "..." with optional @lang or ^^type suffixes.

        Returns:
            List of rows, each row being a list of raw string values (URIs unwrapped).
        """
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(TSV)
        self.sparql.setMethod(POST)
        raw = self.sparql.query().convert()
        # Restore default format for non-batch queries
        self.sparql.setReturnFormat(JSON)

        rows = []
        lines = raw.decode('utf-8').split('\n')
        for line in lines[1:]:  # skip header
            if not line.strip():
                continue
            cols = line.split('\t')
            # Unwrap URIs: <http://...> -> http://...
            # Extract literal values: "value"@en -> value, "value"^^<type> -> value
            parsed = []
            for col in cols:
                if col.startswith('<') and col.endswith('>'):
                    parsed.append(col[1:-1])
                elif col.startswith('"'):
                    # Strip quotes, language tags, and datatype suffixes
                    # Formats: "val", "val"@en, "val"^^<xsd:string>
                    end_quote = col.rfind('"')
                    if end_quote > 0:
                        parsed.append(col[1:end_quote])
                    else:
                        parsed.append(col.strip('"'))
                else:
                    parsed.append(col)
            rows.append(parsed)
        return rows

    def escape_uri_for_values(self, uri: str) -> str:
        """Escape a URI for use in SPARQL VALUES clause."""
        if '<' in uri or '>' in uri or '"' in uri or ' ' in uri:
            logger.warning(f"URI contains special characters, skipping: {uri[:100]}")
            return None
        return f"<{uri}>"

    def batch_fetch_types(self, uris: List[str], batch_size: int = None) -> Dict[str, set]:
        """
        Batch fetch rdf:type for multiple URIs.

        Args:
            uris: List of entity URIs
            batch_size: Number of URIs per query (default: DEFAULT_BATCH_SIZE)

        Returns:
            Dict mapping URI -> set of type URIs
        """
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        result = {}

        for i in range(0, len(uris), batch_size):
            batch = uris[i:i + batch_size]
            escaped = [self.escape_uri_for_values(u) for u in batch]
            escaped = [e for e in escaped if e is not None]

            if not escaped:
                continue

            values_clause = " ".join(escaped)

            query = f"""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?entity ?type WHERE {{
                VALUES ?entity {{ {values_clause} }}
                ?entity rdf:type ?type .
                FILTER(STRSTARTS(STR(?type), "http://"))
            }}
            """

            try:
                rows = self.batch_query_tsv(query)
                for row in rows:
                    if len(row) >= 2:
                        entity, type_uri = row[0], row[1]
                        if entity not in result:
                            result[entity] = set()
                        result[entity].add(type_uri)

            except Exception as e:
                logger.warning(f"Batch type query failed for batch {i//batch_size}: {str(e)}")
                if batch_size > self.DEFAULT_RETRY_SIZE:
                    logger.info(f"Retrying with smaller batch size {self.DEFAULT_RETRY_SIZE}")
                    partial = self.batch_fetch_types(batch, self.DEFAULT_RETRY_SIZE)
                    result.update(partial)

        return result

    def batch_query_outgoing(self, uris: List[str], batch_size: int = None) -> Dict[str, List[Tuple[str, str, Optional[str]]]]:
        """
        Batch query outgoing relationships for multiple URIs.

        Args:
            uris: List of entity URIs
            batch_size: Number of URIs per query

        Returns:
            Dict mapping entity URI -> list of (predicate, object_uri, object_label) tuples
        """
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        result = {}

        for i in range(0, len(uris), batch_size):
            batch = uris[i:i + batch_size]
            escaped = [self.escape_uri_for_values(u) for u in batch]
            escaped = [e for e in escaped if e is not None]

            if not escaped:
                continue

            values_clause = " ".join(escaped)

            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?entity ?p ?o ?oLabel WHERE {{
                VALUES ?entity {{ {values_clause} }}
                ?entity ?p ?o .
                FILTER(isURI(?o))
                OPTIONAL {{ ?o rdfs:label ?oLabel }}
            }}
            """

            try:
                rows = self.batch_query_tsv(query)
                for row in rows:
                    if len(row) >= 3:
                        entity, pred, obj = row[0], row[1], row[2]
                        obj_label = row[3] if len(row) >= 4 and row[3] else None
                        if entity not in result:
                            result[entity] = []
                        result[entity].append((pred, obj, obj_label))

            except Exception as e:
                logger.warning(f"Batch outgoing query failed for batch {i//batch_size}: {str(e)}")
                if batch_size > self.DEFAULT_RETRY_SIZE:
                    logger.info(f"Retrying with smaller batch size {self.DEFAULT_RETRY_SIZE}")
                    partial = self.batch_query_outgoing(batch, self.DEFAULT_RETRY_SIZE)
                    for k, v in partial.items():
                        if k not in result:
                            result[k] = []
                        result[k].extend(v)

        return result

    def batch_query_incoming(self, uris: List[str], batch_size: int = None) -> Dict[str, List[Tuple[str, str, Optional[str]]]]:
        """
        Batch query incoming relationships for multiple URIs.

        Args:
            uris: List of entity URIs
            batch_size: Number of URIs per query

        Returns:
            Dict mapping entity URI -> list of (subject_uri, predicate, subject_label) tuples
        """
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        result = {}

        for i in range(0, len(uris), batch_size):
            batch = uris[i:i + batch_size]
            escaped = [self.escape_uri_for_values(u) for u in batch]
            escaped = [e for e in escaped if e is not None]

            if not escaped:
                continue

            values_clause = " ".join(escaped)

            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?s ?p ?entity ?sLabel WHERE {{
                VALUES ?entity {{ {values_clause} }}
                ?s ?p ?entity .
                FILTER(isURI(?s))
                OPTIONAL {{ ?s rdfs:label ?sLabel }}
            }}
            """

            try:
                rows = self.batch_query_tsv(query)
                for row in rows:
                    if len(row) >= 3:
                        subj, pred, entity = row[0], row[1], row[2]
                        subj_label = row[3] if len(row) >= 4 and row[3] else None
                        if entity not in result:
                            result[entity] = []
                        result[entity].append((subj, pred, subj_label))

            except Exception as e:
                logger.warning(f"Batch incoming query failed for batch {i//batch_size}: {str(e)}")
                if batch_size > self.DEFAULT_RETRY_SIZE:
                    logger.info(f"Retrying with smaller batch size {self.DEFAULT_RETRY_SIZE}")
                    partial = self.batch_query_incoming(batch, self.DEFAULT_RETRY_SIZE)
                    for k, v in partial.items():
                        if k not in result:
                            result[k] = []
                        result[k].extend(v)

        return result

    def batch_fetch_literals(self, uris: List[str], batch_size: int = None) -> Dict[str, Dict[str, List[str]]]:
        """
        Batch fetch literal properties for multiple URIs.

        Args:
            uris: List of entity URIs
            batch_size: Number of URIs per query

        Returns:
            Dict mapping URI -> {property_name: [values]}
        """
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        result = {}

        for i in range(0, len(uris), batch_size):
            batch = uris[i:i + batch_size]
            escaped = [self.escape_uri_for_values(u) for u in batch]
            escaped = [e for e in escaped if e is not None]

            if not escaped:
                continue

            values_clause = " ".join(escaped)

            query = f"""
            SELECT ?entity ?property ?value WHERE {{
                VALUES ?entity {{ {values_clause} }}
                ?entity ?property ?value .
                FILTER(isLiteral(?value))
            }}
            """

            try:
                rows = self.batch_query_tsv(query)
                for row in rows:
                    if len(row) >= 3:
                        entity, prop, value = row[0], row[1], row[2]
                        # Store by property local name
                        prop_name = prop.split('/')[-1].split('#')[-1]
                        if entity not in result:
                            result[entity] = {}
                        if prop_name not in result[entity]:
                            result[entity][prop_name] = []
                        result[entity][prop_name].append(value)

            except Exception as e:
                logger.warning(f"Batch literals query failed for batch {i//batch_size}: {str(e)}")
                if batch_size > self.DEFAULT_RETRY_SIZE:
                    logger.info(f"Retrying with smaller batch size {self.DEFAULT_RETRY_SIZE}")
                    partial = self.batch_fetch_literals(batch, self.DEFAULT_RETRY_SIZE)
                    for k, v in partial.items():
                        if k not in result:
                            result[k] = {}
                        for prop, vals in v.items():
                            if prop not in result[k]:
                                result[k][prop] = []
                            result[k][prop].extend(vals)

        return result

    def batch_fetch_type_labels(self, type_uris: set, batch_size: int = None) -> Dict[str, str]:
        """
        Batch fetch labels for type URIs.

        Args:
            type_uris: Set of type URIs
            batch_size: Number of URIs per query

        Returns:
            Dict mapping type URI -> label
        """
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        result = {}
        uris = list(type_uris)

        for i in range(0, len(uris), batch_size):
            batch = uris[i:i + batch_size]
            escaped = [self.escape_uri_for_values(u) for u in batch]
            escaped = [e for e in escaped if e is not None]

            if not escaped:
                continue

            values_clause = " ".join(escaped)

            query = f"""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?type ?label WHERE {{
                VALUES ?type {{ {values_clause} }}
                ?type rdfs:label ?label .
                FILTER(LANG(?label) = "en" || LANG(?label) = "")
            }}
            """

            try:
                rows = self.batch_query_tsv(query)
                for row in rows:
                    if len(row) >= 2:
                        type_uri, label = row[0], row[1]
                        result[type_uri] = label

            except Exception as e:
                logger.warning(f"Batch type labels query failed: {str(e)}")

        return result

    def batch_fetch_wikidata_ids(self, uris: List[str], batch_size: int = None) -> Dict[str, str]:
        """
        Batch fetch Wikidata IDs (crmdig:L54_is_same-as) for multiple URIs.

        Args:
            uris: List of entity URIs
            batch_size: Number of URIs per query (default: DEFAULT_BATCH_SIZE)

        Returns:
            Dict mapping entity URI -> Wikidata Q-ID string (e.g. "Q12345")
        """
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE

        result = {}

        for i in range(0, len(uris), batch_size):
            batch = uris[i:i + batch_size]
            escaped = [self.escape_uri_for_values(u) for u in batch]
            escaped = [e for e in escaped if e is not None]

            if not escaped:
                continue

            values_clause = " ".join(escaped)

            query = f"""
            PREFIX crmdig: <http://www.ics.forth.gr/isl/CRMdig/>
            SELECT ?entity ?wikidata WHERE {{
                VALUES ?entity {{ {values_clause} }}
                ?entity crmdig:L54_is_same-as ?wikidata .
                FILTER(STRSTARTS(STR(?wikidata), "http://www.wikidata.org/entity/"))
            }}
            """

            try:
                rows = self.batch_query_tsv(query)
                for row in rows:
                    if len(row) >= 2:
                        entity, wikidata_uri = row[0], row[1]
                        # Extract the Q-ID from the URI
                        wikidata_id = wikidata_uri.split('/')[-1]
                        if entity not in result:
                            result[entity] = wikidata_id

            except Exception as e:
                logger.warning(f"Batch wikidata query failed for batch {i//batch_size}: {str(e)}")
                if batch_size > self.DEFAULT_RETRY_SIZE:
                    logger.info(f"Retrying with smaller batch size {self.DEFAULT_RETRY_SIZE}")
                    partial = self.batch_fetch_wikidata_ids(batch, self.DEFAULT_RETRY_SIZE)
                    result.update(partial)

        return result

    def build_image_index(self, dataset_config: dict) -> Dict[str, List[str]]:
        """Build image index using SPARQL pattern from dataset configuration.

        Reads the 'image.sparql' graph pattern from dataset_config, wraps it
        in a SELECT ?entity ?url query, and runs it against the SPARQL endpoint.

        Args:
            dataset_config: Dataset configuration dict with optional 'image.sparql' key.

        Returns:
            Dict mapping entity URI -> list of image URLs. Empty dict if no
            image config is present.
        """
        image_config = dataset_config.get("image")
        if not image_config:
            return {}

        sparql_pattern = image_config.get("sparql")
        if not sparql_pattern:
            logger.warning("Image config exists but has no 'sparql' pattern")
            return {}

        logger.info("Indexing images via SPARQL query...")

        # Separate PREFIX declarations from the graph pattern body
        lines = sparql_pattern.strip().split('\n')
        prefixes = []
        pattern_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith('PREFIX'):
                prefixes.append(stripped)
            elif stripped:
                pattern_lines.append(line)

        prefix_block = '\n'.join(prefixes)
        pattern_block = '\n'.join(pattern_lines)

        query = f"""
        {prefix_block}
        SELECT ?entity ?url WHERE {{
            {pattern_block}
        }}
        """

        result: Dict[str, List[str]] = {}
        try:
            rows = self.batch_query_tsv(query)
            for row in rows:
                if len(row) >= 2:
                    entity_uri, url = row[0], row[1]
                    if url.startswith("http"):
                        result.setdefault(entity_uri, []).append(url)

            entity_count = len(result)
            image_count = sum(len(v) for v in result.values())
            logger.info(f"Found {image_count} images for {entity_count} entities")

        except Exception as e:
            logger.error(f"Error executing image SPARQL query: {e}")
            logger.debug(f"Query was: {query}")

        return result
