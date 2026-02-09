"""
Fundamental Relationship (FR) path-guided traversal for CIDOC-CRM entity documents.

Based on Tzompanaki & Doerr (2012), this module walks curated multi-step property
paths between fundamental categories (Thing, Actor, Place, Event, Concept, Time)
to produce identity-focused entity documents for FAISS embedding.

Used by both bulk_generate_documents.py (offline) and universal_rag_system.py (runtime).
"""

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# CRM class local names that should produce minimal documents (no FR traversal).
# These are vocabulary/classification entities, not real-world things.
MINIMAL_DOC_CLASSES = {
    "E55_Type", "E56_Language", "E57_Material", "E58_Measurement_Unit",
    "E98_Currency", "E30_Right", "E41_Appellation", "E42_Identifier",
    "E35_Title", "E54_Dimension", "E97_Monetary_Amount", "E52_Time-Span",
}

# Satellite classification subsets
_APPELLATION_CLASSES = {"E41_Appellation", "E42_Identifier", "E35_Title"}
_TYPE_CLASSES = {"E55_Type", "E56_Language", "E57_Material", "E58_Measurement_Unit", "E98_Currency"}
_DIMENSION_CLASSES = {"E54_Dimension", "E97_Monetary_Amount"}
_TIME_CLASSES = {"E52_Time-Span"}


def classify_satellite(entity_types: Set[str]) -> str:
    """Classify a satellite entity's kind from its type URIs.

    Returns one of: 'appellation', 'type', 'dimension', 'time', 'other'.
    """
    for type_uri in entity_types:
        local = type_uri.split('/')[-1].split('#')[-1]
        if local in _APPELLATION_CLASSES:
            return "appellation"
        if local in _TYPE_CLASSES:
            return "type"
        if local in _DIMENSION_CLASSES:
            return "dimension"
        if local in _TIME_CLASSES:
            return "time"
    return "other"


# FR labels where targets should include attributes (depiction predicates)
_DEPICTION_LABELS = {"denotes", "portray", "is target of portray",
                     "is denoted by", "is target of is composed of"}

# Predicate local names that give descriptive types (P2_has_type and inverse)
_TYPE_PRED_LOCALS = {"P2_has_type", "P2i_is_type_of"}

# Predicate local names that give iconographic attributes
_ATTR_PRED_LOCALS = {"K14_has_attribute", "P3_has_note"}


def build_target_enrichments(
    target_uris: Set[str],
    outgoing: Dict[str, list],
    entity_labels: Dict[str, str],
    entity_types_map: Dict[str, Set[str]] = None,
    class_labels: Dict[str, str] = None,
) -> Dict[str, Dict]:
    """Build type tags and attribute summaries for FR/direct-predicate target entities.

    For each target URI, looks up:
    - P2_has_type → descriptive type tag (e.g. "Donor", "Founder", "Church")
    - K14_has_attribute → iconographic attributes (e.g. "cross of martyrdom")
    - Falls back to CRM class label if no P2_has_type

    Returns:
        Dict: target_uri → {"type_tag": str|None, "attributes": [str]}
    """
    enrichments = {}

    for uri in target_uris:
        type_tag = None
        attributes = []

        for pred, obj in outgoing.get(uri, []):
            local_pred = pred.split('/')[-1].split('#')[-1]

            if local_pred in _TYPE_PRED_LOCALS and not type_tag:
                type_tag = entity_labels.get(obj, obj.split('/')[-1])

            if local_pred in _ATTR_PRED_LOCALS:
                attr_label = entity_labels.get(obj, obj.split('/')[-1])
                if attr_label:
                    attributes.append(attr_label)

        # Fallback: CRM class label
        if not type_tag and entity_types_map and class_labels:
            types = entity_types_map.get(uri, set())
            for type_uri in types:
                label = class_labels.get(type_uri)
                if label:
                    local = type_uri.split('/')[-1].split('#')[-1]
                    if local not in ("E1_CRM_Entity", "E77_Persistent_Item",
                                     "E71_Human-Made_Thing"):
                        type_tag = label
                        break

        if type_tag or attributes:
            enrichments[uri] = {
                "type_tag": type_tag,
                "attributes": attributes[:5],
            }

    return enrichments


class FRTraversal:
    """Fundamental Relationship path matcher for CIDOC-CRM entity documents."""

    def __init__(self, fr_json_path: str, inverse_properties_path: str,
                 fc_mapping_path: str, property_labels: dict = None):
        """
        Args:
            fr_json_path: Path to fundamental_relationships_cidoc_crm.json
            inverse_properties_path: Path to data/labels/inverse_properties.json
            fc_mapping_path: Path to config/fc_class_mapping.json
            property_labels: Optional dict of predicate URI/local-name -> English label
        """
        self.property_labels = property_labels or {}

        # Load FR definitions
        with open(fr_json_path, 'r', encoding='utf-8') as f:
            fr_data = json.load(f)
        self.fr_list = fr_data["fundamental_relationships"]

        # Load inverse properties (full URI -> full URI, bidirectional)
        with open(inverse_properties_path, 'r', encoding='utf-8') as f:
            self._inverse_full = json.load(f)

        # Build local-name inverse lookup for fast matching
        self._inverse_local = {}
        for uri, inv_uri in self._inverse_full.items():
            local = self._local_name(uri)
            inv_local = self._local_name(inv_uri)
            self._inverse_local[local] = inv_local
            self._inverse_local[inv_local] = local

        # Load FC class mapping
        with open(fc_mapping_path, 'r', encoding='utf-8') as f:
            fc_raw = json.load(f)
        # Build class_local_name -> FC lookup
        self._class_to_fc: Dict[str, str] = {}
        for fc_name, class_list in fc_raw.items():
            if fc_name.startswith("_"):
                continue
            for cls in class_list:
                self._class_to_fc[cls] = fc_name

        # Index: property_local_name -> list of (fr_index, path_index, step_index)
        self._prop_index = self._build_property_index()

        # Index: (fc, property_local_name) -> True for step-0 properties per domain FC
        # Used by collect_direct_predicates to avoid duplication with FR results
        self._step0_props = self._build_step0_index()

        logger.info(f"FRTraversal loaded: {len(self.fr_list)} FRs, "
                    f"{len(self._inverse_local)} inverse props, "
                    f"{len(self._class_to_fc)} class->FC mappings")

    @staticmethod
    def _local_name(uri: str) -> str:
        """Extract local name from a full URI or return as-is if already local."""
        if '/' in uri:
            name = uri.rsplit('/', 1)[-1]
        else:
            name = uri
        if '#' in name:
            name = name.rsplit('#', 1)[-1]
        return name

    def _build_property_index(self) -> Dict[str, List[Tuple[int, int, int]]]:
        """Index property local names used in FR paths for fast lookup."""
        index = defaultdict(list)
        for fr_idx, fr in enumerate(self.fr_list):
            for path_idx, path in enumerate(fr["paths"]):
                for step_idx, step in enumerate(path["steps"]):
                    prop_local = self._local_name(step["property"])
                    index[prop_local].append((fr_idx, path_idx, step_idx))
        return dict(index)

    def _build_step0_index(self) -> Set[Tuple[str, str]]:
        """Build set of (domain_fc, property_local_name) for step-0 properties.

        Only step-0 properties of a path can fire directly from the entity.
        Properties at step 1+ only fire from intermediate nodes reached by earlier steps.
        """
        step0 = set()
        for fr in self.fr_list:
            domain_fc = fr["domain_fc"]
            for path in fr["paths"]:
                if path["steps"]:
                    prop_local = self._local_name(path["steps"][0]["property"])
                    step0.add((domain_fc, prop_local))
                    # Also add the inverse
                    inv = self._inverse_local.get(prop_local)
                    if inv:
                        step0.add((domain_fc, inv))
        return step0

    def get_fc(self, entity_types: Set[str]) -> Optional[str]:
        """Determine the fundamental category for an entity given its rdf:type URIs.

        Returns the most specific FC found. Priority: Actor > Event > Place > Concept > Time > Thing.
        This ensures E21_Person (Actor) takes priority over E77_Persistent_Item (Thing).
        """
        fc_priority = {"Actor": 6, "Event": 5, "Place": 4, "Concept": 3, "Time": 2, "Thing": 1}
        best_fc = None
        best_priority = 0

        for type_uri in entity_types:
            local = self._local_name(type_uri)
            fc = self._class_to_fc.get(local)
            if fc and fc_priority.get(fc, 0) > best_priority:
                best_fc = fc
                best_priority = fc_priority[fc]

        return best_fc

    def is_minimal_doc_entity(self, entity_types: Set[str]) -> bool:
        """Check if entity should get a minimal document (no FR traversal)."""
        for type_uri in entity_types:
            local = self._local_name(type_uri)
            if local in MINIMAL_DOC_CLASSES:
                return True
        return False

    def match_fr_paths(self, entity_uri: str, entity_types: Set[str],
                       outgoing: Dict[str, List[Tuple[str, str]]],
                       incoming: Dict[str, List[Tuple[str, str]]],
                       entity_labels: Dict[str, str],
                       entity_types_map: Dict[str, Set[str]] = None,
                       max_results_per_fr: int = 5) -> List[dict]:
        """Match FR paths from an entity and return reached targets.

        Args:
            entity_uri: The entity to traverse from
            entity_types: rdf:type URIs of the entity
            outgoing: uri -> [(pred_uri, object_uri), ...] for the full graph
            incoming: uri -> [(pred_uri, subject_uri), ...] for the full graph
            entity_labels: uri -> label for the full graph
            entity_types_map: uri -> set of type URIs (for FC checking of targets)
            max_results_per_fr: Max target entities per FR to prevent explosion

        Returns:
            List of dicts: [{"fr_id": ..., "fr_label": ..., "targets": [(uri, label), ...]}]
        """
        entity_fc = self.get_fc(entity_types)
        if not entity_fc:
            return []

        results = []
        seen_targets = {}  # fr_id -> set of target URIs (dedup across paths of same FR)

        for fr in self.fr_list:
            if fr["domain_fc"] != entity_fc:
                continue

            fr_id = fr["id"]
            fr_label = fr["label"]

            if fr_id not in seen_targets:
                seen_targets[fr_id] = set()

            for path in fr["paths"]:
                # Walk this path starting from entity_uri
                current_nodes = {entity_uri}

                for step in path["steps"]:
                    prop_local = self._local_name(step["property"])
                    inv_local = self._inverse_local.get(prop_local)
                    is_recursive = step.get("recursive", False)

                    next_nodes = set()

                    if is_recursive:
                        # BFS until no new nodes
                        frontier = set(current_nodes)
                        visited_rec = set(current_nodes)
                        while frontier:
                            new_frontier = set()
                            for node in frontier:
                                targets = self._follow_property(
                                    node, prop_local, inv_local, outgoing, incoming
                                )
                                for t in targets:
                                    if t not in visited_rec:
                                        visited_rec.add(t)
                                        new_frontier.add(t)
                            frontier = new_frontier
                        # All reachable nodes minus the starting set
                        next_nodes = visited_rec - current_nodes
                    else:
                        for node in current_nodes:
                            targets = self._follow_property(
                                node, prop_local, inv_local, outgoing, incoming
                            )
                            next_nodes.update(targets)

                    # Remove the origin entity from targets to prevent self-reference
                    next_nodes.discard(entity_uri)

                    if not next_nodes:
                        break
                    current_nodes = next_nodes
                else:
                    # Path completed successfully - current_nodes are the targets
                    # Optionally filter by range FC if entity_types_map provided
                    range_fc = fr.get("range_fc")
                    valid_targets = set()
                    for t in current_nodes:
                        if entity_types_map and range_fc:
                            t_types = entity_types_map.get(t, set())
                            t_fc = self.get_fc(t_types)
                            if t_fc and t_fc != range_fc:
                                continue
                        valid_targets.add(t)

                    new_targets = valid_targets - seen_targets[fr_id]
                    seen_targets[fr_id].update(new_targets)

            # After processing all paths for this FR, collect results
            all_targets = seen_targets.get(fr_id, set())
            if all_targets:
                # Limit targets per FR
                target_list = []
                for t in list(all_targets)[:max_results_per_fr]:
                    label = entity_labels.get(t, self._local_name(t))
                    target_list.append((t, label))

                results.append({
                    "fr_id": fr_id,
                    "fr_label": fr_label,
                    "targets": target_list
                })

        return results

    def _follow_property(self, node: str, prop_local: str, inv_local: Optional[str],
                         outgoing: Dict[str, List[Tuple[str, str]]],
                         incoming: Dict[str, List[Tuple[str, str]]]) -> Set[str]:
        """Follow a property from a node, checking both outgoing and inverse incoming."""
        targets = set()

        # Check outgoing edges for property match
        for pred, obj in outgoing.get(node, []):
            pred_local = self._local_name(pred)
            if pred_local == prop_local:
                targets.add(obj)

        # Check incoming edges for inverse property match
        # If the step property is P14_carried_out_by, and the inverse is P14i_performed,
        # then we look for incoming edges where pred_local == P14_carried_out_by
        # (since incoming stores (pred, subj) where the triple is subj --pred--> node)
        # So an incoming (P14_carried_out_by, subj) means subj --P14_carried_out_by--> node
        # We want: node --P14i_performed--> subj, which is equivalent
        if inv_local:
            for pred, subj in incoming.get(node, []):
                pred_local = self._local_name(pred)
                if pred_local == inv_local:
                    targets.add(subj)

        return targets

    def collect_direct_predicates(self, entity_uri: str,
                                  outgoing: Dict[str, List[Tuple[str, str]]],
                                  incoming: Dict[str, List[Tuple[str, str]]],
                                  entity_labels: Dict[str, str],
                                  entity_types: Set[str] = None,
                                  schema_filter=None,
                                  max_per_predicate: int = 5) -> List[dict]:
        """Collect direct (1-hop) non-FR, non-schema predicates for VIR extensions etc.

        Only filters predicates that appear as step-0 of FR paths for this entity's
        FC. This ensures predicates like P108i_was_produced_by (which appears at step 1+
        in FR paths but not step 0 for Thing) are still included.

        Args:
            entity_uri: The entity
            outgoing: Full graph outgoing index
            incoming: Full graph incoming index
            entity_labels: Full graph label index
            entity_types: rdf:type URIs of the entity (for FC determination)
            schema_filter: Optional callable(pred_uri) -> bool for schema filtering
            max_per_predicate: Max targets per predicate

        Returns:
            List of dicts: [{"predicate_label": ..., "targets": [(uri, label), ...]}]
        """
        entity_fc = self.get_fc(entity_types) if entity_types else None
        results = defaultdict(set)  # pred_label -> set of (uri, label)

        for pred, obj in outgoing.get(entity_uri, []):
            if schema_filter and schema_filter(pred):
                continue
            pred_local = self._local_name(pred)
            # Only skip if this property is a step-0 FR property for this entity's FC
            if entity_fc and (entity_fc, pred_local) in self._step0_props:
                continue
            # Skip rdf:type — handled separately
            if 'rdf-syntax-ns#type' in pred or pred_local == 'type':
                continue
            label = self._get_predicate_label(pred, pred_local)
            obj_label = entity_labels.get(obj, self._local_name(obj))
            results[label].add((obj, obj_label))

        for pred, subj in incoming.get(entity_uri, []):
            if schema_filter and schema_filter(pred):
                continue
            pred_local = self._local_name(pred)
            inv_local = self._inverse_local.get(pred_local)
            # For incoming, the entity sees the inverse. Check if that inverse is step-0.
            check_local = inv_local or pred_local
            if entity_fc and (entity_fc, check_local) in self._step0_props:
                continue
            if 'rdf-syntax-ns#type' in pred or pred_local == 'type':
                continue
            # For incoming, use inverse label if available
            if inv_local:
                inv_label = self._get_predicate_label(None, inv_local)
            else:
                inv_label = f"is target of {self._get_predicate_label(pred, pred_local)}"
            subj_label = entity_labels.get(subj, self._local_name(subj))
            results[inv_label].add((subj, subj_label))

        # Convert to list, cap per predicate
        output = []
        for pred_label, target_set in results.items():
            targets = list(target_set)[:max_per_predicate]
            output.append({
                "predicate_label": pred_label,
                "targets": targets
            })
        return output

    def _get_predicate_label(self, full_uri: Optional[str], local_name: str) -> str:
        """Get human-readable label for a predicate."""
        if full_uri and self.property_labels:
            label = self.property_labels.get(full_uri)
            if label:
                return label
        if self.property_labels:
            label = self.property_labels.get(local_name)
            if label:
                return label
        # Fallback: strip CRM prefix (e.g. "P14_carried_out_by" -> "carried out by")
        stripped = re.sub(r'^[A-Z]\d+[a-z]?_', '', local_name)
        return stripped.replace('_', ' ')

    def format_absorbed_satellites(self, satellite_info: Dict[str, List[str]],
                                   parent_label: str) -> List[str]:
        """Format absorbed satellite info as compact lines for a parent document.

        Args:
            satellite_info: kind -> [label, ...] dict from _identify_satellites
            parent_label: The parent entity's label (to deduplicate)

        Returns:
            List of formatted lines (0-5 typically)
        """
        lines = []
        parent_lower = parent_label.lower().strip() if parent_label else ""

        # Appellations -> "Also known as: ..."
        app_labels = satellite_info.get("appellation", [])
        # Deduplicate against parent label
        unique_apps = []
        seen = {parent_lower}
        for a in app_labels:
            a_lower = a.lower().strip()
            if a_lower not in seen:
                seen.add(a_lower)
                unique_apps.append(a)
        if unique_apps:
            lines.append(f"Also known as: {', '.join(unique_apps[:5])}")

        # Dimensions -> "Dimensions: ..."
        dim_labels = satellite_info.get("dimension", [])
        if dim_labels:
            lines.append(f"Dimensions: {', '.join(dim_labels[:5])}")

        # Time-spans -> "Time-span: ..."
        time_labels = satellite_info.get("time", [])
        if time_labels:
            lines.append(f"Time-span: {', '.join(time_labels[:5])}")

        # Types are intentionally omitted — already captured by "Has type:" from FR traversal

        return lines

    def format_fr_document(self, entity_uri: str, label: str,
                           types_display: List[str], literals: dict,
                           fr_results: List[dict],
                           direct_predicates: List[dict] = None,
                           fc: str = None,
                           absorbed_lines: List[str] = None,
                           target_enrichments: Dict[str, Dict] = None) -> str:
        """Format an FR-organized entity document for FAISS embedding.

        Args:
            entity_uri: Entity URI
            label: Entity label
            types_display: Human-readable type names (from rdf:type class labels)
            literals: prop_name -> [values] dict
            fr_results: Output from match_fr_paths()
            direct_predicates: Output from collect_direct_predicates()
            fc: Fundamental category string (Thing, Actor, etc.)
            absorbed_lines: Lines from absorbed satellite entities
            target_enrichments: Output from build_target_enrichments() —
                maps target URI → {"type_tag": str, "attributes": [str]}

        Returns:
            Document text (no frontmatter — that's handled by save_document)
        """
        lines = []

        # Line 1: [FC/Type] Label — puts type at highest embedding weight position
        type_tag = fc or (types_display[0] if types_display else "Entity")
        lines.append(f"[{type_tag}] {label}")
        lines.append("")

        # Literal properties (2-5 lines)
        literal_keys_priority = [
            'label', 'preflabel', 'altlabel', 'name', 'title',
            'description', 'note', 'comment',
            'value', 'date', 'begin_of_the_begin', 'end_of_the_end'
        ]
        added_literals = set()
        for key in literal_keys_priority:
            for prop_name, values in literals.items():
                if prop_name.lower() == key.lower() and prop_name not in added_literals:
                    display_name = prop_name.replace('_', ' ').title()
                    val_str = "; ".join(v[:200] for v in values[:3])
                    lines.append(f"{display_name}: {val_str}")
                    added_literals.add(prop_name)

        # Add remaining literals not in priority list (up to a few more)
        remaining_count = 0
        for prop_name, values in sorted(literals.items()):
            if prop_name in added_literals:
                continue
            if remaining_count >= 3:
                break
            display_name = prop_name.replace('_', ' ').title()
            val_str = "; ".join(v[:200] for v in values[:3])
            lines.append(f"{display_name}: {val_str}")
            added_literals.add(prop_name)
            remaining_count += 1

        if added_literals:
            lines.append("")

        # Absorbed satellite info (names, dimensions, dates from satellite entities)
        if absorbed_lines:
            lines.extend(absorbed_lines)
            lines.append("")

        # FR results (one line per FR, ~20 lines max)
        fr_line_count = 0
        for fr_result in fr_results:
            if fr_line_count >= 20:
                break
            fr_label = fr_result["fr_label"]
            fr_label_cap = fr_label[0].upper() + fr_label[1:]
            include_attrs = fr_label.lower() in _DEPICTION_LABELS
            formatted = self._format_targets(
                fr_result["targets"], target_enrichments, include_attrs)
            lines.append(f"{fr_label_cap}: {formatted}")
            fr_line_count += 1

        # Direct (non-FR) predicates — VIR extensions etc.
        if direct_predicates:
            for dp in direct_predicates:
                if fr_line_count >= 25:
                    break
                pred_label = dp["predicate_label"]
                pred_cap = pred_label[0].upper() + pred_label[1:]
                include_attrs = pred_label.lower() in _DEPICTION_LABELS
                formatted = self._format_targets(
                    dp["targets"], target_enrichments, include_attrs)
                lines.append(f"{pred_cap}: {formatted}")
                fr_line_count += 1

        return "\n".join(lines)

    def _format_targets(self, targets: List[Tuple[str, str]],
                        target_enrichments: Dict[str, Dict] = None,
                        include_attrs: bool = False) -> str:
        """Format a list of (uri, label) targets with optional type tags and attributes.

        Args:
            targets: [(uri, label), ...]
            target_enrichments: uri -> {"type_tag": str|None, "attributes": [str]}
            include_attrs: If True, include attributes for depiction-like predicates

        Returns:
            Comma-separated string like "Saint Anastasia (Saint, cross of martyrdom, bottle of medicine)"
        """
        parts = []
        for uri, label in targets:
            enr = (target_enrichments or {}).get(uri)
            if enr:
                tag = enr.get("type_tag")
                attrs = enr.get("attributes", []) if include_attrs else []
                if tag and attrs:
                    annotation = ", ".join([tag] + attrs)
                    parts.append(f"{label} ({annotation})")
                elif tag:
                    parts.append(f"{label} ({tag})")
                elif attrs:
                    parts.append(f"{label} ({', '.join(attrs)})")
                else:
                    parts.append(label)
            else:
                parts.append(label)
        return ", ".join(parts)

    def format_minimal_document(self, entity_uri: str, label: str,
                                types_display: List[str], literals: dict) -> str:
        """Format a minimal document for E55_Type / E30_Right etc.

        Just label, type, and literal properties — 2-5 lines, no FR traversal.
        """
        lines = []

        type_tag = types_display[0] if types_display else "Concept"
        lines.append(f"[{type_tag}] {label}")
        lines.append("")

        for prop_name, values in sorted(literals.items()):
            display_name = prop_name.replace('_', ' ').title()
            val_str = "; ".join(v[:200] for v in values[:3])
            lines.append(f"{display_name}: {val_str}")

        return "\n".join(lines)
