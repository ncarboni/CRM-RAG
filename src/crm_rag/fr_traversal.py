"""
Fundamental Relationship (FR) formatting and classification for CIDOC-CRM entity documents.

Based on Tzompanaki & Doerr (2012).  FR path traversal is now handled by
fr_materializer.py (igraph-native walker).  This module provides:
  - FC classification (get_fc)
  - Satellite identification (is_minimal_doc_entity, classify_satellite)
  - Document formatting (format_fr_document, format_minimal_document, etc.)
  - Target enrichments (build_target_enrichments)
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
    "E41_E33_Linguistic_Appellation", "E33_E41_Linguistic_Appellation",
    "PC67_refers_to",
}

# Satellite classification subsets
_APPELLATION_CLASSES = {"E41_Appellation", "E42_Identifier", "E35_Title",
                        "E41_E33_Linguistic_Appellation", "E33_E41_Linguistic_Appellation"}
_TYPE_CLASSES = {"E55_Type", "E56_Language", "E57_Material", "E58_Measurement_Unit", "E98_Currency"}
_DIMENSION_CLASSES = {"E54_Dimension", "E97_Monetary_Amount"}
_TIME_CLASSES = {"E52_Time-Span"}
_REFERENCE_CLASSES = {"PC67_refers_to"}


def classify_satellite(entity_types: Set[str]) -> str:
    """Classify a satellite entity's kind from its type URIs.

    Returns one of: 'appellation', 'type', 'dimension', 'time', 'reference', 'other'.
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
        if local in _REFERENCE_CLASSES:
            return "reference"
    return "other"


# High-cardinality summary threshold: above this, show count + examples
SUMMARY_THRESHOLD = 5

# Noise detection patterns for FR target labels
_RE_PURE_NUMERIC = re.compile(r'^\d+$')
_RE_UUID = re.compile(r'^[0-9a-fA-F-]{32,}')

# FR labels where targets should include attributes (depiction predicates)
_DEPICTION_LABELS = {"denotes", "portray", "is target of portray",
                     "is denoted by", "is target of is composed of"}

# Predicate local names that give descriptive types (P2_has_type and inverse)
_TYPE_PRED_LOCALS = {"P2_has_type", "P2i_is_type_of"}

# Predicate local names that give iconographic attributes
_ATTR_PRED_LOCALS = {"K14_has_attribute", "P3_has_note"}

# Predicate local names that give venue/place info
_VENUE_PRED_LOCALS = {"P7_took_place_at", "P7i_witnessed"}


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
        venue = None

        for pred, obj in outgoing.get(uri, []):
            local_pred = pred.split('/')[-1].split('#')[-1]

            if local_pred in _TYPE_PRED_LOCALS and not type_tag:
                type_tag = entity_labels.get(obj, obj.split('/')[-1])

            if local_pred in _ATTR_PRED_LOCALS:
                attr_label = entity_labels.get(obj, obj.split('/')[-1])
                if attr_label:
                    attributes.append(attr_label)

            if local_pred in _VENUE_PRED_LOCALS and not venue:
                venue = entity_labels.get(obj, obj.split('/')[-1])

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

        if type_tag or attributes or venue:
            enrichments[uri] = {
                "type_tag": type_tag,
                "attributes": attributes[:5],
                "venue": venue,
            }

    return enrichments


class FRTraversal:
    """FR classification, formatting, and satellite handling for CIDOC-CRM documents.

    FR path traversal is handled by fr_materializer.py.  This class provides
    FC classification, document formatting, and satellite identification.
    """

    def __init__(self, inverse_properties_path: str,
                 fc_mapping_path: str, property_labels: dict = None):
        """
        Args:
            inverse_properties_path: Path to data/labels/inverse_properties.json
            fc_mapping_path: Path to config/fc_class_mapping.json
            property_labels: Optional dict of predicate URI/local-name -> English label
        """
        self.property_labels = property_labels or {}

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

        # Load FR definitions from new module for FR metadata (fr_list for _build_aggregation_context)
        from crm_rag.fundamental_relationships import build_fully_expanded
        expanded_frs = build_fully_expanded()
        self.fr_list = [
            {
                "id": fr.id,
                "label": fr.label,
                "domain_fc": fr.domain_fc,
                "range_fc": fr.range_fc,
                "paths": [{"steps": [{"property": s.property} for s in p.steps]} for p in fr.paths],
            }
            for fr in expanded_frs
        ]

        # Build step0 predicate index from new FR definitions
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

    @staticmethod
    def _format_time_entry(entry) -> str:
        """Format a single time-span satellite entry as a human-readable string.

        Args:
            entry: Either a plain label string (legacy) or a dict with keys
                   "label", "begin", "end", "within" from date literal lookup.

        Returns:
            Formatted date string, e.g. "2023-01-26 to 2023-06-18" or the label.
        """
        if isinstance(entry, str):
            return entry

        # Dict with date values from _identify_satellites_from_prefetched
        begin = entry.get("begin")
        end = entry.get("end")
        within = entry.get("within")
        label = entry.get("label", "")

        if begin and end:
            return f"{begin} to {end}"
        if begin:
            return f"from {begin}"
        if end:
            return f"until {end}"
        if within:
            return str(within)
        # No P82/P81 date values found — return empty rather than using the
        # label which is not an authoritative date source.
        return ""

    def format_absorbed_satellites(self, satellite_info: Dict[str, list],
                                   parent_label: str) -> List[str]:
        """Format absorbed satellite info as compact lines for a parent document.

        Args:
            satellite_info: kind -> [label_or_dict, ...] dict from _identify_satellites.
                For time satellites, entries may be dicts with date values.
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

        # Time-spans -> "Time-span: ..." with actual date values when available
        time_entries = satellite_info.get("time", [])
        if time_entries:
            formatted = [s for s in (self._format_time_entry(e) for e in time_entries[:5]) if s]
            if formatted:
                lines.append(f"Time-span: {', '.join(formatted)}")

        # References (PC67_refers_to reification nodes) -> "Referenced in: ..."
        ref_labels = satellite_info.get("reference", [])
        if ref_labels:
            for ref in ref_labels[:5]:
                lines.append(f"Referenced in: {ref}")

        # Types are intentionally omitted — already captured by "Has type:" from FR traversal

        return lines

    @staticmethod
    def _base_label(label: str) -> str:
        """Strip enrichment tag in parentheses to get the base label for dedup."""
        # "digital (300404202)" → "digital"
        idx = label.rfind(" (")
        return label[:idx].strip() if idx > 0 else label.strip()

    @staticmethod
    def _is_noise_label(label: str) -> bool:
        """Check if a target label is noise (numeric ID, UUID)."""
        base = FRTraversal._base_label(label)
        if _RE_PURE_NUMERIC.match(base):
            return True
        if _RE_UUID.match(base):
            return True
        return False

    @staticmethod
    def _is_low_uniqueness(targets: List[Tuple[str, str]], total_count: int) -> bool:
        """Check if an FR line has low-uniqueness generic labels.

        Returns True when ≤2 unique base labels AND total_count > SUMMARY_THRESHOLD.
        E.g. "creation" repeated 66 times, "digital" repeated 183 times.
        """
        if total_count <= SUMMARY_THRESHOLD:
            return False
        unique_bases = {FRTraversal._base_label(lbl) for _, lbl in targets}
        return len(unique_bases) <= 2

    @staticmethod
    def _dedup_fr_results(fr_results: List[dict]) -> List[dict]:
        """Remove FR results whose target URIs overlap >80% with a larger FR.

        When two FRs share >80% of targets, keep the one with more targets.
        E.g. "is origin of" (66 targets) vs "is generator of" (66 targets)
        — if they share >80%, keep the first one encountered with more targets.
        """
        if len(fr_results) <= 1:
            return fr_results

        # Build target URI sets per FR result
        uri_sets = []
        for fr in fr_results:
            uris = {uri for uri, _lbl in fr["targets"]}
            # Include unseen targets (total_count > shown) — use shown URIs as proxy
            uri_sets.append(uris)

        keep = [True] * len(fr_results)
        for i in range(len(fr_results)):
            if not keep[i]:
                continue
            total_i = fr_results[i].get("total_count", len(uri_sets[i]))
            for j in range(i + 1, len(fr_results)):
                if not keep[j]:
                    continue
                total_j = fr_results[j].get("total_count", len(uri_sets[j]))
                # Check overlap using the shown targets
                if not uri_sets[i] or not uri_sets[j]:
                    continue
                overlap = len(uri_sets[i] & uri_sets[j])
                smaller = min(len(uri_sets[i]), len(uri_sets[j]))
                if smaller > 0 and overlap / smaller > 0.8:
                    # Drop the one with fewer total targets
                    if total_i >= total_j:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        return [fr for fr, k in zip(fr_results, keep) if k]

    def _format_fr_line(self, fr_label: str, targets: List[Tuple[str, str]],
                        total_count: int, target_enrichments: Dict[str, Dict] = None,
                        include_attrs: bool = False,
                        time_span_dates: Dict[str, str] = None) -> Optional[str]:
        """Format a single FR/direct-predicate line, applying noise filter and summary mode.

        Returns None if the line should be skipped (all noise).
        """
        # Noise filter: skip if ALL targets are noise labels
        non_noise = [(uri, lbl) for uri, lbl in targets
                     if not self._is_noise_label(lbl)]
        if not non_noise and targets:
            return None

        # Low-uniqueness filter: skip if ≤2 unique base labels and high cardinality
        if self._is_low_uniqueness(targets, total_count):
            return None

        fr_label_cap = fr_label[0].upper() + fr_label[1:]

        if total_count > SUMMARY_THRESHOLD:
            # Summary mode: count + 3 examples
            examples = self._format_targets(
                targets[:3], target_enrichments, include_attrs, time_span_dates)
            return f"{fr_label_cap}: {total_count} items including {examples}"
        else:
            # List mode: show all
            formatted = self._format_targets(
                targets, target_enrichments, include_attrs, time_span_dates)
            return f"{fr_label_cap}: {formatted}"

    def format_fr_document(self, entity_uri: str, label: str,
                           types_display: List[str], literals: dict,
                           fr_results: List[dict],
                           direct_predicates: List[dict] = None,
                           fc: str = None,
                           absorbed_lines: List[str] = None,
                           target_enrichments: Dict[str, Dict] = None,
                           time_span_dates: Dict[str, str] = None) -> str:
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
            time_span_dates: URI → formatted date string for time-span resolution

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

        # Deduplicate overlapping FR results (>80% shared targets)
        deduped_frs = self._dedup_fr_results(fr_results)

        # FR results + direct predicates (deduplicated by formatted line text)
        seen_lines = set()
        fr_line_count = 0

        for fr_result in deduped_frs:
            if fr_line_count >= 20:
                break
            include_attrs = fr_result["fr_label"].lower() in _DEPICTION_LABELS
            total = fr_result.get("total_count", len(fr_result["targets"]))
            line = self._format_fr_line(
                fr_result["fr_label"], fr_result["targets"], total,
                target_enrichments, include_attrs, time_span_dates)
            if line and line not in seen_lines:
                seen_lines.add(line)
                lines.append(line)
                fr_line_count += 1

        # Direct (non-FR) predicates — VIR extensions etc.
        if direct_predicates:
            for dp in direct_predicates:
                if fr_line_count >= 25:
                    break
                include_attrs = dp["predicate_label"].lower() in _DEPICTION_LABELS
                total = dp.get("total_count", len(dp["targets"]))
                line = self._format_fr_line(
                    dp["predicate_label"], dp["targets"], total,
                    target_enrichments, include_attrs, time_span_dates)
                if line and line not in seen_lines:
                    seen_lines.add(line)
                    lines.append(line)
                    fr_line_count += 1

        return "\n".join(lines)

    def _format_targets(self, targets: List[Tuple[str, str]],
                        target_enrichments: Dict[str, Dict] = None,
                        include_attrs: bool = False,
                        time_span_dates: Dict[str, str] = None) -> str:
        """Format a list of (uri, label) targets with optional type tags, attributes, and dates.

        Disambiguates same-label targets using date, venue, and type tag
        instead of generic " x{N}" counts.

        Args:
            targets: [(uri, label), ...]
            target_enrichments: uri -> {"type_tag": str|None, "attributes": [str], "venue": str|None}
            include_attrs: If True, include attributes for depiction-like predicates
            time_span_dates: uri -> formatted date string for time-span targets

        Returns:
            Comma-separated string like "Saint Anastasia (Saint, cross of martyrdom)"
        """
        time_dates = time_span_dates or {}

        # Build (uri, base_label, formatted_label) for each target
        entries = []  # [(uri, base_label, formatted_label)]
        for uri, label in targets:
            date_str = time_dates.get(uri)
            if date_str:
                enr = (target_enrichments or {}).get(uri)
                tag = enr.get("type_tag") if enr else None
                if tag and tag != "Time-Span":
                    entries.append((uri, label, f"{label} ({date_str})"))
                else:
                    entries.append((uri, label, date_str))
                continue

            enr = (target_enrichments or {}).get(uri)
            if enr:
                tag = enr.get("type_tag")
                attrs = enr.get("attributes", []) if include_attrs else []
                if tag and attrs:
                    annotation = ", ".join([tag] + attrs)
                    entries.append((uri, label, f"{label} ({annotation})"))
                elif tag:
                    entries.append((uri, label, f"{label} ({tag})"))
                elif attrs:
                    entries.append((uri, label, f"{label} ({', '.join(attrs)})"))
                else:
                    entries.append((uri, label, label))
            else:
                entries.append((uri, label, label))

        # Group by formatted label to find collisions
        from collections import defaultdict
        label_groups = defaultdict(list)
        for uri, base_label, fmt_label in entries:
            label_groups[fmt_label].append(uri)

        # Build final parts, disambiguating same-label groups
        parts = []
        seen_uris = set()
        for uri, base_label, fmt_label in entries:
            if uri in seen_uris:
                continue
            seen_uris.add(uri)

            if len(label_groups[fmt_label]) <= 1:
                # Unique label — keep as-is
                parts.append(fmt_label)
            else:
                # Collision — rebuild with disambiguation
                date_str = time_dates.get(uri, "")
                enr = (target_enrichments or {}).get(uri)
                venue = enr.get("venue", "") if enr else ""

                disambig = []
                enr_tag = enr.get("type_tag") if enr else None
                if enr_tag and enr_tag != "Time-Span":
                    disambig.append(enr_tag)
                if date_str:
                    disambig.append(date_str)
                if venue:
                    disambig.append(venue)

                if disambig:
                    parts.append(f"{base_label} ({', '.join(disambig)})")
                else:
                    # URI suffix fallback
                    suffix = uri.rsplit('/', 1)[-1][:30]
                    parts.append(f"{base_label} [{suffix}]")

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
