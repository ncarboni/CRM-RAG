"""
CIDOC-CRM Fundamental Relationships (TR-429) -- Consolidated Module

Source: Tzompanaki & Doerr, "Fundamental Categories and Relationships for
intuitive querying CIDOC-CRM based repositories", FORTH-ICS TR-429, April 2012.

This single file contains:
  1. Core dataclasses (Step, PropertyPath, FundamentalRelationship)
  2. All 98 FR definitions with compact (representative) paths
  3. Combinatorial expansion engine producing fully expanded paths
  4. SPARQL generation grouped by Fundamental Category pair

Usage:
    from fundamental_relationships import build_fully_expanded, generate_sparql_by_entity

    ALL_FRS_EXPANDED = build_fully_expanded()   # 98 FRs, 2325 paths
    sparql_text = generate_sparql_by_entity()    # grouped by FC pair

Cross-referenced with CIDOC-CRM v7.1.3 RDF for domain/range accuracy.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Step:
    property: str
    domain: str
    range: str
    recursive: bool = False

    def to_sparql(self, prefix="crm"):
        prop = f"{prefix}:{self.property}"
        if self.recursive:
            prop = f"{prop}*"
        return prop


@dataclass
class PropertyPath:
    id: str
    description: str
    steps: list[Step]

    def to_sparql_path(self, prefix="crm"):
        return "/".join(s.to_sparql(prefix) for s in self.steps)

    def to_sparql_pattern(self, subject="?s", object_="?o", prefix="crm"):
        return f"{subject} {self.to_sparql_path(prefix)} {object_} ."


@dataclass
class FundamentalRelationship:
    id: str
    label: str
    domain_fc: str
    range_fc: str
    domain_class: str
    range_class: str
    paths: list[PropertyPath]
    specialization_of: Optional[str] = None

    def to_sparql(self, subject="?s", object_="?o", prefix="crm"):
        blocks = []
        for p in self.paths:
            blocks.append(f"  {{ {p.to_sparql_pattern(subject, object_, prefix)} }}")
        union = "\n  UNION\n".join(blocks)
        return (
            f"# FR: {self.id} -- {self.label} ({self.domain_fc} -> {self.range_fc})\n"
            f"SELECT DISTINCT {subject} {object_} WHERE {{\n"
            f"{union}\n"
            f"}}"
        )


def S(prop, dom, rng, rec=False):
    """Shorthand for Step."""
    return Step(prop, dom, rng, rec)


def P(id_, desc, *steps):
    """Shorthand for PropertyPath."""
    return PropertyPath(id_, desc, list(steps))

FR = FundamentalRelationship

# ==========================================================================
# COMPACT FR DEFINITIONS (98 FRs, 212 representative paths)
# ==========================================================================

# =============================================================================
# THING-PLACE
# =============================================================================

thing_place_a = FR(
    "thing_place_a", "refers to or is about", "Thing", "Place",
    "C1.Object", "E53_Place",
    [
        P("tp_a_01", "depicts Place",
          S("P62_depicts", "E24", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_02", "refers to Place",
          S("P67_refers_to", "E89", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_03", "depicts feature at Place",
          S("P62_depicts", "E24", "E26_Physical_Feature"),
          S("P53_has_former_or_current_location", "E26", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_04", "carries -> refers to Place",
          S("P128_carries", "E18", "E90"),
          S("P67_refers_to", "E89", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_05", "P130* -> depicts Place",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P62_depicts", "E24", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_06", "P130* -> refers to Place",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P67_refers_to", "E89", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_07", "P130* -> carries -> refers to Place",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P128_carries", "E18", "E90"),
          S("P67_refers_to", "E89", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_08", "P130i* -> depicts Place",
          S("P130i_features_are_also_found_on", "E70", "E70", rec=True),
          S("P62_depicts", "E24", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_09", "P130i* -> refers to Place",
          S("P130i_features_are_also_found_on", "E70", "E70", rec=True),
          S("P67_refers_to", "E89", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_10", "P46* -> depicts Place",
          S("P46_is_composed_of", "E18", "E18", rec=True),
          S("P62_depicts", "E24", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_11", "P46* -> carries -> refers to Place",
          S("P46_is_composed_of", "E18", "E18", rec=True),
          S("P128_carries", "E18", "E90"),
          S("P67_refers_to", "E89", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_12", "P106* -> refers to Place",
          S("P106_is_composed_of", "E90", "E90", rec=True),
          S("P67_refers_to", "E89", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_13", "P148* -> refers to Place",
          S("P148_has_component", "E89", "E89", rec=True),
          S("P67_refers_to", "E89", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
        P("tp_a_14", "CRMdig: derivative -> digitized -> depicts Place",
          S("F1_is_derivative_of", "D1", "D1", rec=True),
          S("L11i_was_output_of", "D1", "D7"),
          S("P9i_forms_part_of", "D7", "D2", rec=True),
          S("L1_digitized", "D2", "E18"),
          S("P62_depicts", "E24", "E53_Place"),
          S("P89i_contains", "E53_Place", "E53_Place", rec=True)),
    ],
)

thing_place_b = FR(
    "thing_place_b", "is referred to at", "Thing", "Place",
    "C1.Object", "E53_Place",
    [
        P("tp_b_01", "is referred to -> created at Place",
          S("P67i_is_referred_to_by", "E1", "E89"),
          S("P94i_was_created_by", "E89", "E65"),
          S("P9i_forms_part_of", "E65", "E5", rec=True),
          S("P7_took_place_at", "E5", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_b_02", "is referred to -> carried by -> location",
          S("P67i_is_referred_to_by", "E1", "E89"),
          S("P128i_is_carried_by", "E90", "E18"),
          S("P53_has_former_or_current_location", "E18", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_b_03", "is depicted by -> location",
          S("P62i_is_depicted_by", "E1", "E24"),
          S("P53_has_former_or_current_location", "E24", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_b_04", "is depicted by -> produced at Place",
          S("P62i_is_depicted_by", "E1", "E24"),
          S("P108i_was_produced_by", "E24", "E12"),
          S("P9i_forms_part_of", "E12", "E5", rec=True),
          S("P7_took_place_at", "E5", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
    ],
)

thing_place_c = FR(
    "thing_place_c", "from (history)", "Thing", "Place",
    "C1.Object", "E53_Place",
    [
        P("tp_c_01", "has location",
          S("P53_has_former_or_current_location", "E18", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_c_02", "has permanent location",
          S("P54_has_current_permanent_location", "E18", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_c_03", "brought into existence at Place",
          S("P92i_was_brought_into_existence_by", "E77", "E63"),
          S("P9i_forms_part_of", "E63", "E5", rec=True),
          S("P7_took_place_at", "E5", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_c_04", "created by actor residing at Place",
          S("P92i_was_brought_into_existence_by", "E77", "E63"),
          S("P9i_forms_part_of", "E63", "E5", rec=True),
          S("P14_carried_out_by", "E7", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
          S("P74_has_current_or_former_residence", "E39", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_c_05", "created by actor born at Place",
          S("P92i_was_brought_into_existence_by", "E77", "E63"),
          S("P9i_forms_part_of", "E63", "E5", rec=True),
          S("P14_carried_out_by", "E7", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
          S("P92i_was_brought_into_existence_by", "E39", "E63"),
          S("P9i_forms_part_of", "E63", "E5", rec=True),
          S("P7_took_place_at", "E5", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_c_06", "moved to Place",
          S("P25i_moved_by", "E19", "E9"),
          S("P26_moved_to", "E9", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_c_07", "moved from Place",
          S("P25i_moved_by", "E19", "E9"),
          S("P27_moved_from", "E9", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_c_08", "found at Place",
          S("P12i_was_present_at", "E18", "E5"),
          S("P9i_forms_part_of", "E5", "E5", rec=True),
          S("P7_took_place_at", "E5", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
        P("tp_c_09", "acquired at Place",
          S("P24i_changed_ownership_through", "E18", "E8"),
          S("P9i_forms_part_of", "E8", "E5", rec=True),
          S("P7_took_place_at", "E5", "E53_Place"),
          S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
    ],
)

# Thing-Place specializations
thing_place_c_created_in = FR(
    "thing_place_c_created_in", "created in", "Thing", "Place",
    "C1.Object", "E53_Place",
    [P("tp_c_ci_01", "brought into existence at Place",
       S("P92i_was_brought_into_existence_by", "E77", "E63"),
       S("P9i_forms_part_of", "E63", "E5", rec=True),
       S("P7_took_place_at", "E5", "E53_Place"),
       S("P89_falls_within", "E53_Place", "E53_Place", rec=True))],
    specialization_of="thing_place_c",
)

thing_place_c_found_acquired = FR(
    "thing_place_c_found_acquired", "found or acquired at", "Thing", "Place",
    "E19_Physical_Object", "E53_Place",
    [P("tp_c_fa_01", "found at Place",
       S("P12i_was_present_at", "E19", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P7_took_place_at", "E5", "E53_Place"),
       S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
     P("tp_c_fa_02", "acquired at Place",
       S("P24i_changed_ownership_through", "E19", "E8"),
       S("P9i_forms_part_of", "E8", "E5", rec=True),
       S("P7_took_place_at", "E5", "E53_Place"),
       S("P89_falls_within", "E53_Place", "E53_Place", rec=True))],
    specialization_of="thing_place_c",
)

thing_place_c_by_person_from = FR(
    "thing_place_c_by_person_from", "created by person from", "Thing", "Place",
    "C1.Object", "E53_Place",
    [P("tp_c_bp_01", "created by actor residing at Place",
       S("P92i_was_brought_into_existence_by", "E77", "E63"),
       S("P9i_forms_part_of", "E63", "E5", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P74_has_current_or_former_residence", "E39", "E53_Place"),
       S("P89_falls_within", "E53_Place", "E53_Place", rec=True)),
     P("tp_c_bp_02", "created by actor born at Place",
       S("P92i_was_brought_into_existence_by", "E77", "E63"),
       S("P9i_forms_part_of", "E63", "E5", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P92i_was_brought_into_existence_by", "E39", "E63"),
       S("P7_took_place_at", "E63", "E53_Place"),
       S("P89_falls_within", "E53_Place", "E53_Place", rec=True))],
    specialization_of="thing_place_c",
)

thing_place_c_located_in = FR(
    "thing_place_c_located_in", "is/was located in", "Thing", "Place",
    "E18_Physical_Thing", "E53_Place",
    [P("tp_c_li_01", "has former or current location",
       S("P53_has_former_or_current_location", "E18", "E53_Place"),
       S("P89_falls_within", "E53_Place", "E53_Place", rec=True))],
    specialization_of="thing_place_c",
)

thing_place_c_moved_from = FR(
    "thing_place_c_moved_from", "moved from", "Thing", "Place",
    "E19_Physical_Thing", "E53_Place",
    [P("tp_c_mf_01", "moved from Place",
       S("P25i_moved_by", "E19", "E9"),
       S("P27_moved_from", "E9", "E53_Place"),
       S("P89_falls_within", "E53_Place", "E53_Place", rec=True))],
    specialization_of="thing_place_c",
)

thing_place_c_moved_to = FR(
    "thing_place_c_moved_to", "moved to", "Thing", "Place",
    "E19_Physical_Thing", "E53_Place",
    [P("tp_c_mt_01", "moved to Place",
       S("P25i_moved_by", "E19", "E9"),
       S("P26_moved_to", "E9", "E53_Place"),
       S("P89_falls_within", "E53_Place", "E53_Place", rec=True))],
    specialization_of="thing_place_c",
)

# =============================================================================
# THING-THING
# =============================================================================

thing_thing_a = FR(
    "thing_thing_a", "has met", "Thing", "Thing",
    "C1.Object", "E70_Thing",
    [P("tt_a_01", "part -> present at -> in presence of thing",
       S("P46i_forms_part_of", "E18", "E18", rec=True),
       S("P12i_was_present_at", "E18", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P12_occurred_in_the_presence_of", "E5", "E70"),
       S("P46_is_composed_of", "E70", "E70", rec=True))],
)

thing_thing_b = FR(
    "thing_thing_b", "refers to or is about", "Thing", "Thing",
    "C1.Object", "C1.Object",
    [
        P("tt_b_01", "P130* -> P46*/P106*/P148* -> depicts Thing",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P46_is_composed_of", "C1", "C1", rec=True),
          S("P62_depicts", "E24", "C1"),
          S("P46_is_composed_of", "C1", "C1", rec=True)),
        P("tt_b_02", "P130* -> P46*/P106*/P148* -> refers to Thing",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P46_is_composed_of", "C1", "C1", rec=True),
          S("P67_refers_to", "E89", "C1"),
          S("P46_is_composed_of", "C1", "C1", rec=True)),
        P("tt_b_03", "P130* -> carries -> refers to Thing",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P46_is_composed_of", "C1", "C1", rec=True),
          S("P128_carries", "E24", "E73"),
          S("P67_refers_to", "E73", "C1"),
          S("P46_is_composed_of", "C1", "C1", rec=True)),
        P("tt_b_04", "CRMdig: derivative -> digitized Thing",
          S("F1_is_derivative_of", "D1", "D1", rec=True),
          S("L11i_was_output_of", "D1", "D7"),
          S("P9i_forms_part_of", "D7", "D2", rec=True),
          S("L1_digitized", "D2", "C1"),
          S("P46_is_composed_of", "C1", "C1", rec=True)),
    ],
)

thing_thing_c = FR(
    "thing_thing_c", "is referred to by", "Thing", "Thing",
    "C1.Object", "C1.Object",
    [
        P("tt_c_01", "is referred to by -> component of",
          S("P46_is_composed_of", "C1", "C1", rec=True),
          S("P67i_is_referred_to_by", "C1", "E89"),
          S("P148i_is_component_of", "E89", "E89", rec=True)),
        P("tt_c_02", "is referred to by -> carried by",
          S("P46_is_composed_of", "C1", "C1", rec=True),
          S("P67i_is_referred_to_by", "C1", "E89"),
          S("P128i_is_carried_by", "E73", "E24"),
          S("P46i_forms_part_of", "E24", "E24", rec=True)),
        P("tt_c_03", "is depicted by",
          S("P46_is_composed_of", "C1", "C1", rec=True),
          S("P62i_is_depicted_by", "C1", "E24"),
          S("P46i_forms_part_of", "E24", "E24", rec=True)),
    ],
)

thing_thing_d = FR(
    "thing_thing_d", "from", "Thing", "Thing",
    "C1.Object", "C1.Object",
    [
        P("tt_d_01", "part of (P46i/P106i/P148i)",
          S("P46i_forms_part_of", "E18", "E18", rec=True)),
        P("tt_d_02", "resulted from transformation",
          S("P123i_resulted_from", "E18", "E81"),
          S("P9i_forms_part_of", "E81", "E81", rec=True),
          S("P124_transformed", "E81", "C1")),
        P("tt_d_03", "modified -> augmented",
          S("P31i_was_modified_by", "E24", "E11"),
          S("P9i_forms_part_of", "E11", "E7", rec=True),
          S("P110_augmented", "E7", "C1")),
        P("tt_d_04", "modified -> diminished",
          S("P31i_was_modified_by", "E24", "E11"),
          S("P9i_forms_part_of", "E11", "E7", rec=True),
          S("P112_diminished", "E7", "E18")),
    ],
)

thing_thing_d_is_part_of = FR(
    "thing_thing_d_is_part_of", "is part of", "Thing", "Thing",
    "C1.Object", "C1.Object",
    [P("tt_d_ip_01", "P46i*/P106i*/P148i*",
       S("P46i_forms_part_of", "E18", "E18", rec=True))],
    specialization_of="thing_thing_d",
)

thing_thing_e = FR(
    "thing_thing_e", "has part", "Thing", "Thing",
    "C1.Object", "C1.Object",
    [
        P("tt_e_01", "P46/P106/P148",
          S("P46_is_composed_of", "E18", "E18", rec=True)),
        P("tt_e_02", "part addition",
          S("P108i_was_produced_by", "E24", "E79"),
          S("P9i_forms_part_of", "E79", "E79", rec=True),
          S("P111_added", "E79", "E18")),
    ],
)

thing_thing_f = FR(
    "thing_thing_f", "is similar or same with", "Thing", "Thing",
    "C1.Object", "C1.Object",
    [
        P("tt_f_01", "P130*",
          S("P130_shows_features_of", "E70", "E70", rec=True)),
        P("tt_f_02", "P130i*",
          S("P130i_features_are_also_found_on", "E70", "E70", rec=True)),
        P("tt_f_03", "same-as (CRMdig)",
          S("L54i_is_same-as", "C1", "D38"),
          S("L54_is_same-as", "D38", "C1")),
    ],
)

# =============================================================================
# THING-ACTOR
# =============================================================================

thing_actor_a = FR(
    "thing_actor_a", "has met", "Thing", "Actor",
    "C1.Object", "E39_Actor",
    [P("ta_a_01", "part -> present at -> actor",
       S("P46i_forms_part_of", "E18", "E18", rec=True),
       S("P12i_was_present_at", "E18", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P12_occurred_in_the_presence_of", "E5", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))],
)

thing_actor_b = FR(
    "thing_actor_b", "is referred to by", "Thing", "Actor",
    "C1.Object", "E39_Actor",
    [
        P("ta_b_01", "is referred to -> created by actor",
          S("P67i_is_referred_to_by", "C1", "E89"),
          S("P94i_was_created_by", "E89", "E65"),
          S("P9i_forms_part_of", "E65", "E65", rec=True),
          S("P14_carried_out_by", "E65", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_b_02", "is referred to -> carried by -> produced by actor",
          S("P67i_is_referred_to_by", "C1", "E89"),
          S("P128i_is_carried_by", "E73", "E24"),
          S("P46i_forms_part_of", "E24", "E24", rec=True),
          S("P108i_was_produced_by", "E24", "E12"),
          S("P9i_forms_part_of", "E12", "E5", rec=True),
          S("P14_carried_out_by", "E7", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_b_03", "is depicted by -> produced by actor",
          S("P62i_is_depicted_by", "C1", "E24"),
          S("P46i_forms_part_of", "E24", "E24", rec=True),
          S("P108i_was_produced_by", "E24", "E12"),
          S("P9i_forms_part_of", "E12", "E5", rec=True),
          S("P14_carried_out_by", "E7", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
    ],
)

thing_actor_c = FR(
    "thing_actor_c", "refers to or is about", "Thing", "Actor",
    "C1.Object", "E39_Actor",
    [
        P("ta_c_01", "P130* -> depicts Actor",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P62_depicts", "E24", "E39"),
          S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
        P("ta_c_02", "P130* -> refers to Actor",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P67_refers_to", "E89", "E39"),
          S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
        P("ta_c_03", "P130* -> carries -> refers to Actor",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P128_carries", "E18", "E73"),
          S("P67_refers_to", "E73", "E39"),
          S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
    ],
)

thing_actor_d = FR(
    "thing_actor_d", "by", "Thing", "Actor",
    "C1.Object", "E39_Actor",
    [
        P("ta_d_01", "brought into existence by actor",
          S("P92i_was_brought_into_existence_by", "C1", "E63"),
          S("P9i_forms_part_of", "E63", "E7", rec=True),
          S("P14_carried_out_by", "E7", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_d_02", "used for by actor",
          S("P16i_was_used_for", "C1", "E7"),
          S("P9i_forms_part_of", "E7", "E7", rec=True),
          S("P14_carried_out_by", "E7", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_d_03", "modified by actor",
          S("P31i_was_modified_by", "E24", "E11"),
          S("P9i_forms_part_of", "E11", "E7", rec=True),
          S("P14_carried_out_by", "E7", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_d_04", "found by actor",
          S("P12i_was_present_at", "E18", "E5"),
          S("P9i_forms_part_of", "E5", "E5", rec=True),
          S("P11_had_participant", "E5", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_d_05", "acquired -> transferred title to actor",
          S("P24i_changed_ownership_through", "E18", "E8"),
          S("P22_transferred_title_to", "E8", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_d_06", "has owner",
          S("P51_has_former_or_current_owner", "E18", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
    ],
)

thing_actor_d_created_by = FR(
    "thing_actor_d_created_by", "created by", "Thing", "Actor",
    "C1.Object", "E39_Actor",
    [P("ta_d_cb_01", "brought into existence by",
       S("P92i_was_brought_into_existence_by", "C1", "E63"),
       S("P9i_forms_part_of", "E63", "E5", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))],
    specialization_of="thing_actor_d",
)

thing_actor_d_used_by = FR(
    "thing_actor_d_used_by", "used by", "Thing", "Actor",
    "C1.Object", "E39_Actor",
    [P("ta_d_ub_01", "used for by",
       S("P16i_was_used_for", "C1", "E7"),
       S("P9i_forms_part_of", "E7", "E5", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))],
    specialization_of="thing_actor_d",
)

thing_actor_d_modified_by = FR(
    "thing_actor_d_modified_by", "modified by", "Thing", "Actor",
    "E24_Physical_Human-Made_Thing", "E39_Actor",
    [P("ta_d_mb_01", "modified by",
       S("P31i_was_modified_by", "E24", "E11"),
       S("P9i_forms_part_of", "E11", "E5", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))],
    specialization_of="thing_actor_d",
)

thing_actor_d_found_acquired_by = FR(
    "thing_actor_d_found_acquired_by", "found or acquired by", "Thing", "Actor",
    "E18_Physical_Thing", "E39_Actor",
    [
        P("ta_d_fa_01", "found by",
          S("P12i_was_present_at", "E18", "E5"),
          S("P9i_forms_part_of", "E5", "E5", rec=True),
          S("P11_had_participant", "E5", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_d_fa_02", "acquired by",
          S("P24i_changed_ownership_through", "E18", "E8"),
          S("P9i_forms_part_of", "E8", "E5", rec=True),
          S("P14_carried_out_by", "E7", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_d_fa_03", "has owner",
          S("P51_has_former_or_current_owner", "E18", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
    ],
    specialization_of="thing_actor_d",
)

thing_actor_e = FR(
    "thing_actor_e", "from (keeper/owner)", "Thing", "Actor",
    "E18_Physical_Thing", "E39_Actor",
    [
        P("ta_e_01", "has keeper",
          S("P49_has_former_or_current_keeper", "E18", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
        P("ta_e_02", "has owner",
          S("P51_has_former_or_current_owner", "E18", "E39"),
          S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
    ],
)

# =============================================================================
# THING-EVENT
# =============================================================================

thing_event_a = FR(
    "thing_event_a", "refers to or is about", "Thing", "Event",
    "C1.Object", "E5_Event",
    [
        P("te_a_01", "P130* -> depicts Event",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P62_depicts", "E24", "E5"),
          S("P9i_forms_part_of", "E5", "E5", rec=True)),
        P("te_a_02", "P130* -> refers to Event",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P67_refers_to", "E89", "E5"),
          S("P9i_forms_part_of", "E5", "E5", rec=True)),
        P("te_a_03", "P130* -> carries -> refers to Event",
          S("P130_shows_features_of", "E70", "E70", rec=True),
          S("P128_carries", "E18", "E73"),
          S("P67_refers_to", "E73", "E5"),
          S("P9i_forms_part_of", "E5", "E5", rec=True)),
        P("te_a_04", "CRMdig: derivative -> digitized -> depicts Event",
          S("F1_is_derivative_of", "D1", "D1", rec=True),
          S("L11i_was_output_of", "D1", "D7"),
          S("P9i_forms_part_of", "D7", "D2", rec=True),
          S("L1_digitized", "D2", "E18"),
          S("P62_depicts", "E24", "E5"),
          S("P9i_forms_part_of", "E5", "E5", rec=True)),
    ],
)

thing_event_b = FR(
    "thing_event_b", "is referred to at", "Thing", "Event",
    "C1.Object", "E5_Event",
    [
        P("te_b_01", "is referred to -> creation event",
          S("P67i_is_referred_to_by", "C1", "E89"),
          S("P94i_was_created_by", "E89", "E65"),
          S("P9i_forms_part_of", "E65", "E5", rec=True)),
        P("te_b_02", "is referred to -> carried by -> production event",
          S("P67i_is_referred_to_by", "C1", "E89"),
          S("P128i_is_carried_by", "E73", "E24"),
          S("P46i_forms_part_of", "E24", "E24", rec=True),
          S("P108i_was_produced_by", "E24", "E12"),
          S("P9i_forms_part_of", "E12", "E5", rec=True)),
        P("te_b_03", "is depicted by -> production event",
          S("P62i_is_depicted_by", "C1", "E24"),
          S("P108i_was_produced_by", "E24", "E12"),
          S("P9i_forms_part_of", "E12", "E5", rec=True)),
    ],
)

thing_event_c = FR(
    "thing_event_c", "has met (from)", "Thing", "Event",
    "C1.Object", "E5_Event",
    [P("te_c_01", "present at Event",
       S("P12i_was_present_at", "C1", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True))],
)

thing_event_c_destroyed_in = FR(
    "thing_event_c_destroyed_in", "destroyed in", "Thing", "Event",
    "C1.Object", "E5_Event",
    [P("te_c_di_01", "taken out of existence by",
       S("P93i_was_taken_out_of_existence_by", "C1", "E64"),
       S("P9i_forms_part_of", "E64", "E5", rec=True))],
    specialization_of="thing_event_c",
)

thing_event_c_created_in = FR(
    "thing_event_c_created_in", "created in", "Thing", "Event",
    "C1.Object", "E5_Event",
    [P("te_c_ci_01", "brought into existence by",
       S("P92i_was_brought_into_existence_by", "C1", "E63"),
       S("P9i_forms_part_of", "E63", "E5", rec=True))],
    specialization_of="thing_event_c",
)

thing_event_c_modified_in = FR(
    "thing_event_c_modified_in", "modified in", "Thing", "Event",
    "E24", "E5_Event",
    [P("te_c_mi_01", "modified by",
       S("P31i_was_modified_by", "E24", "E11"),
       S("P9i_forms_part_of", "E11", "E5", rec=True))],
    specialization_of="thing_event_c",
)

thing_event_c_used_in = FR(
    "thing_event_c_used_in", "used in", "Thing", "Event",
    "C1.Object", "E5_Event",
    [P("te_c_ui_01", "used for",
       S("P16i_was_used_for", "C1", "E7"),
       S("P9i_forms_part_of", "E7", "E5", rec=True))],
    specialization_of="thing_event_c",
)

thing_event_c_digitized_in = FR(
    "thing_event_c_digitized_in", "digitized in", "Thing", "Event",
    "C1.Object", "E5_Event",
    [P("te_c_dg_01", "digitized by",
       S("L1i_was_digitized_by", "C1", "D2"),
       S("P9i_forms_part_of", "D2", "E5", rec=True))],
    specialization_of="thing_event_c",
)

# =============================================================================
# THING-CONCEPT
# =============================================================================

thing_concept_a = FR(
    "thing_concept_a", "has type", "Thing", "Concept",
    "C1.Object", "E55_Type",
    [
        P("tc_a_01", "has type",
          S("P2_has_type", "E1", "E55"),
          S("P127_has_broader_term", "E55", "E55", rec=True)),
        P("tc_a_02", "consists of material",
          S("P45_consists_of", "E18", "E57"),
          S("P127_has_broader_term", "E57", "E55", rec=True)),
        P("tc_a_03", "technique -> material",
          S("P92i_was_brought_into_existence_by", "C1", "E7"),
          S("P9i_forms_part_of", "E7", "E7", rec=True),
          S("P33_used_specific_technique", "E7", "E29"),
          S("P68_foresees_use_of", "E29", "E57")),
        P("tc_a_04", "modification employed material",
          S("P92i_was_brought_into_existence_by", "C1", "E7"),
          S("P9i_forms_part_of", "E7", "E7", rec=True),
          S("P126_employed", "E11", "E57")),
    ],
)

thing_concept_a_made_of = FR(
    "thing_concept_a_made_of", "is made of", "Thing", "Concept",
    "C1.Object", "E57_Material",
    [
        P("tc_a_mo_01", "consists of",
          S("P45_consists_of", "E18", "E57")),
        P("tc_a_mo_02", "technique -> material",
          S("P92i_was_brought_into_existence_by", "C1", "E7"),
          S("P9i_forms_part_of", "E7", "E7", rec=True),
          S("P33_used_specific_technique", "E7", "E29"),
          S("P68_foresees_use_of", "E29", "E57")),
    ],
    specialization_of="thing_concept_a",
)

# =============================================================================
# PLACE-PLACE, PLACE-THING, PLACE-ACTOR, PLACE-EVENT
# =============================================================================

place_place_a = FR("place_place_a", "has part", "Place", "Place", "E53_Place", "E53_Place",
    [P("pp_a_01", "contains", S("P89i_contains", "E53", "E53", rec=True))])

place_place_b = FR("place_place_b", "is part of", "Place", "Place", "E53_Place", "E53_Place",
    [P("pp_b_01", "falls within", S("P89_falls_within", "E53", "E53", rec=True))])

place_place_c = FR("place_place_c", "borders or overlaps with", "Place", "Place", "E53_Place", "E53_Place",
    [P("pp_c_01", "contains -> borders -> falls within",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P122_borders_with", "E53", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True)),
     P("pp_c_02", "contains -> overlaps -> falls within",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P121_overlaps_with", "E53", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True))])

place_thing_a = FR("place_thing_a", "refers to", "Place", "Thing", "E53_Place", "C1.Object",
    [P("pt_a_01", "location of thing -> depicts",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P53i_is_former_or_current_location_of", "E53", "E24"),
       S("P62_depicts", "E24", "C1")),
     P("pt_a_02", "location of thing -> carries -> refers to",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P53i_is_former_or_current_location_of", "E53", "E24"),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "C1")),
     P("pt_a_03", "residence of actor -> owns -> depicts",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P74i_is_current_or_former_residence_of", "E53", "E39"),
       S("P51i_is_former_or_current_owner_of", "E39", "E24"),
       S("P62_depicts", "E24", "C1")),
     P("pt_a_04", "witnessed -> created -> refers to",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P7i_witnessed", "E53", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P94_has_created", "E65", "E89"),
       S("P67_refers_to", "E89", "C1")),
     P("pt_a_05", "witnessed -> produced -> depicts",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P7i_witnessed", "E53", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P108_has_produced", "E12", "E24"),
       S("P62_depicts", "E24", "C1"))])

place_thing_b = FR("place_thing_b", "is referred to by", "Place", "Thing", "E53_Place", "C1.Object",
    [P("pt_b_01", "is referred to by -> component of",
       S("P89_falls_within", "E53", "E53", rec=True),
       S("P67i_is_referred_to_by", "E53", "E89"),
       S("P148i_is_component_of", "E89", "E89", rec=True)),
     P("pt_b_02", "is referred to by -> carried by",
       S("P89_falls_within", "E53", "E53", rec=True),
       S("P67i_is_referred_to_by", "E53", "E89"),
       S("P128i_is_carried_by", "E73", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True)),
     P("pt_b_03", "is depicted by",
       S("P89_falls_within", "E53", "E53", rec=True),
       S("P62i_is_depicted_by", "E53", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True))])

place_thing_c = FR("place_thing_c", "has met", "Place", "Thing", "E53_Place", "E18_Physical_Thing",
    [P("pt_c_01", "location of thing",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P53i_is_former_or_current_location_of", "E53", "E18"),
       S("P46_is_composed_of", "E18", "E18", rec=True)),
     P("pt_c_02", "witnessed creation of thing",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P7i_witnessed", "E53", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P46_is_composed_of", "C1", "C1", rec=True)),
     P("pt_c_03", "residence of keeper of thing",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P74i_is_current_or_former_residence_of", "E53", "E39"),
       S("P49i_is_former_or_current_keeper_of", "E39", "E18"),
       S("P46_is_composed_of", "E18", "E18", rec=True))])

place_actor_a = FR("place_actor_a", "refers to", "Place", "Actor", "E53_Place", "E39_Actor",
    [P("pa_a_01", "location of thing -> depicts Actor",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P53i_is_former_or_current_location_of", "E53", "E24"),
       S("P62_depicts", "E24", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
     P("pa_a_02", "location of thing -> carries -> refers to Actor",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P53i_is_former_or_current_location_of", "E53", "E24"),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
     P("pa_a_03", "witnessed -> created -> refers to Actor",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P7i_witnessed", "E53", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P94_has_created", "E65", "E89"),
       S("P67_refers_to", "E89", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True))])

place_actor_b = FR("place_actor_b", "is referred to by", "Place", "Actor", "E53_Place", "E39_Actor",
    [P("pa_b_01", "is referred to -> created by actor",
       S("P89_falls_within", "E53", "E53", rec=True),
       S("P67i_is_referred_to_by", "E53", "E89"),
       S("P94i_was_created_by", "E89", "E65"),
       S("P9i_forms_part_of", "E65", "E7", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
     P("pa_b_02", "is depicted by -> produced by actor",
       S("P89_falls_within", "E53", "E53", rec=True),
       S("P62i_is_depicted_by", "E53", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P108i_was_produced_by", "E24", "E12"),
       S("P9i_forms_part_of", "E12", "E5", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True))])

place_actor_c = FR("place_actor_c", "has met", "Place", "Actor", "E53_Place", "E39_Actor",
    [P("pa_c_01", "residence of actor",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P74i_is_current_or_former_residence_of", "E53", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
     P("pa_c_02", "witnessed event -> actor present",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P7i_witnessed", "E53", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P12_occurred_in_the_presence_of", "E5", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))])

place_actor_c_origin_of = FR("place_actor_c_origin_of", "is origin of", "Place", "Actor", "E53_Place", "E39_Actor",
    [P("pa_c_oo_01", "witnessed birth",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P7i_witnessed", "E53", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P98_brought_into_life", "E67", "E21"),
       S("P107_has_current_or_former_member", "E21", "E39", rec=True))],
    specialization_of="place_actor_c")

place_event_a = FR("place_event_a", "has met", "Place", "Event", "E53_Place", "E5_Event",
    [P("pe_a_01", "witnessed Event",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P7i_witnessed", "E53", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True))])

place_event_b = FR("place_event_b", "refers to", "Place", "Event", "E53_Place", "E5_Event",
    [P("pe_b_01", "location of thing -> depicts Event",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P53i_is_former_or_current_location_of", "E53", "E24"),
       S("P46_is_composed_of", "E24", "E24", rec=True),
       S("P62_depicts", "E24", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True)),
     P("pe_b_02", "location of thing -> carries -> refers to Event",
       S("P89i_contains", "E53", "E53", rec=True),
       S("P53i_is_former_or_current_location_of", "E53", "E24"),
       S("P46_is_composed_of", "E24", "E24", rec=True),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True))])

# =============================================================================
# ACTOR-PLACE, ACTOR-THING, ACTOR-ACTOR, ACTOR-EVENT
# =============================================================================

actor_place_a = FR("actor_place_a", "refers to", "Actor", "Place", "E39_Actor", "E53_Place",
    [P("ap_a_01", "performed -> created -> depicts Place",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P62_depicts", "E24", "E53"),
       S("P89i_contains", "E53", "E53", rec=True)),
     P("ap_a_02", "performed -> created -> refers to Place",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P67_refers_to", "E89", "E53"),
       S("P89i_contains", "E53", "E53", rec=True))])

actor_place_b = FR("actor_place_b", "is referred to at", "Actor", "Place", "E39_Actor", "E53_Place",
    [P("ap_b_01", "is referred to -> created at Place",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P67i_is_referred_to_by", "E39", "E89"),
       S("P94i_was_created_by", "E89", "E65"),
       S("P9i_forms_part_of", "E65", "E5", rec=True),
       S("P7_took_place_at", "E5", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True)),
     P("ap_b_02", "is referred to -> carried by -> location",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P67i_is_referred_to_by", "E39", "E89"),
       S("P128i_is_carried_by", "E73", "E24"),
       S("P46i_forms_part_of", "E24", "E18", rec=True),
       S("P53_has_former_or_current_location", "E18", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True)),
     P("ap_b_03", "is depicted by -> location",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P62i_is_depicted_by", "E39", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P53_has_former_or_current_location", "E24", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True)),
     P("ap_b_04", "is depicted by -> produced at Place",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P62i_is_depicted_by", "E39", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P108i_was_produced_by", "E24", "E12"),
       S("P9i_forms_part_of", "E12", "E5", rec=True),
       S("P7_took_place_at", "E5", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True))])

actor_place_c = FR("actor_place_c", "has met", "Actor", "Place", "E39_Actor", "E53_Place",
    [P("ap_c_01", "present at event at Place",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P12i_was_present_at", "E39", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P7_took_place_at", "E5", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True)),
     P("ap_c_02", "residence",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P74_has_current_or_former_residence", "E39", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True))])

actor_place_d = FR("actor_place_d", "from", "Actor", "Place", "E39_Actor", "E53_Place",
    [P("ap_d_01", "residence",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P74_has_current_or_former_residence", "E39", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True)),
     P("ap_d_02", "born at Place",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P92i_was_brought_into_existence_by", "E39", "E63"),
       S("P9i_forms_part_of", "E63", "E5", rec=True),
       S("P7_took_place_at", "E5", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True))])

actor_thing_a = FR("actor_thing_a", "refers to", "Actor", "Thing", "E39_Actor", "C1.Object",
    [P("at_a_01", "performed -> created -> P130* -> depicts Thing",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "E24", "E24", rec=True),
       S("P62_depicts", "E24", "C1")),
     P("at_a_02", "performed -> created -> P130* -> carries -> refers to Thing",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "E24", "E24", rec=True),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "C1"))])

actor_thing_b = FR("actor_thing_b", "is referred to by", "Actor", "Thing", "E39_Actor", "C1.Object",
    [P("at_b_01", "is referred to by -> component of",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P67i_is_referred_to_by", "E39", "E89"),
       S("P148i_is_component_of", "E89", "E89", rec=True)),
     P("at_b_02", "is referred to -> carried by",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P67i_is_referred_to_by", "E39", "E89"),
       S("P128i_is_carried_by", "E73", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True)),
     P("at_b_03", "is depicted by",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P62i_is_depicted_by", "E39", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True))])

actor_thing_c = FR("actor_thing_c", "is origin of", "Actor", "Thing", "E39_Actor", "C1.Object",
    [P("at_c_01", "performed -> brought into existence",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P46_is_composed_of", "C1", "C1", rec=True)),
     P("at_c_02", "is keeper of",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P49i_is_former_or_current_keeper_of", "E39", "E18"),
       S("P46_is_composed_of", "E18", "E18", rec=True)),
     P("at_c_03", "is owner of",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P51i_is_former_or_current_owner_of", "E39", "E18"),
       S("P46_is_composed_of", "E18", "E18", rec=True))])

# Actor-Actor
actor_actor_a = FR("actor_actor_a", "refers to", "Actor", "Actor", "E39_Actor", "E39_Actor",
    [P("aa_a_01", "performed -> created -> P130* -> depicts Actor",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "E24", "E24", rec=True),
       S("P62_depicts", "E24", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
     P("aa_a_02", "performed -> created -> carries -> refers to Actor",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "E24", "E24", rec=True),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True))])

actor_actor_c = FR("actor_actor_c", "is referred to by", "Actor", "Actor", "E39_Actor", "E39_Actor",
    [P("aa_c_01", "is referred to -> created by actor",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P67i_is_referred_to_by", "E39", "E89"),
       S("P94i_was_created_by", "E89", "E65"),
       S("P9i_forms_part_of", "E65", "E7", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
     P("aa_c_02", "is depicted by -> produced by actor",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P62i_is_depicted_by", "E39", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P108i_was_produced_by", "E24", "E12"),
       S("P9i_forms_part_of", "E12", "E7", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))])

actor_actor_d = FR("actor_actor_d", "from", "Actor", "Actor", "E39_Actor", "E39_Actor",
    [P("aa_d_01", "born from father/mother",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P98i_was_born", "E21", "E67"),
       S("P97_from_father", "E67", "E21")),
     P("aa_d_02", "group formed by actor",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P95i_was_formed_by", "E74", "E66"),
       S("P9_consists_of", "E66", "E7", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))])

actor_actor_e = FR("actor_actor_e", "is origin of", "Actor", "Actor", "E39_Actor", "E39_Actor",
    [P("aa_e_01", "is father/mother of",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P97i_was_father_for", "E21", "E67"),
       S("P98_brought_into_life", "E67", "E21")),
     P("aa_e_02", "performed -> formation of group",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9i_forms_part_of", "E7", "E5", rec=True),
       S("P95_has_formed", "E66", "E74"),
       S("P107_has_current_or_former_member", "E74", "E39", rec=True))])

actor_actor_f = FR("actor_actor_f", "has member", "Actor", "Actor", "E74_Group", "E39_Actor",
    [P("aa_f_01", "has member",
       S("P107_has_current_or_former_member", "E74", "E39", rec=True)),
     P("aa_f_02", "gained member by joining",
       S("P144i_gained_member_by", "E74", "E85"),
       S("P143i_joined", "E85", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True))])

actor_actor_g = FR("actor_actor_g", "is member of", "Actor", "Actor", "E39_Actor", "E74_Group",
    [P("aa_g_01", "is member of",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
     P("aa_g_02", "was joined by joining",
       S("P143_was_joined_by", "E39", "E85"),
       S("P144_joined_with", "E85", "E74"),
       S("P107i_is_current_or_former_member_of", "E74", "E39", rec=True))])

# =============================================================================
# ACTOR-EVENT
# =============================================================================

actor_event_a = FR("actor_event_a", "refers to", "Actor", "Event", "E39_Actor", "E5_Event",
    [P("ae_a_01", "performed -> created -> depicts Event",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P62_depicts", "E24", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True)),
     P("ae_a_02", "performed -> created -> carries -> refers to Event",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True))])

actor_event_b = FR("actor_event_b", "is referred to at", "Actor", "Event", "E39_Actor", "E5_Event",
    [P("ae_b_01", "is referred to -> creation event",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P67i_is_referred_to_by", "E39", "E89"),
       S("P94i_was_created_by", "E89", "E65"),
       S("P9i_forms_part_of", "E65", "E5", rec=True)),
     P("ae_b_02", "is depicted by -> production event",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P62i_is_depicted_by", "E39", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P108i_was_produced_by", "E24", "E12"),
       S("P9i_forms_part_of", "E12", "E5", rec=True))])

actor_event_c = FR("actor_event_c", "has met", "Actor", "Event", "E39_Actor", "E5_Event",
    [P("ae_c_01", "performed event",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9_consists_of", "E7", "E5", rec=True)),
     P("ae_c_02", "present at event",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P12i_was_present_at", "E39", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True))])

actor_concept_a = FR("actor_concept_a", "has type", "Actor", "Concept", "E39_Actor", "E55_Type",
    [P("ac_a_01", "has type",
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True),
       S("P2_has_type", "E39", "E55"),
       S("P127_has_broader_term", "E55", "E55", rec=True))])

# =============================================================================
# EVENT-PLACE
# =============================================================================

event_place_a = FR("event_place_a", "refers to", "Event", "Place", "E5_Event", "E53_Place",
    [P("ep_a_01", "created -> P130* -> depicts Place",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P62_depicts", "E24", "E53"),
       S("P89i_contains", "E53", "E53", rec=True)),
     P("ep_a_02", "created -> carries -> refers to Place",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "E53"),
       S("P89i_contains", "E53", "E53", rec=True))])

event_place_b = FR("event_place_b", "is referred to at", "Event", "Place", "E5_Event", "E53_Place",
    [P("ep_b_01", "is referred to -> created at Place",
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P67i_is_referred_to_by", "E5", "E89"),
       S("P94i_was_created_by", "E89", "E65"),
       S("P9i_forms_part_of", "E65", "E5", rec=True),
       S("P7_took_place_at", "E5", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True)),
     P("ep_b_02", "is referred to -> carried by -> location",
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P67i_is_referred_to_by", "E5", "E89"),
       S("P128i_is_carried_by", "E73", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P53_has_former_or_current_location", "E24", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True)),
     P("ep_b_03", "is depicted by -> location",
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P62i_is_depicted_by", "E5", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P53_has_former_or_current_location", "E24", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True))])

event_place_c = FR("event_place_c", "from", "Event", "Place", "E5_Event", "E53_Place",
    [P("ep_c_01", "took place at",
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P7_took_place_at", "E5", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True))])

# =============================================================================
# EVENT-THING
# =============================================================================

event_thing_a = FR("event_thing_a", "refers to or is about", "Event", "Thing", "E5_Event", "C1.Object",
    [P("et_a_01", "created -> P130* -> depicts Thing",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P62_depicts", "E24", "C1"),
       S("P46_is_composed_of", "C1", "C1", rec=True)),
     P("et_a_02", "created -> carries -> refers to Thing",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "C1"),
       S("P46_is_composed_of", "C1", "C1", rec=True))])

event_thing_b = FR("event_thing_b", "is referred to by", "Event", "Thing", "E5_Event", "C1.Object",
    [P("et_b_01", "is referred to by -> P130* -> component",
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P67i_is_referred_to_by", "E5", "E89"),
       S("P130_shows_features_of", "E89", "E89", rec=True),
       S("P148_has_component", "E89", "E89", rec=True)),
     P("et_b_02", "is depicted by -> P130* -> part of",
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P62i_is_depicted_by", "E5", "E24"),
       S("P130_shows_features_of", "E24", "E24", rec=True),
       S("P46i_forms_part_of", "E24", "E24", rec=True))])

event_thing_c = FR("event_thing_c", "has met", "Event", "Thing", "E5_Event", "C1.Object",
    [P("et_c_01", "occurred in presence of",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P12_occurred_in_the_presence_of", "E5", "C1"),
       S("P46_is_composed_of", "C1", "C1", rec=True))])

event_thing_c_created = FR("event_thing_c_created", "created", "Event", "Thing", "E5_Event", "C1.Object",
    [P("et_c_cr_01", "brought into existence",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P46_is_composed_of", "C1", "C1", rec=True))],
    specialization_of="event_thing_c")

event_thing_c_destroyed = FR("event_thing_c_destroyed", "destroyed", "Event", "Thing", "E5_Event", "C1.Object",
    [P("et_c_de_01", "took out of existence",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P93_took_out_of_existence", "E64", "C1"),
       S("P46_is_composed_of", "C1", "C1", rec=True))],
    specialization_of="event_thing_c")

event_thing_c_modified = FR("event_thing_c_modified", "modified", "Event", "Thing", "E5_Event", "E24",
    [P("et_c_mo_01", "modified",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P31_has_modified", "E11", "E24"),
       S("P46_is_composed_of", "E24", "E24", rec=True))],
    specialization_of="event_thing_c")

# =============================================================================
# EVENT-ACTOR, EVENT-EVENT, EVENT-CONCEPT
# =============================================================================

event_actor_a = FR("event_actor_a", "has met", "Event", "Actor", "E5_Event", "E39_Actor",
    [P("ea_a_01", "had participant",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P11_had_participant", "E5", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True))])

event_event_a = FR("event_event_a", "has met", "Event", "Event", "E5_Event", "E5_Event",
    [P("ee_a_01", "overlaps in time",
       S("P118_overlaps_in_time_with", "E5", "E5")),
     P("ee_a_02", "meets in time",
       S("P119_meets_in_time_with", "E5", "E5")),
     P("ee_a_03", "occurs during",
       S("P117_occurs_during", "E5", "E5"))])

event_event_c = FR("event_event_c", "from", "Event", "Event", "E5_Event", "E5_Event",
    [P("ee_c_01", "forms part of",
       S("P9i_forms_part_of", "E5", "E5", rec=True)),
     P("ee_c_02", "falls within",
       S("P10_falls_within", "E5", "E5", rec=True)),
     P("ee_c_03", "all temporal relations",
       S("P119_meets_in_time_with", "E5", "E5")),
     P("ee_c_04", "is equal in time to",
       S("P114_is_equal_in_time_to", "E5", "E5"))])

event_event_d = FR("event_event_d", "has part", "Event", "Event", "E5_Event", "E5_Event",
    [P("ee_d_01", "consists of",
       S("P9_consists_of", "E5", "E5", rec=True)),
     P("ee_d_02", "contains",
       S("P10i_contains", "E5", "E5", rec=True))])

event_concept_a = FR("event_concept_a", "has type", "Event", "Concept", "E5_Event", "E55_Type",
    [P("ec_a_01", "has type",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P2_has_type", "E5", "E55"),
       S("P127_has_broader_term", "E55", "E55", rec=True))])

# =============================================================================
# CONCEPT-*
# =============================================================================

concept_place_a = FR("concept_place_a", "is type of", "Concept", "Place", "E55_Type", "E53_Place",
    [P("cp_a_01", "is type of Place",
       S("P127i_has_narrower_term", "E55", "E55", rec=True),
       S("P2i_is_type_of", "E55", "E53"),
       S("P89_falls_within", "E53", "E53", rec=True))])

concept_thing_a = FR("concept_thing_a", "is type of", "Concept", "Thing", "E55_Type", "C1.Object",
    [P("ct_a_01", "is type of Thing",
       S("P127i_has_narrower_term", "E55", "E55", rec=True),
       S("P2i_is_type_of", "E55", "C1"),
       S("P46i_forms_part_of", "C1", "C1", rec=True))])

concept_actor_a = FR("concept_actor_a", "is type of", "Concept", "Actor", "E55_Type", "E39_Actor",
    [P("ca_a_01", "is type of Actor",
       S("P127i_has_narrower_term", "E55", "E55", rec=True),
       S("P2i_is_type_of", "E55", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))])

concept_event_a = FR("concept_event_a", "is type of", "Concept", "Event", "E55_Type", "E5_Event",
    [P("ce_a_01", "is type of Event",
       S("P127i_has_narrower_term", "E55", "E55", rec=True),
       S("P2i_is_type_of", "E55", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True))])

concept_concept_a = FR("concept_concept_a", "has type (broader)", "Concept", "Concept", "E55_Type", "E55_Type",
    [P("cc_a_01", "has broader term",
       S("P127_has_broader_term", "E55", "E55", rec=True))])

concept_concept_b = FR("concept_concept_b", "is type of (narrower)", "Concept", "Concept", "E55_Type", "E55_Type",
    [P("cc_b_01", "has narrower term",
       S("P127i_has_narrower_term", "E55", "E55", rec=True))])

place_concept_a = FR("place_concept_a", "has type", "Place", "Concept", "E53_Place", "E55_Type",
    [P("pc_a_01", "has type",
       S("P89_falls_within", "E53", "E53", rec=True),
       S("P2_has_type", "E53", "E55"),
       S("P127_has_broader_term", "E55", "E55", rec=True))])


# =============================================================================
# REGISTRY: all FRs as a flat list
# =============================================================================


# =============================================================================
# MISSING FRs - Actor-Thing d, Actor-Actor b (has met), 
# Actor-Event c/d + specializations, Event-Actor a/b/c/d + specs,
# Event-Event a/b, Event-Thing c_used, Place-Event c/d
# =============================================================================

# Actor-Thing d. has met (was missing entirely)
actor_thing_d = FR("actor_thing_d", "has met", "Actor", "Thing", "E39_Actor", "E70_Thing",
    [P("at_d_01", "present at event -> in presence of Thing",
       S("P12i_was_present_at", "E39", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P12_occurred_in_the_presence_of", "E5", "E70"))])

# Actor-Actor b. has met (was missing)
actor_actor_b = FR("actor_actor_b", "has met", "Actor", "Actor", "E39_Actor", "E39_Actor",
    [P("aa_b_01", "present at event -> in presence of Actor",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P12i_was_present_at", "E39", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P12_occurred_in_the_presence_of", "E5", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))])

# Actor-Event c. from (was mislabeled as "has met")
# Fix: rename actor_event_c to actor_event_d (has met), create proper c (from)
actor_event_c_from = FR("actor_event_c_from", "from", "Actor", "Event", "E39_Actor", "E5_Event",
    [P("ae_c_f_01", "group formed by event",
       S("P107i_is_current_or_former_member_of", "E74", "E74", rec=True),
       S("P95i_was_formed_by", "E74", "E66"),
       S("P9i_forms_part_of", "E66", "E5", rec=True)),
     P("ae_c_f_02", "person born at event",
       S("P98i_was_born", "E21", "E67"),
       S("P9i_forms_part_of", "E67", "E5", rec=True))])

# Actor-Event d. has met (general)
actor_event_d = FR("actor_event_d", "has met", "Actor", "Event", "E39_Actor", "E5_Event",
    [P("ae_d_01", "present at event",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P12i_was_present_at", "E39", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True))])

# Actor-Event d. specialization: performed action at
actor_event_d_performed = FR("actor_event_d_performed", "performed action at", "Actor", "Event", "E39_Actor", "E5_Event",
    [P("ae_d_pa_01", "performed action at Event",
       S("P107_has_current_or_former_member", "E39", "E39", rec=True),
       S("P14i_performed", "E39", "E7"),
       S("P9i_forms_part_of", "E7", "E5", rec=True))],
    specialization_of="actor_event_d")

# Event-Actor a. refers to or is about
event_actor_a_refers = FR("event_actor_a_refers", "refers to or is about", "Event", "Actor", "E5_Event", "E39_Actor",
    [P("ea_ar_01", "created -> P130* -> depicts Actor",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P62_depicts", "E24", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
     P("ea_ar_02", "created -> P130* -> refers to Actor",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P67_refers_to", "E89", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True)),
     P("ea_ar_03", "created -> carries -> refers to Actor",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True))])

# Event-Actor b. is referred to by
event_actor_b = FR("event_actor_b", "is referred to by", "Event", "Actor", "E5_Event", "E39_Actor",
    [P("ea_b_01", "is referred to -> P130* -> created by actor",
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P67i_is_referred_to_by", "E5", "E89"),
       S("P130_shows_features_of", "E89", "E89", rec=True),
       S("P148_has_component", "E89", "E89", rec=True),
       S("P94i_was_created_by", "E89", "E65"),
       S("P9i_forms_part_of", "E65", "E65", rec=True),
       S("P14_carried_out_by", "E65", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
     P("ea_b_02", "is depicted by -> P130* -> produced by actor",
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P62i_is_depicted_by", "E5", "E24"),
       S("P130_shows_features_of", "E24", "E24", rec=True),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P108i_was_produced_by", "E24", "E12"),
       S("P9i_forms_part_of", "E12", "E12", rec=True),
       S("P14_carried_out_by", "E12", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))])

# Event-Actor c. by
event_actor_c = FR("event_actor_c", "by", "Event", "Actor", "E5_Event", "E39_Actor",
    [P("ea_c_01", "carried out by actor",
       S("P9i_forms_part_of", "E5", "E7", rec=True),
       S("P14_carried_out_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)),
     P("ea_c_02", "influenced by actor",
       S("P9i_forms_part_of", "E5", "E7", rec=True),
       S("P15_was_influenced_by", "E7", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))])

# Event-Actor d. has met (rename old event_actor_a)
event_actor_d = FR("event_actor_d", "has met", "Event", "Actor", "E5_Event", "E39_Actor",
    [P("ea_d_01", "occurred in presence of actor",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P12_occurred_in_the_presence_of", "E5", "E39"),
       S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True))])

# Event-Actor d. specializations
event_actor_d_brought_into_existence = FR(
    "event_actor_d_brought_into_existence", "brought into existence", "Event", "Actor", "E5_Event", "E39_Actor",
    [P("ea_d_bie_01", "brought into existence actor",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True))],
    specialization_of="event_actor_d")

event_actor_d_took_out_of_existence = FR(
    "event_actor_d_took_out_of_existence", "took out of existence", "Event", "Actor", "E5_Event", "E39_Actor",
    [P("ea_d_tooe_01", "took out of existence actor",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P93_took_out_of_existence", "E64", "E39"),
       S("P107_has_current_or_former_member", "E39", "E39", rec=True))],
    specialization_of="event_actor_d")

# Event-Thing c. used (was missing)
event_thing_c_used = FR("event_thing_c_used", "used", "Event", "Thing", "E5_Event", "C1.Object",
    [P("et_c_us_01", "used specific object",
       S("P9_consists_of", "E5", "E7", rec=True),
       S("P16_used_specific_object", "E7", "C1"),
       S("P46_is_composed_of", "C1", "C1", rec=True)),
     P("et_c_us_02", "used object of type",
       S("P9_consists_of", "E5", "E7", rec=True),
       S("P125_used_object_of_type", "E7", "E55"))],
    specialization_of="event_thing_c")

# Event-Event a. refers to or is about (was mislabeled as "has met")
event_event_a_refers = FR("event_event_a_refers", "refers to or is about", "Event", "Event", "E5_Event", "E5_Event",
    [P("ee_ar_01", "created -> P130* -> depicts Event",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P62_depicts", "E24", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True)),
     P("ee_ar_02", "created -> P130* -> refers to Event",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P67_refers_to", "E89", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True)),
     P("ee_ar_03", "created -> carries -> refers to Event",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P92_brought_into_existence", "E63", "C1"),
       S("P130_shows_features_of", "C1", "C1", rec=True),
       S("P128_carries", "E24", "E73"),
       S("P67_refers_to", "E73", "E5"),
       S("P9_consists_of", "E5", "E5", rec=True))])

# Event-Event b. is referred to at
event_event_b = FR("event_event_b", "is referred to at", "Event", "Event", "E5_Event", "E5_Event",
    [P("ee_b_01", "is referred to -> P130* -> created at Event",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P67i_is_referred_to_by", "E5", "E89"),
       S("P130_shows_features_of", "E89", "E89", rec=True),
       S("P148_has_component", "E89", "E89", rec=True),
       S("P94i_was_created_by", "E89", "E65"),
       S("P9i_forms_part_of", "E65", "E5", rec=True)),
     P("ee_b_02", "is referred to -> carried by -> produced at Event",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P67i_is_referred_to_by", "E5", "E89"),
       S("P128i_is_carried_by", "E73", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P108i_was_produced_by", "E24", "E12"),
       S("P9i_forms_part_of", "E12", "E5", rec=True)),
     P("ee_b_03", "is depicted by -> P130* -> produced at Event",
       S("P9_consists_of", "E5", "E5", rec=True),
       S("P62i_is_depicted_by", "E5", "E24"),
       S("P130_shows_features_of", "E24", "E24", rec=True),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P108i_was_produced_by", "E24", "E12"),
       S("P9i_forms_part_of", "E12", "E5", rec=True))])

# Event-Event a. has met (the original was temporal, rename to event_event_has_met)
event_event_has_met = FR("event_event_has_met", "has met (temporal)", "Event", "Event", "E5_Event", "E5_Event",
    [P("ee_hm_01", "overlaps in time",
       S("P118_overlaps_in_time_with", "E5", "E5")),
     P("ee_hm_02", "meets in time",
       S("P119_meets_in_time_with", "E5", "E5")),
     P("ee_hm_03", "occurs during",
       S("P117_occurs_during", "E5", "E5"))])

# Place-Event c. (was labelled d. in PDF) is referred to at
place_event_c = FR("place_event_c", "is referred to at", "Place", "Event", "E53_Place", "E5_Event",
    [P("pe_c_01", "is referred to -> P130* -> created at Event",
       S("P89_falls_within", "E53", "E53", rec=True),
       S("P67i_is_referred_to_by", "E53", "E89"),
       S("P130_shows_features_of", "E89", "E89", rec=True),
       S("P94i_was_created_by", "E89", "E65"),
       S("P9i_forms_part_of", "E65", "E5", rec=True)),
     P("pe_c_02", "is referred to -> carried by -> produced at Event",
       S("P89_falls_within", "E53", "E53", rec=True),
       S("P67i_is_referred_to_by", "E53", "E89"),
       S("P128i_is_carried_by", "E73", "E24"),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P108i_was_produced_by", "E24", "E12"),
       S("P9i_forms_part_of", "E12", "E5", rec=True)),
     P("pe_c_03", "is depicted by -> P130* -> produced at Event",
       S("P89_falls_within", "E53", "E53", rec=True),
       S("P62i_is_depicted_by", "E53", "E24"),
       S("P130_shows_features_of", "E24", "E24", rec=True),
       S("P46i_forms_part_of", "E24", "E24", rec=True),
       S("P108i_was_produced_by", "E24", "E12"),
       S("P9i_forms_part_of", "E12", "E5", rec=True))])

# Place-Concept (was missing from list)
# Already present as place_concept_a above

# =============================================================================
# EVENT-TIME / THING-TIME / ACTOR-TIME
# =============================================================================

event_time_a = FR(
    "event_time_a", "occurred at time", "Event", "Time",
    "E2_Temporal_Entity", "E52_Time-Span",
    [P("et_a_01", "has time-span",
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
)

thing_time_a = FR(
    "thing_time_a", "has time", "Thing", "Time",
    "E77_Persistent_Item", "E52_Time-Span",
    [P("tt_a_01", "was present at -> time-span",
       S("P12i_was_present_at", "E77", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True)),
     P("tt_a_02", "changed ownership -> time-span",
       S("P24i_changed_ownership_through", "E18", "E8"),
       S("P9i_forms_part_of", "E8", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
)

thing_time_a_created = FR(
    "thing_time_a_created", "was created at time", "Thing", "Time",
    "E77_Persistent_Item", "E52_Time-Span",
    [P("tt_a_cr_01", "brought into existence -> time-span",
       S("P92i_was_brought_into_existence_by", "E77", "E63"),
       S("P9i_forms_part_of", "E63", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
    specialization_of="thing_time_a",
)

thing_time_a_modified = FR(
    "thing_time_a_modified", "was modified at time", "Thing", "Time",
    "E77_Persistent_Item", "E52_Time-Span",
    [P("tt_a_mo_01", "modified by -> time-span",
       S("P31i_was_modified_by", "E18", "E7"),
       S("P9i_forms_part_of", "E7", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
    specialization_of="thing_time_a",
)

thing_time_a_moved = FR(
    "thing_time_a_moved", "was moved at time", "Thing", "Time",
    "E19_Physical_Object", "E52_Time-Span",
    [P("tt_a_mv_01", "moved by -> time-span",
       S("P25i_moved_by", "E19", "E9"),
       S("P9i_forms_part_of", "E9", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
    specialization_of="thing_time_a",
)

thing_time_a_destroyed = FR(
    "thing_time_a_destroyed", "was destroyed at time", "Thing", "Time",
    "E77_Persistent_Item", "E52_Time-Span",
    [P("tt_a_de_01", "taken out of existence -> time-span",
       S("P93i_was_taken_out_of_existence_by", "E77", "E64"),
       S("P9i_forms_part_of", "E64", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
    specialization_of="thing_time_a",
)

actor_time_a = FR(
    "actor_time_a", "participated at time", "Actor", "Time",
    "E39_Actor", "E52_Time-Span",
    [P("at_a_01", "participated in -> time-span",
       S("P11i_participated_in", "E39", "E5"),
       S("P9i_forms_part_of", "E5", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
)

actor_time_b = FR(
    "actor_time_b", "was born or formed at time", "Actor", "Time",
    "E39_Actor", "E52_Time-Span",
    [P("at_b_01", "brought into existence -> time-span",
       S("P92i_was_brought_into_existence_by", "E39", "E63"),
       S("P9i_forms_part_of", "E63", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
)

actor_time_b_born = FR(
    "actor_time_b_born", "was born at time", "Actor", "Time",
    "E21_Person", "E52_Time-Span",
    [P("at_b_bo_01", "was born -> time-span",
       S("P98i_was_born", "E21", "E67"),
       S("P9i_forms_part_of", "E67", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
    specialization_of="actor_time_b",
)

actor_time_c = FR(
    "actor_time_c", "died or dissolved at time", "Actor", "Time",
    "E39_Actor", "E52_Time-Span",
    [P("at_c_01", "taken out of existence -> time-span",
       S("P93i_was_taken_out_of_existence_by", "E39", "E64"),
       S("P9i_forms_part_of", "E64", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
)

actor_time_c_died = FR(
    "actor_time_c_died", "died at time", "Actor", "Time",
    "E21_Person", "E52_Time-Span",
    [P("at_c_di_01", "died in -> time-span",
       S("P100i_died_in", "E21", "E69"),
       S("P9i_forms_part_of", "E69", "E5", rec=True),
       S("P4_has_time-span", "E2", "E52_Time-Span"),
       S("P86_falls_within", "E52_Time-Span", "E52_Time-Span", rec=True))],
    specialization_of="actor_time_c",
)


ALL_FRS = [
    # Thing-Place
    thing_place_a, thing_place_b, thing_place_c,
    thing_place_c_created_in, thing_place_c_found_acquired,
    thing_place_c_by_person_from, thing_place_c_located_in,
    thing_place_c_moved_from, thing_place_c_moved_to,
    # Thing-Thing
    thing_thing_a, thing_thing_b, thing_thing_c, thing_thing_d,
    thing_thing_d_is_part_of, thing_thing_e, thing_thing_f,
    # Thing-Actor
    thing_actor_a, thing_actor_b, thing_actor_c, thing_actor_d,
    thing_actor_d_created_by, thing_actor_d_used_by,
    thing_actor_d_modified_by, thing_actor_d_found_acquired_by,
    thing_actor_e,
    # Thing-Event
    thing_event_a, thing_event_b, thing_event_c,
    thing_event_c_destroyed_in, thing_event_c_created_in,
    thing_event_c_modified_in, thing_event_c_used_in,
    thing_event_c_digitized_in,
    # Thing-Concept
    thing_concept_a, thing_concept_a_made_of,
    # Place-Place
    place_place_a, place_place_b, place_place_c,
    # Place-Thing
    place_thing_a, place_thing_b, place_thing_c,
    # Place-Actor
    place_actor_a, place_actor_b, place_actor_c, place_actor_c_origin_of,
    # Place-Event
    place_event_a, place_event_b, place_event_c,
    # Place-Concept
    place_concept_a,
    # Actor-Place
    actor_place_a, actor_place_b, actor_place_c, actor_place_d,
    # Actor-Thing
    actor_thing_a, actor_thing_b, actor_thing_c, actor_thing_d,
    # Actor-Actor
    actor_actor_a, actor_actor_b, actor_actor_c, actor_actor_d,
    actor_actor_e, actor_actor_f, actor_actor_g,
    # Actor-Event
    actor_event_a, actor_event_b,
    actor_event_c_from, actor_event_d, actor_event_d_performed,
    # Actor-Concept
    actor_concept_a,
    # Event-Place
    event_place_a, event_place_b, event_place_c,
    # Event-Thing
    event_thing_a, event_thing_b, event_thing_c,
    event_thing_c_created, event_thing_c_destroyed, event_thing_c_modified,
    event_thing_c_used,
    # Event-Actor
    event_actor_a_refers, event_actor_b, event_actor_c, event_actor_d,
    event_actor_d_brought_into_existence, event_actor_d_took_out_of_existence,
    # Event-Event
    event_event_a_refers, event_event_b, event_event_has_met,
    event_event_c, event_event_d,
    # Event-Concept
    event_concept_a,
    # Concept-*
    concept_place_a, concept_thing_a, concept_actor_a, concept_event_a,
    concept_concept_a, concept_concept_b,
    # Event-Time
    event_time_a,
    # Thing-Time
    thing_time_a, thing_time_a_created, thing_time_a_modified,
    thing_time_a_moved, thing_time_a_destroyed,
    # Actor-Time
    actor_time_a, actor_time_b, actor_time_b_born,
    actor_time_c, actor_time_c_died,
]

# ==========================================================================
# COMBINATORIAL EXPANSION ENGINE
# ==========================================================================

from itertools import product


# ─── REUSABLE LAYER COMPONENTS ────────────────────────────────────────────────

COPY = [
    [],
    [S("P130_shows_features_of", "E70", "E70", rec=True)],
    [S("P130i_features_are_also_found_on", "E70", "E70", rec=True)],
]

PW_FWD = [
    [],
    [S("P46_is_composed_of", "E18", "E18", rec=True)],
    [S("P106_is_composed_of", "E90", "E90", rec=True)],
    [S("P148_has_component", "E89", "E89", rec=True)],
]

PW_INV = [
    [],
    [S("P46i_forms_part_of", "E18", "E18", rec=True)],
    [S("P106i_forms_part_of", "E90", "E90", rec=True)],
    [S("P148i_is_component_of", "E89", "E89", rec=True)],
]

# Range suffixes by FC
SUF_PLACE_FWD = [[], [S("P89i_contains", "E53", "E53", rec=True)]]
SUF_PLACE_INV = [[], [S("P89_falls_within", "E53", "E53", rec=True)]]
SUF_EVENT_FWD = [[], [S("P9_consists_of", "E5", "E5", rec=True)]]
SUF_EVENT_INV = [[], [S("P9i_forms_part_of", "E5", "E5", rec=True)]]
SUF_ACTOR_FWD = [[], [S("P107_has_current_or_former_member", "E39", "E39", rec=True)]]
SUF_ACTOR_INV = [[], [S("P107i_is_current_or_former_member_of", "E39", "E39", rec=True)]]
SUF_THING_FWD = [
    [],
    [S("P46_is_composed_of", "C1", "C1", rec=True)],
    [S("P106_is_composed_of", "E90", "E90", rec=True)],
    [S("P148_has_component", "E89", "E89", rec=True)],
]
SUF_THING_INV = [
    [],
    [S("P46i_forms_part_of", "E18", "E18", rec=True)],
    [S("P106i_forms_part_of", "E90", "E90", rec=True)],
    [S("P148i_is_component_of", "E89", "E89", rec=True)],
]

# Domain prefixes for non-Thing FCs
PREFIX_PLACE = [[], [S("P89i_contains", "E53", "E53", rec=True)]]
PREFIX_ACTOR = [[], [S("P107_has_current_or_former_member", "E39", "E39", rec=True)]]
PREFIX_EVENT_FWD = [[], [S("P9_consists_of", "E5", "E5", rec=True)]]
PREFIX_EVENT_INV = [[], [S("P9i_forms_part_of", "E5", "E5", rec=True)]]

# CRMdig prefix
DIG = [
    S("F1_is_derivative_of", "D1", "D1", rec=True),
    S("L11i_was_output_of", "D1", "D7"),
    S("P9i_forms_part_of", "D7", "D2", rec=True),
    S("L1_digitized", "D2", "E18"),
]


# ─── CORE REFERENCING MECHANISMS ──────────────────────────────────────────────

def _ref_cores(target):
    """Standard 'refers to' cores: P62, P67, P128+P67."""
    return [
        [S("P62_depicts", "E24", target)],
        [S("P67_refers_to", "E89", target)],
        [S("P128_carries", "E18", "E73"), S("P67_refers_to", "E73", target)],
    ]


def _ref_cores_place():
    """Extended cores for Place: includes Physical Feature variant."""
    return _ref_cores("E53") + [
        [S("P62_depicts", "E24", "E26"),
         S("P53_has_former_or_current_location", "E26", "E53")],
        [S("P67_refers_to", "E89", "E26"),
         S("P53_has_former_or_current_location", "E26", "E53")],
        [S("P128_carries", "E18", "E73"),
         S("P67_refers_to", "E73", "E26"),
         S("P53_has_former_or_current_location", "E26", "E53")],
    ]


def _inv_ref_cores():
    """Inverse 'is referred to by' cores: P67i, P62i."""
    return [
        [S("P67i_is_referred_to_by", "E1", "E89")],
        [S("P62i_is_depicted_by", "E1", "E24")],
    ]


# ─── MEDIATION STEPS (for non-Thing domain FCs) ──────────────────────────────

# Place -> Thing: "location_of" mediation steps
PLACE_LOC_MEDIATION = [
    [S("P53i_is_former_or_current_location_of", "E53", "E24")],
]

# Place -> Thing: "witnessed" (event) mediation
PLACE_EVENT_MEDIATION_CREATION = [
    [S("P7i_witnessed", "E53", "E5"),
     S("P9_consists_of", "E5", "E5", rec=True),
     S("P94_has_created", "E65", "C1")],
]

PLACE_EVENT_MEDIATION_PRODUCTION = [
    [S("P7i_witnessed", "E53", "E5"),
     S("P9_consists_of", "E5", "E5", rec=True),
     S("P108_has_produced", "E12", "E24")],
]

# Place -> Actor: "residence" mediation
PLACE_RESIDENCE_MEDIATION = [
    [S("P74i_is_current_or_former_residence_of", "E53", "E39"),
     S("P49i_is_former_or_current_keeper_of", "E39", "E24")],
    [S("P74i_is_current_or_former_residence_of", "E53", "E39"),
     S("P51i_is_former_or_current_owner_of", "E39", "E24")],
]

# Actor -> Thing: "performed + created" mediation
ACTOR_CREATION_MEDIATION = [
    [S("P14i_performed", "E39", "E7"),
     S("P9_consists_of", "E7", "E5", rec=True),
     S("P92_brought_into_existence", "E63", "C1")],
]

# Event -> Thing: "created" mediation
EVENT_CREATION_MEDIATION = [
    [S("P92_brought_into_existence", "E63", "C1")],
]


# ─── BUILDER UTILITIES ───────────────────────────────────────────────────────

def _lbl(steps):
    return "/".join(
        s.property.split("_")[0] + ("*" if s.recursive else "")
        for s in steps
    )


def _cross(*layer_lists):
    """Cartesian product of multiple layer lists, concatenating steps."""
    return [
        sum(combo, [])
        for combo in product(*layer_lists)
    ]


def _build_paths(fr_id, layers_list, extras=None):
    """Build path list from fully enumerated step lists."""
    paths = []
    n = 1
    for steps in layers_list:
        paths.append(P(f"{fr_id}_{n:03d}", _lbl(steps), *steps))
        n += 1
    for ep in (extras or []):
        paths.append(P(f"{fr_id}_{n:03d}", _lbl(ep), *ep))
        n += 1
    return paths


def _fr(fr_id, label, dfc, rfc, dcls, rcls, paths, spec=None):
    return FR(fr_id, label, dfc, rfc, dcls, rcls, paths, spec)


# ─── THING -> X "refers to or is about" ──────────────────────────────────────

def _thing_prefixes():
    """copy x part-whole for Thing domain."""
    return _cross(COPY, PW_FWD)


def expand_thing_place_a():
    """Thing -> Place: refers to or is about."""
    main = _cross(_thing_prefixes(), _ref_cores_place(), SUF_PLACE_FWD)
    dig_cores = [
        [S("P53_has_former_or_current_location", "E18", "E53")],
        [S("P62_depicts", "E24", "E53")],
        [S("P128_carries", "E18", "E73"), S("P67_refers_to", "E73", "E53")],
    ]
    dig = _cross([DIG], dig_cores, SUF_PLACE_FWD)
    paths = _build_paths("tp_a", main, dig)
    return _fr("thing_place_a", "refers to or is about",
               "Thing", "Place", "C1", "E53", paths)


def expand_thing_thing_b():
    """Thing -> Thing: refers to or is about."""
    main = _cross(_thing_prefixes(), _ref_cores("C1"), SUF_THING_FWD)
    dig = _cross([DIG], [[]], SUF_THING_FWD)
    paths = _build_paths("tt_b", main, dig)
    return _fr("thing_thing_b", "refers to or is about",
               "Thing", "Thing", "C1", "C1", paths)


def expand_thing_actor_c():
    """Thing -> Actor: refers to or is about."""
    main = _cross(_thing_prefixes(), _ref_cores("E39"), SUF_ACTOR_FWD)
    paths = _build_paths("ta_c", main)
    return _fr("thing_actor_c", "refers to or is about",
               "Thing", "Actor", "C1", "E39", paths)


def expand_thing_event_a():
    """Thing -> Event: refers to or is about."""
    main = _cross(_thing_prefixes(), _ref_cores("E5"), SUF_EVENT_INV)
    dig_cores = [
        [S("P62_depicts", "E24", "E5")],
        [S("P128_carries", "E18", "E73"), S("P67_refers_to", "E73", "E5")],
    ]
    dig = _cross([DIG], dig_cores, SUF_EVENT_INV)
    paths = _build_paths("te_a", main, dig)
    return _fr("thing_event_a", "refers to or is about",
               "Thing", "Event", "C1", "E5", paths)


# ─── THING -> X "is referred to by / at" ─────────────────────────────────────

def _thing_inv_prefixes():
    """copy x part-whole-inv for the object side of 'is referred to by'."""
    return _cross(COPY, PW_INV)


def _inv_ref_obj_suffixes():
    """Object-side continuation after inv ref core."""
    return [
        [S("P148i_is_component_of", "E89", "E89", rec=True)],
        [S("P128i_is_carried_by", "E73", "E24"),
         S("P46i_forms_part_of", "E24", "E24", rec=True)],
    ]


def expand_thing_thing_c():
    """Thing -> Thing: is referred to by."""
    main = _cross(_thing_inv_prefixes(), _inv_ref_cores(), _inv_ref_obj_suffixes())
    paths = _build_paths("tt_c", main)
    return _fr("thing_thing_c", "is referred to by",
               "Thing", "Thing", "C1", "C1", paths)


def expand_thing_place_b():
    """Thing -> Place: is referred to at."""
    inv_place_continuations = [
        [S("P94i_was_created_by", "E89", "E65"),
         S("P9i_forms_part_of", "E65", "E5", rec=True),
         S("P7_took_place_at", "E5", "E53")],
        [S("P128i_is_carried_by", "E73", "E24"),
         S("P53_has_former_or_current_location", "E24", "E53")],
        [S("P128i_is_carried_by", "E73", "E24"),
         S("P108i_was_produced_by", "E24", "E12"),
         S("P9i_forms_part_of", "E12", "E5", rec=True),
         S("P7_took_place_at", "E5", "E53")],
    ]
    main = _cross(_thing_inv_prefixes(), _inv_ref_cores(),
                   inv_place_continuations, SUF_PLACE_INV)
    paths = _build_paths("tp_b", main)
    return _fr("thing_place_b", "is referred to at",
               "Thing", "Place", "C1", "E53", paths)


def expand_thing_actor_b():
    """Thing -> Actor: is referred to by."""
    inv_actor_continuations = [
        [S("P94i_was_created_by", "E89", "E65"),
         S("P9i_forms_part_of", "E65", "E7", rec=True),
         S("P14_carried_out_by", "E7", "E39")],
        [S("P128i_is_carried_by", "E73", "E24"),
         S("P46i_forms_part_of", "E24", "E24", rec=True),
         S("P108i_was_produced_by", "E24", "E12"),
         S("P9i_forms_part_of", "E12", "E5", rec=True),
         S("P14_carried_out_by", "E7", "E39")],
    ]
    main = _cross(_thing_inv_prefixes(), _inv_ref_cores(),
                   inv_actor_continuations, SUF_ACTOR_INV)
    paths = _build_paths("ta_b", main)
    return _fr("thing_actor_b", "is referred to by",
               "Thing", "Actor", "C1", "E39", paths)


def expand_thing_event_b():
    """Thing -> Event: is referred to at."""
    inv_event_continuations = [
        [S("P94i_was_created_by", "E89", "E65"),
         S("P9i_forms_part_of", "E65", "E5", rec=True)],
        [S("P128i_is_carried_by", "E73", "E24"),
         S("P108i_was_produced_by", "E24", "E12"),
         S("P9i_forms_part_of", "E12", "E5", rec=True)],
    ]
    main = _cross(_thing_inv_prefixes(), _inv_ref_cores(), inv_event_continuations)
    paths = _build_paths("te_b", main)
    return _fr("thing_event_b", "is referred to at",
               "Thing", "Event", "C1", "E5", paths)


# ─── PLACE -> X "refers to" ──────────────────────────────────────────────────
# Pattern: P89i* (place prefix) -> mediation -> ref core -> suffix
# Three mediation branches:
#   1. location_of -> (P62/P128+P67) -> target
#   2. witnessed -> event -> created -> (P62/P67/P128+P67) -> target
#   3. residence -> keeper/owner -> (P62/P128+P67) -> target

def _place_to_thing_branches(target, target_suffix):
    """All Place -> target branches via location, event, and residence mediation."""
    results = []

    # Branch 1: location_of -> Thing -> ref core -> target
    loc_ref = _cross(PLACE_LOC_MEDIATION, _ref_cores(target), target_suffix)
    results.extend(loc_ref)

    # Branch 2: witnessed -> event -> created -> ref core -> target
    evt_creation_ref = _cross(
        PLACE_EVENT_MEDIATION_CREATION,
        [[S("P67_refers_to", "E89", target)]],
        target_suffix
    )
    results.extend(evt_creation_ref)

    evt_production_ref = _cross(
        PLACE_EVENT_MEDIATION_PRODUCTION,
        _ref_cores(target),
        target_suffix
    )
    results.extend(evt_production_ref)

    # Branch 3: residence -> keeper/owner -> ref core -> target
    res_ref = _cross(PLACE_RESIDENCE_MEDIATION, _ref_cores(target), target_suffix)
    results.extend(res_ref)

    return results


def expand_place_thing_a():
    """Place -> Thing: refers to."""
    branches = _place_to_thing_branches("C1", SUF_THING_FWD)
    main = _cross(PREFIX_PLACE, branches)
    paths = _build_paths("pt_a", main)
    return _fr("place_thing_a", "refers to",
               "Place", "Thing", "E53", "C1", paths)


def expand_place_actor_a():
    """Place -> Actor: refers to."""
    branches = _place_to_thing_branches("E39", SUF_ACTOR_FWD)
    main = _cross(PREFIX_PLACE, branches)
    paths = _build_paths("pa_a", main)
    return _fr("place_actor_a", "refers to",
               "Place", "Actor", "E53", "E39", paths)


def expand_place_event_b():
    """Place -> Event: refers to."""
    # Place ref to Event uses P46* on the Thing before ref core
    loc_ref = _cross(
        PLACE_LOC_MEDIATION,
        [[S("P46_is_composed_of", "E18", "E18", rec=True)], []],
        [
            [S("P62_depicts", "E24", "E5")],
            [S("P128_carries", "E18", "E73"), S("P67_refers_to", "E73", "E5")],
        ],
        SUF_EVENT_FWD,
    )
    main = _cross(PREFIX_PLACE, loc_ref)
    paths = _build_paths("pe_b", main)
    return _fr("place_event_b", "refers to",
               "Place", "Event", "E53", "E5", paths)


# ─── PLACE -> X "is referred to by / at" ─────────────────────────────────────

def _place_inv_to_thing_branches():
    """Inverse Place: P89* prefix -> P67i/P62i -> copy -> continuation."""
    inv_continuations = [
        [S("P94i_was_created_by", "E89", "E65"),
         S("P9i_forms_part_of", "E65", "E5", rec=True)],
        [S("P128i_is_carried_by", "E73", "E24"),
         S("P46i_forms_part_of", "E24", "E24", rec=True),
         S("P108i_was_produced_by", "E24", "E12"),
         S("P9i_forms_part_of", "E12", "E5", rec=True)],
    ]
    return _cross(COPY, _inv_ref_cores(), inv_continuations)


def expand_place_thing_b():
    """Place -> Thing: is referred to by."""
    branches = _place_inv_to_thing_branches()
    main = _cross(SUF_PLACE_INV, branches)
    paths = _build_paths("pt_b", main)
    return _fr("place_thing_b", "is referred to by",
               "Place", "Thing", "E53", "C1", paths)


def expand_place_actor_b():
    """Place -> Actor: is referred to by."""
    branches = _place_inv_to_thing_branches()
    main = _cross(SUF_PLACE_INV, branches)
    paths = _build_paths("pab", main)
    return _fr("place_actor_b", "is referred to by",
               "Place", "Actor", "E53", "E39", paths)


def expand_place_event_c():
    """Place -> Event: is referred to at."""
    branches = _place_inv_to_thing_branches()
    main = _cross(SUF_PLACE_INV, branches)
    paths = _build_paths("pec", main)
    return _fr("place_event_c", "is referred to at",
               "Place", "Event", "E53", "E5", paths)


# ─── ACTOR -> X "refers to" ──────────────────────────────────────────────────
# Pattern: P107* (group prefix) -> P14i performed -> P9* event -> P92 created
#          -> copy x pw x ref_core -> target suffix

def expand_actor_place_a():
    """Actor -> Place: refers to."""
    inner = _cross(COPY, _ref_cores_place(), SUF_PLACE_FWD)
    main = _cross(PREFIX_ACTOR, ACTOR_CREATION_MEDIATION, inner)
    paths = _build_paths("apl_a", main)
    return _fr("actor_place_a", "refers to",
               "Actor", "Place", "E39", "E53", paths)


def expand_actor_thing_a():
    """Actor -> Thing: refers to."""
    inner = _cross(COPY, _ref_cores("C1"), SUF_THING_FWD)
    main = _cross(PREFIX_ACTOR, ACTOR_CREATION_MEDIATION, inner)
    paths = _build_paths("at_a", main)
    return _fr("actor_thing_a", "refers to",
               "Actor", "Thing", "E39", "C1", paths)


def expand_actor_actor_a():
    """Actor -> Actor: refers to."""
    inner = _cross(COPY, _ref_cores("E39"), SUF_ACTOR_FWD)
    main = _cross(PREFIX_ACTOR, ACTOR_CREATION_MEDIATION, inner)
    paths = _build_paths("aa_a", main)
    return _fr("actor_actor_a", "refers to",
               "Actor", "Actor", "E39", "E39", paths)


def expand_actor_event_a():
    """Actor -> Event: refers to."""
    inner = _cross(COPY, _ref_cores("E5"), SUF_EVENT_INV)
    main = _cross(PREFIX_ACTOR, ACTOR_CREATION_MEDIATION, inner)
    paths = _build_paths("ae_a", main)
    return _fr("actor_event_a", "refers to",
               "Actor", "Event", "E39", "E5", paths)


# ─── ACTOR -> X "is referred to by / at" ─────────────────────────────────────

def _actor_inv_branches():
    """Actor inv: P107i* prefix -> P67i/P62i -> copy -> continuation."""
    inv_continuations = [
        [S("P94i_was_created_by", "E89", "E65"),
         S("P9i_forms_part_of", "E65", "E5", rec=True),
         S("P14_carried_out_by", "E7", "E39")],
        [S("P128i_is_carried_by", "E73", "E24"),
         S("P46i_forms_part_of", "E24", "E24", rec=True),
         S("P108i_was_produced_by", "E24", "E12"),
         S("P9i_forms_part_of", "E12", "E5", rec=True),
         S("P14_carried_out_by", "E7", "E39")],
    ]
    return _cross(COPY, _inv_ref_cores(), inv_continuations)


def expand_actor_place_b():
    """Actor -> Place: is referred to at."""
    place_cont = [
        [S("P7_took_place_at", "E5", "E53")],
    ]
    branches = _cross(_actor_inv_branches(), place_cont, SUF_PLACE_INV)
    main = _cross(SUF_ACTOR_INV, branches)
    paths = _build_paths("apl_b", main)
    return _fr("actor_place_b", "is referred to at",
               "Actor", "Place", "E39", "E53", paths)


def expand_actor_thing_b():
    """Actor -> Thing: is referred to by."""
    branches = _actor_inv_branches()
    main = _cross(SUF_ACTOR_INV, branches)
    paths = _build_paths("at_b", main)
    return _fr("actor_thing_b", "is referred to by",
               "Actor", "Thing", "E39", "C1", paths)


def expand_actor_actor_c():
    """Actor -> Actor: is referred to by."""
    branches = _actor_inv_branches()
    main = _cross(SUF_ACTOR_INV, branches)
    paths = _build_paths("aa_c", main)
    return _fr("actor_actor_c", "is referred to by",
               "Actor", "Actor", "E39", "E39", paths)


def expand_actor_event_b():
    """Actor -> Event: is referred to at."""
    branches = _actor_inv_branches()
    main = _cross(SUF_ACTOR_INV, branches)
    paths = _build_paths("ae_b", main)
    return _fr("actor_event_b", "is referred to at",
               "Actor", "Event", "E39", "E5", paths)


# ─── EVENT -> X "refers to" ──────────────────────────────────────────────────
# Pattern: P9* (event prefix) -> P92 created -> copy x pw x ref core -> suffix

def expand_event_place_a():
    """Event -> Place: refers to."""
    inner = _cross(COPY, PW_FWD, _ref_cores_place(), SUF_PLACE_FWD)
    main = _cross(PREFIX_EVENT_FWD, EVENT_CREATION_MEDIATION, inner)
    paths = _build_paths("epl_a", main)
    return _fr("event_place_a", "refers to",
               "Event", "Place", "E5", "E53", paths)


def expand_event_thing_a():
    """Event -> Thing: refers to or is about."""
    inner = _cross(COPY, PW_FWD, _ref_cores("C1"), SUF_THING_FWD)
    main = _cross(PREFIX_EVENT_FWD, EVENT_CREATION_MEDIATION, inner)
    paths = _build_paths("et_a", main)
    return _fr("event_thing_a", "refers to or is about",
               "Event", "Thing", "E5", "C1", paths)


def expand_event_actor_a():
    """Event -> Actor: refers to or is about."""
    inner = _cross(COPY, _ref_cores("E39"), SUF_ACTOR_FWD)
    main = _cross(PREFIX_EVENT_FWD, EVENT_CREATION_MEDIATION, inner)
    paths = _build_paths("ea_a", main)
    return _fr("event_actor_a_refers", "refers to or is about",
               "Event", "Actor", "E5", "E39", paths)


def expand_event_event_a():
    """Event -> Event: refers to or is about."""
    inner = _cross(COPY, _ref_cores("E5"), SUF_EVENT_INV)
    main = _cross(PREFIX_EVENT_FWD, EVENT_CREATION_MEDIATION, inner)
    paths = _build_paths("ee_a", main)
    return _fr("event_event_a_refers", "refers to or is about",
               "Event", "Event", "E5", "E5", paths)


# ─── EVENT -> X "is referred to by / at" ─────────────────────────────────────

def _event_inv_branches():
    """Event inv: P9i* prefix -> P67i/P62i -> copy -> continuation."""
    inv_continuations = [
        [S("P94i_was_created_by", "E89", "E65"),
         S("P9i_forms_part_of", "E65", "E5", rec=True)],
        [S("P128i_is_carried_by", "E73", "E24"),
         S("P46i_forms_part_of", "E24", "E24", rec=True),
         S("P108i_was_produced_by", "E24", "E12"),
         S("P9i_forms_part_of", "E12", "E5", rec=True)],
    ]
    return _cross(COPY, _inv_ref_cores(), inv_continuations)


def expand_event_place_b():
    """Event -> Place: is referred to at."""
    place_cont = [
        [S("P7_took_place_at", "E5", "E53")],
    ]
    branches = _cross(_event_inv_branches(), place_cont, SUF_PLACE_INV)
    main = _cross(PREFIX_EVENT_INV, branches)
    paths = _build_paths("epl_b", main)
    return _fr("event_place_b", "is referred to at",
               "Event", "Place", "E5", "E53", paths)


def expand_event_thing_b():
    """Event -> Thing: is referred to by."""
    branches = _event_inv_branches()
    main = _cross(PREFIX_EVENT_INV, branches)
    paths = _build_paths("et_b", main)
    return _fr("event_thing_b", "is referred to by",
               "Event", "Thing", "E5", "C1", paths)


def expand_event_actor_b():
    """Event -> Actor: is referred to by."""
    branches = _event_inv_branches()
    main = _cross(PREFIX_EVENT_INV, branches)
    paths = _build_paths("ea_b", main)
    return _fr("event_actor_b", "is referred to by",
               "Event", "Actor", "E5", "E39", paths)


def expand_event_event_b():
    """Event -> Event: is referred to at."""
    branches = _event_inv_branches()
    main = _cross(PREFIX_EVENT_INV, branches)
    paths = _build_paths("ee_b", main)
    return _fr("event_event_b", "is referred to at",
               "Event", "Event", "E5", "E5", paths)


# ─── PART-WHOLE EXPANSION (remaining FRs with P46 needing P106/P148) ─────────

def expand_thing_thing_d():
    """Thing -> Thing: from (history).
    TR-429: {P46i | P106i | P148i} prefix, plus P123i and P31i branches
    with {P46i | P106i | P148i} suffix on the P124 branch."""
    # Direct part-whole paths (3 variants, skip empty)
    pw_paths = [pw for pw in PW_INV if pw]

    # P123i -> P9i* -> P124 -> {pw suffix}
    transform_base = [S("P123i_resulted_from", "E18", "E81"),
                      S("P9i_forms_part_of", "E81", "E81", rec=True),
                      S("P124_transformed", "E81", "C1")]
    transform_paths = _cross([[transform_base[i] for i in range(3)]],
                              [pw for pw in PW_INV])

    # P31i -> P9i* -> P110 (augmented)
    modify_aug = [[S("P31i_was_modified_by", "E24", "E11"),
                   S("P9i_forms_part_of", "E11", "E7", rec=True),
                   S("P110_augmented", "E7", "C1")]]

    # P31i -> P9i* -> P112 (diminished)
    modify_dim = [[S("P31i_was_modified_by", "E24", "E11"),
                   S("P9i_forms_part_of", "E11", "E7", rec=True),
                   S("P112_diminished", "E7", "E18")]]

    all_paths = pw_paths + transform_paths + modify_aug + modify_dim
    paths = _build_paths("tt_d", all_paths)
    return _fr("thing_thing_d", "from", "Thing", "Thing", "C1", "C1", paths)


def expand_thing_thing_d_is_part_of():
    """Thing -> Thing: is part of (specialization of from).
    TR-429: {P46i | P106i | P148i}."""
    pw_paths = [pw for pw in PW_INV if pw]
    paths = _build_paths("tt_d_ip", pw_paths)
    return _fr("thing_thing_d_is_part_of", "is part of",
               "Thing", "Thing", "C1", "C1", paths,
               spec="thing_thing_d")


def expand_thing_thing_e():
    """Thing -> Thing: has part.
    TR-429: {P46 | P106 | P148}, plus P108i -> P9i* -> P111 branch."""
    pw_paths = [pw for pw in PW_FWD if pw]

    addition_paths = [[
        S("P108i_was_produced_by", "E24", "E79"),
        S("P9i_forms_part_of", "E79", "E79", rec=True),
        S("P111_added", "E79", "E18"),
    ]]

    all_paths = pw_paths + addition_paths
    paths = _build_paths("tt_e", all_paths)
    return _fr("thing_thing_e", "has part",
               "Thing", "Thing", "C1", "C1", paths)


def expand_event_thing_c():
    """Event -> Thing: has met.
    TR-429: P9* -> P12 -> {P46 | P106 | P148}."""
    main = _cross(
        PREFIX_EVENT_FWD,
        [[S("P12_occurred_in_the_presence_of", "E5", "C1")]],
        SUF_THING_FWD,
    )
    paths = _build_paths("et_c", main)
    return _fr("event_thing_c", "has met",
               "Event", "Thing", "E5", "C1", paths)


def expand_event_thing_c_created():
    """Event -> Thing: created.
    TR-429: P9* -> P92 -> {P46 | P106 | P148}."""
    main = _cross(
        PREFIX_EVENT_FWD,
        [[S("P92_brought_into_existence", "E63", "C1")]],
        SUF_THING_FWD,
    )
    paths = _build_paths("et_c_cr", main)
    return _fr("event_thing_c_created", "created",
               "Event", "Thing", "E5", "C1", paths,
               spec="event_thing_c")


def expand_event_thing_c_destroyed():
    """Event -> Thing: destroyed.
    TR-429: P9* -> P93 -> {P46 | P106 | P148}."""
    main = _cross(
        PREFIX_EVENT_FWD,
        [[S("P93_took_out_of_existence", "E6", "C1")]],
        SUF_THING_FWD,
    )
    paths = _build_paths("et_c_de", main)
    return _fr("event_thing_c_destroyed", "destroyed",
               "Event", "Thing", "E5", "C1", paths,
               spec="event_thing_c")


def expand_event_thing_c_modified():
    """Event -> Thing: modified.
    TR-429: P9* -> P31 -> {P46 | P106 | P148}."""
    main = _cross(
        PREFIX_EVENT_FWD,
        [[S("P31_has_modified", "E11", "C1")]],
        SUF_THING_FWD,
    )
    paths = _build_paths("et_c_mo", main)
    return _fr("event_thing_c_modified", "modified",
               "Event", "Thing", "E5", "C1", paths,
               spec="event_thing_c")


def expand_event_thing_c_used():
    """Event -> Thing: used.
    TR-429: P9* -> P16 -> {P46 | P106 | P148}, plus P9* -> P125."""
    use_specific = _cross(
        PREFIX_EVENT_FWD,
        [[S("P16_used_specific_object", "E7", "C1")]],
        SUF_THING_FWD,
    )
    use_type = _cross(
        PREFIX_EVENT_FWD,
        [[S("P125_used_object_of_type", "E7", "E55")]],
    )
    paths = _build_paths("et_c_us", use_specific, use_type)
    return _fr("event_thing_c_used", "used",
               "Event", "Thing", "E5", "C1", paths,
               spec="event_thing_c")


def expand_concept_thing_a():
    """Concept -> Thing: is type of.
    TR-429: P127i* -> P2i -> {P46i | P106i | P148i}."""
    main = _cross(
        [[], [S("P127i_has_narrower_term", "E55", "E55", rec=True)]],
        [[S("P2i_is_type_of", "E55", "C1")]],
        [pw for pw in PW_INV],
    )
    paths = _build_paths("ct_a", main)
    return _fr("concept_thing_a", "is type of",
               "Concept", "Thing", "E55", "C1", paths)


def expand_place_thing_c():
    """Place -> Thing: has met.
    TR-429: P89i* -> {location branch with P46*, event branch with {P46i|P106i|P148i}}."""
    # Location branch: P53i -> P46* only (physical)
    loc_branch = _cross(
        [[S("P53i_is_former_or_current_location_of", "E53", "E24")]],
        [[S("P46_is_composed_of", "E24", "E24", rec=True)], []],
    )

    # Event branch: P7i -> P9* -> P92 -> {P46|P106|P148}
    evt_branch = _cross(
        [[S("P7i_witnessed", "E53", "E5"),
          S("P9_consists_of", "E5", "E5", rec=True),
          S("P92_brought_into_existence", "E63", "C1")]],
        SUF_THING_FWD,
    )

    # Residence branch: P74i -> P49i -> P46* only (physical)
    res_branch = _cross(
        [[S("P74i_is_current_or_former_residence_of", "E53", "E39"),
          S("P49i_is_former_or_current_keeper_of", "E39", "E24")]],
        [[S("P46_is_composed_of", "E24", "E24", rec=True)], []],
    )

    all_branches = loc_branch + evt_branch + res_branch
    main = _cross(PREFIX_PLACE, all_branches)
    paths = _build_paths("pt_c", main)
    return _fr("place_thing_c", "has met",
               "Place", "Thing", "E53", "C1", paths)


def expand_actor_thing_c():
    """Actor -> Thing: is origin of.
    TR-429: P107* -> {created | keeper | owner} -> {P46|P106|P148}."""
    # Created branch: P14i -> P9* -> P92 -> {P46|P106|P148}
    created = _cross(
        [[S("P14i_performed", "E39", "E7"),
          S("P9_consists_of", "E7", "E5", rec=True),
          S("P92_brought_into_existence", "E63", "C1")]],
        SUF_THING_FWD,
    )

    # Keeper: P49i -> P46* only (physical)
    keeper = _cross(
        [[S("P49i_is_former_or_current_keeper_of", "E39", "E24")]],
        [[S("P46_is_composed_of", "E24", "E24", rec=True)], []],
    )

    # Owner: P51i -> P46* only (physical)
    owner = _cross(
        [[S("P51i_is_former_or_current_owner_of", "E39", "E24")]],
        [[S("P46_is_composed_of", "E24", "E24", rec=True)], []],
    )

    all_branches = created + keeper + owner
    main = _cross(PREFIX_ACTOR, all_branches)
    paths = _build_paths("at_c", main)
    return _fr("actor_thing_c", "is origin of",
               "Actor", "Thing", "E39", "C1", paths)


# ─── MASTER EXPANSION ────────────────────────────────────────────────────────

def expand_all():
    """Generate all fully expanded FRs, keyed by fr_id."""
    expanded = {}

    # Thing -> X forward
    expanded["thing_place_a"] = expand_thing_place_a()
    expanded["thing_thing_b"] = expand_thing_thing_b()
    expanded["thing_actor_c"] = expand_thing_actor_c()
    expanded["thing_event_a"] = expand_thing_event_a()

    # Thing -> X inverse
    expanded["thing_thing_c"] = expand_thing_thing_c()
    expanded["thing_place_b"] = expand_thing_place_b()
    expanded["thing_actor_b"] = expand_thing_actor_b()
    expanded["thing_event_b"] = expand_thing_event_b()

    # Place -> X forward
    expanded["place_thing_a"] = expand_place_thing_a()
    expanded["place_actor_a"] = expand_place_actor_a()
    expanded["place_event_b"] = expand_place_event_b()

    # Place -> X inverse
    expanded["place_thing_b"] = expand_place_thing_b()
    expanded["place_actor_b"] = expand_place_actor_b()
    expanded["place_event_c"] = expand_place_event_c()

    # Actor -> X forward
    expanded["actor_place_a"] = expand_actor_place_a()
    expanded["actor_thing_a"] = expand_actor_thing_a()
    expanded["actor_actor_a"] = expand_actor_actor_a()
    expanded["actor_event_a"] = expand_actor_event_a()

    # Actor -> X inverse
    expanded["actor_place_b"] = expand_actor_place_b()
    expanded["actor_thing_b"] = expand_actor_thing_b()
    expanded["actor_actor_c"] = expand_actor_actor_c()
    expanded["actor_event_b"] = expand_actor_event_b()

    # Event -> X forward
    expanded["event_place_a"] = expand_event_place_a()
    expanded["event_thing_a"] = expand_event_thing_a()
    expanded["event_actor_a_refers"] = expand_event_actor_a()
    expanded["event_event_a_refers"] = expand_event_event_a()

    # Event -> X inverse
    expanded["event_place_b"] = expand_event_place_b()
    expanded["event_thing_b"] = expand_event_thing_b()
    expanded["event_actor_b"] = expand_event_actor_b()
    expanded["event_event_b"] = expand_event_event_b()

    # Part-whole expansions (remaining FRs with P46 needing P106/P148)
    expanded["thing_thing_d"] = expand_thing_thing_d()
    expanded["thing_thing_d_is_part_of"] = expand_thing_thing_d_is_part_of()
    expanded["thing_thing_e"] = expand_thing_thing_e()
    expanded["event_thing_c"] = expand_event_thing_c()
    expanded["event_thing_c_created"] = expand_event_thing_c_created()
    expanded["event_thing_c_destroyed"] = expand_event_thing_c_destroyed()
    expanded["event_thing_c_modified"] = expand_event_thing_c_modified()
    expanded["event_thing_c_used"] = expand_event_thing_c_used()
    expanded["concept_thing_a"] = expand_concept_thing_a()
    expanded["place_thing_c"] = expand_place_thing_c()
    expanded["actor_thing_c"] = expand_actor_thing_c()

    return expanded


def build_fully_expanded():
    """Replace compact FRs with expanded versions; keep others as-is."""
    expanded = expand_all()
    result = []
    for fr in ALL_FRS:
        if fr.id in expanded:
            result.append(expanded[fr.id])
        else:
            result.append(fr)
    return result


# ─── SPARQL GENERATION BY ENTITY ─────────────────────────────────────────────

def generate_sparql_by_entity(all_frs=None):
    """Generate SPARQL grouped by FC pair following TR-429 ordering."""
    from collections import defaultdict

    if all_frs is None:
        all_frs = build_fully_expanded()

    FC_ORDER = ["Thing", "Place", "Actor", "Event", "Concept", "Time"]
    FC_HEADERS = {
        "Thing": "THING",
        "Place": "PLACE",
        "Actor": "ACTOR",
        "Event": "EVENT-TIME",
        "Concept": "CONCEPT",
        "Time": "TIME",
    }

    by_pair = defaultdict(list)
    for fr in all_frs:
        by_pair[(fr.domain_fc, fr.range_fc)].append(fr)

    lines = [
        "# " + "=" * 74,
        "# CIDOC-CRM Fundamental Relationships (TR-429)",
        "# Fully expanded SPARQL property paths",
        "# Source: Tzompanaki & Doerr, FORTH-ICS TR-429, April 2012",
        "# " + "=" * 74,
        "",
        "PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>",
        "",
    ]

    total_paths = sum(len(fr.paths) for fr in all_frs)
    lines.append(f"# Total: {len(all_frs)} FRs, {total_paths} paths")
    lines.append("")

    for dfc in FC_ORDER:
        lines.append("")
        lines.append("# " + "=" * 74)
        lines.append(f"# {FC_HEADERS[dfc]}")
        lines.append("# " + "=" * 74)
        lines.append("")

        for rfc in FC_ORDER:
            pair = (dfc, rfc)
            if pair not in by_pair:
                continue

            frs = by_pair[pair]
            pair_paths = sum(len(fr.paths) for fr in frs)

            lines.append("# " + "-" * 74)
            lines.append(f"# {dfc}-{rfc}  ({len(frs)} FRs, {pair_paths} paths)")
            lines.append("# " + "-" * 74)
            lines.append("")

            for fr in frs:
                spec_note = ""
                if fr.specialization_of:
                    spec_note = f"  [specialization of {fr.specialization_of}]"
                lines.append(f"# {fr.label}{spec_note}")
                lines.append(f"# {len(fr.paths)} paths")
                lines.append(fr.to_sparql())
                lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    all_frs = build_fully_expanded()
    total = sum(len(fr.paths) for fr in all_frs)
    print(f"FRs: {len(all_frs)}, Paths: {total}")

    sparql = generate_sparql_by_entity(all_frs)
    with open("fr_sparql_by_entity.sparql", "w") as f:
        f.write(sparql)
    print(f"SPARQL written to fr_sparql_by_entity.sparql")
