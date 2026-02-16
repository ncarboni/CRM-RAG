from dataclasses import dataclass, field


@dataclass
class Step:
    property: str
    domain: str
    range: str
    recursive: bool = False

    def to_sparql(self):
        prop = f"crm:{self.property}"
        if self.recursive:
            prop = f"{prop}*"
        return prop


@dataclass
class PropertyPath:
    id: str
    description: str
    steps: list[Step]

    def to_sparql_path(self):
        return "/".join(s.to_sparql() for s in self.steps)

    def to_sparql_pattern(self, subject="?thing", object_="?place"):
        return f"{subject} {self.to_sparql_path()} {object_} ."


@dataclass
class FundamentalRelationship:
    id: str
    label: str
    domain_fc: str
    range_fc: str
    domain_class: str
    range_class: str
    paths: list[PropertyPath]

    def to_sparql(self, subject="?thing", object_="?place"):
        blocks = []
        for p in self.paths:
            blocks.append(f"    {{ {p.to_sparql_pattern(subject, object_)} }}")
        union = "\n    UNION\n".join(blocks)
        return (
            f"# FR: {self.label} ({self.domain_fc} -> {self.range_fc})\n"
            f"SELECT DISTINCT {subject} {object_} WHERE {{\n"
            f"{union}\n"
            f"}}"
        )


thing_place_refers_to = FundamentalRelationship(
    id="thing_place_refers_to",
    label="refers to or is about",
    domain_fc="Thing",
    range_fc="Place",
    domain_class="C1.Object",
    range_class="E53_Place",
    paths=[
        PropertyPath("tp_a_01", "depicts Place", [
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_02", "refers to Place", [
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_03", "depicts feature at Place", [
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_04", "refers to feature at Place", [
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_05", "carries info that refers to feature at Place", [
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_06", "carries info that refers to Place", [
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_07", "copy (P130) -> depicts Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_08", "copy (P130) -> refers to Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_09", "copy (P130) -> depicts feature at Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_10", "copy (P130) -> refers to feature at Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_11", "copy (P130) -> carries info refers to feature at Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_12", "copy (P130) -> carries info refers to Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_13", "inverse copy (P130i) -> depicts Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_14", "inverse copy (P130i) -> refers to Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_15", "inverse copy (P130i) -> depicts feature at Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_16", "inverse copy (P130i) -> refers to feature at Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_17", "inverse copy (P130i) -> carries info refers to feature at Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_18", "inverse copy (P130i) -> carries info refers to Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_19", "part (P46) -> depicts Place", [
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_20", "part (P46) -> depicts feature at Place", [
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_21", "part (P46) -> carries info refers to feature at Place", [
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_22", "part (P46) -> carries info refers to Place", [
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_23", "part (P106) -> refers to Place", [
            Step("P106_is_composed_of", "E90_Symbolic_Object", "E90_Symbolic_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_24", "part (P106) -> refers to feature at Place", [
            Step("P106_is_composed_of", "E90_Symbolic_Object", "E90_Symbolic_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_25", "component (P148) -> refers to Place", [
            Step("P148_has_component", "E89_Propositional_Object", "E89_Propositional_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_26", "component (P148) -> refers to feature at Place", [
            Step("P148_has_component", "E89_Propositional_Object", "E89_Propositional_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_27", "copy (P130) -> part (P46) -> depicts Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_28", "copy (P130) -> part (P46) -> depicts feature at Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_29", "copy (P130) -> part (P46) -> carries info refers to Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_30", "copy (P130) -> part (P46) -> carries info refers to feature", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_31", "copy (P130) -> part (P106) -> refers to Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P106_is_composed_of", "E90_Symbolic_Object", "E90_Symbolic_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_32", "copy (P130) -> part (P106) -> refers to feature at Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P106_is_composed_of", "E90_Symbolic_Object", "E90_Symbolic_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_33", "copy (P130) -> component (P148) -> refers to Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P148_has_component", "E89_Propositional_Object", "E89_Propositional_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_34", "copy (P130) -> component (P148) -> refers to feature at Place", [
            Step("P130_shows_features_of", "E70_Thing", "E70_Thing", recursive=True),
            Step("P148_has_component", "E89_Propositional_Object", "E89_Propositional_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_35", "inverse copy (P130i) -> part (P46) -> depicts Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_36", "inverse copy (P130i) -> part (P46) -> depicts feature at Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_37", "inverse copy (P130i) -> part (P46) -> carries info refers to Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_38", "inverse copy (P130i) -> part (P46) -> carries info refers to feature", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P46_is_composed_of", "E18_Physical_Thing", "E18_Physical_Thing", recursive=True),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_39", "inverse copy (P130i) -> part (P106) -> refers to Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P106_is_composed_of", "E90_Symbolic_Object", "E90_Symbolic_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_40", "inverse copy (P130i) -> part (P106) -> refers to feature at Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P106_is_composed_of", "E90_Symbolic_Object", "E90_Symbolic_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_41", "inverse copy (P130i) -> component (P148) -> refers to Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P148_has_component", "E89_Propositional_Object", "E89_Propositional_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_42", "inverse copy (P130i) -> component (P148) -> refers to feature at Place", [
            Step("P130i_features_are_also_found_on", "E70_Thing", "E70_Thing", recursive=True),
            Step("P148_has_component", "E89_Propositional_Object", "E89_Propositional_Object", recursive=True),
            Step("P67_refers_to", "E89_Propositional_Object", "E26_Physical_Feature"),
            Step("P53_has_former_or_current_location", "E26_Physical_Feature", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_43", "digital derivative -> digitized feature at Place", [
            Step("F1_is_derivative_of", "D1_Digital_Object", "D1_Digital_Object", recursive=True),
            Step("L11i_was_output_of", "D1_Digital_Object", "D7_Digital_Machine_Event"),
            Step("P9i_forms_part_of", "D7_Digital_Machine_Event", "D2_Digitization_Process", recursive=True),
            Step("L1_digitized", "D2_Digitization_Process", "E18_Physical_Thing"),
            Step("P53_has_former_or_current_location", "E18_Physical_Thing", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_44", "digital derivative -> digitized thing depicts Place", [
            Step("F1_is_derivative_of", "D1_Digital_Object", "D1_Digital_Object", recursive=True),
            Step("L11i_was_output_of", "D1_Digital_Object", "D7_Digital_Machine_Event"),
            Step("P9i_forms_part_of", "D7_Digital_Machine_Event", "D2_Digitization_Process", recursive=True),
            Step("L1_digitized", "D2_Digitization_Process", "E18_Physical_Thing"),
            Step("P62_depicts", "E24_Physical_Human-Made_Thing", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
        PropertyPath("tp_a_45", "digital derivative -> digitized thing carries info refers to Place", [
            Step("F1_is_derivative_of", "D1_Digital_Object", "D1_Digital_Object", recursive=True),
            Step("L11i_was_output_of", "D1_Digital_Object", "D7_Digital_Machine_Event"),
            Step("P9i_forms_part_of", "D7_Digital_Machine_Event", "D2_Digitization_Process", recursive=True),
            Step("L1_digitized", "D2_Digitization_Process", "E18_Physical_Thing"),
            Step("P128_carries", "E18_Physical_Thing", "E90_Symbolic_Object"),
            Step("P67_refers_to", "E89_Propositional_Object", "E53_Place"),
            Step("P89i_contains", "E53_Place", "E53_Place", recursive=True),
        ]),
    ],
)


if __name__ == "__main__":
    print(thing_place_refers_to.to_sparql())
    print()
    print(f"Total paths: {len(thing_place_refers_to.paths)}")
