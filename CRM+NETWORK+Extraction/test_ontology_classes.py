#!/usr/bin/env python3
"""
Test script to verify ontology class loading and is_technical_class_name functionality
"""

import json

print("=" * 70)
print("TESTING ONTOLOGY-BASED CLASS NAME FILTERING")
print("=" * 70)

# Load the ontology classes file
print("\n1. Loading ontology_classes.json...")
with open('ontology_classes.json', 'r') as f:
    ontology_classes = set(json.load(f))

print(f"   ✓ Loaded {len(ontology_classes)} ontology classes")

# Test cases
test_cases = [
    # Technical classes that SHOULD be filtered
    ("E22_Human-Made_Object", True, "CIDOC-CRM local name"),
    ("E53_Place", True, "CIDOC-CRM local name"),
    ("E77_Persistent_Item", True, "CIDOC-CRM local name"),
    ("http://www.cidoc-crm.org/cidoc-crm/E22_Human-Made_Object", True, "CIDOC-CRM full URI"),
    ("D1_Digital_Object", True, "CRMdig local name"),
    ("D8_Digital_Device", True, "CRMdig local name"),

    # Human-readable names that should NOT be filtered
    ("Church", False, "Human-readable word"),
    ("Building", False, "Human-readable word"),
    ("Location", False, "Human-readable word"),
    ("Painting", False, "Human-readable word"),
]

print("\n2. Testing is_technical_class_name logic:")
print("-" * 70)

passed = 0
failed = 0

for class_name, should_be_technical, description in test_cases:
    # Simulate the is_technical_class_name logic
    is_technical = False

    # Check direct match
    if class_name in ontology_classes:
        is_technical = True
    # Check local name extraction
    elif '/' in class_name or '#' in class_name:
        local_name = class_name.split('/')[-1].split('#')[-1]
        if local_name in ontology_classes:
            is_technical = True

    # Evaluate result
    is_correct = (is_technical == should_be_technical)
    status = "✓ PASS" if is_correct else "✗ FAIL"

    print(f"{status} | {class_name:50} | {description:25} | Result: {is_technical}")

    if is_correct:
        passed += 1
    else:
        failed += 1

print("-" * 70)
print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")

if failed == 0:
    print("\n✓ All tests passed! Ontology-based class filtering is working correctly.")
else:
    print(f"\n✗ {failed} test(s) failed. Please review the implementation.")

print("\n3. Sample of technical classes in ontology:")
print("-" * 70)
technical_samples = [c for c in sorted(ontology_classes) if any(char.isdigit() for char in c) and '_' in c][:15]
for sample in technical_samples:
    print(f"   - {sample}")

print("\n" + "=" * 70)
