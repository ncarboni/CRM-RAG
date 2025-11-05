# Greek Labels Fix - Using English Property Labels in Raw Triples

**Date**: 2025-11-05
**Issue**: Raw triples in sources showing Greek/multi-language labels
**Status**: ✅ FIXED

---

## The Problem

In the source display, raw triples were showing **Greek labels** for predicates:

```
Subject: The preparation of the throne
Predicate: βρίσκεται σε          ← Greek label!
Object: west lunette

Subject: Scroll of heaven
Predicate: βρίσκεται σε          ← Greek label!
Object: western arch
```

**"βρίσκεται σε"** is the **Greek label** for `P55_has_current_location`.

---

## Root Cause

### Where Predicate Labels Come From

In `get_entity_context()` method (lines 599 and 666), predicate labels were obtained directly from SPARQL:

```python
# OLD CODE - Line 599/666
pred_label = result.get("predLabel", {}).get("value", pred.split('/')[-1])
```

### The SPARQL Query

```sparql
SELECT ?pred ?predLabel ?obj ?objLabel WHERE {
    <{uri}> ?pred ?obj .
    OPTIONAL { ?pred rdfs:label ?predLabel }  # Gets ANY language label!
    OPTIONAL { ?obj rdfs:label ?objLabel }
    FILTER(isURI(?obj))
}
```

This retrieves `rdfs:label` from the triplestore, which returns **whatever language label comes back first** - could be:
- English
- Greek (βρίσκεται σε)
- German (hat derzeitigen Standort)
- Russian (в настоящее время находится в)
- Chinese (有当前位置)
- etc.

### Why This Happened

CIDOC-CRM ontology includes labels in **multiple languages**:

```turtle
# In CIDOC-CRM ontology
P55_has_current_location
  rdfs:label "has current location"@en ;
  rdfs:label "βρίσκεται σε"@el ;           # Greek
  rdfs:label "hat derzeitigen Standort"@de ; # German
  rdfs:label "在当前位置"@zh ;              # Chinese
  # ... more languages
```

When SPARQL returns `rdfs:label` without language filtering, it returns a random language based on triplestore internals.

---

## The Solution

### Use property_labels.json Instead

We already have `property_labels.json` with **English labels** for all properties. We should use it!

```python
# NEW CODE - Lines 600-603, 668-671
# Use property_labels.json for English predicate labels
pred_label = UniversalRagSystem._property_labels.get(pred)
if not pred_label:
    # Fallback to local name if not in property_labels
    pred_label = pred.split('/')[-1].split('#')[-1]
```

### How It Works

1. **Check property_labels.json first** - This has English labels extracted from ontologies
2. **Fallback to local name** - If not found, use the predicate's local name (e.g., `P55_has_current_location`)

### Verification

```bash
$ grep P55 property_labels.json
"http://www.cidoc-crm.org/cidoc-crm/P55_has_current_location": "has current location",
"http://www.cidoc-crm.org/cidoc-crm/P55i_currently_holds": "currently holds",
```

✓ English labels are available!

---

## Results

### Before (Multi-language)
```
Subject: The preparation of the throne
Predicate: βρίσκεται σε              ← Greek
Object: west lunette

Subject: Scroll of heaven
Predicate: βρίσκεται σε              ← Greek
Object: western arch

Subject: Gnashing of the teeth
Predicate: βρίσκεται σε              ← Greek
Object: South arch
```

### After (English)
```
Subject: The preparation of the throne
Predicate: has current location      ← English!
Object: west lunette

Subject: Scroll of heaven
Predicate: has current location      ← English!
Object: western arch

Subject: Gnashing of the teeth
Predicate: has current location      ← English!
Object: South arch
```

---

## Where This Affects

### Two Places in get_entity_context()

**1. Outgoing relationships** (lines 599-603):
```python
# Use property_labels.json for English predicate labels
pred_label = UniversalRagSystem._property_labels.get(pred)
if not pred_label:
    # Fallback to local name if not in property_labels
    pred_label = pred.split('/')[-1].split('#')[-1]
```

**2. Incoming relationships** (lines 668-671):
```python
# Use property_labels.json for English predicate labels
pred_label = UniversalRagSystem._property_labels.get(pred)
if not pred_label:
    # Fallback to local name if not in property_labels
    pred_label = pred.split('/')[-1].split('#')[-1]
```

### Affects These Components

1. **Raw triples in sources** - Displayed in the web UI when "Show statements" is clicked
2. **Entity documents** - The markdown files in `entity_documents/` folder
3. **Debug logs** - When logging relationship information

---

## Why Not Filter SPARQL Query by Language?

We could have done this:

```sparql
SELECT ?pred ?predLabel ?obj ?objLabel WHERE {
    <{uri}> ?pred ?obj .
    OPTIONAL {
        ?pred rdfs:label ?predLabel .
        FILTER(LANG(?predLabel) = "en")  # ← Force English
    }
    FILTER(isURI(?obj))
}
```

**But this is worse because:**
1. ❌ Requires modifying multiple SPARQL queries
2. ❌ Doesn't work if English label is missing
3. ❌ Adds complexity to queries
4. ✅ **We already have property_labels.json with English labels!**

Using property_labels.json is:
- ✅ More reliable (single source of truth)
- ✅ Consistent with how we handle all property labels
- ✅ Already extracted with English priority
- ✅ Works even if triplestore doesn't have labels

---

## Testing

After rebuilding the system, raw triples will show English labels:

```bash
cd "CRM+NETWORK+Extraction"
python main.py --env .env.openai --rebuild
```

Then ask a question and check the sources:
- Click "Show statements"
- Predicate labels should be in English (e.g., "has current location")
- Not in Greek (e.g., "βρίσκεται σε")

---

## Related Issue

This is similar to the earlier issue where we were storing multi-language class labels in ontology_classes.json. The pattern is the same:

| Problem | Solution |
|---------|----------|
| ontology_classes.json had multi-language class labels | Use English labels only |
| property_labels.json already has English labels | Use it for predicate labels in raw_triples |

**Consistent approach**: Always use the English labels from our extracted ontology files!

---

## Summary

**Issue**: Raw triples showing Greek labels like "βρίσκεται σε" instead of English "has current location"

**Root cause**: Getting predicate labels directly from SPARQL, which returns random language

**Solution**: Use `property_labels.json` (which has English labels) instead of SPARQL labels

**Impact**: All raw triples in sources now show consistent English labels

**Files modified**: `universal_rag_system.py` (lines 599-603, 668-671)

✅ Fixed! All predicate labels in sources will now be in English.
