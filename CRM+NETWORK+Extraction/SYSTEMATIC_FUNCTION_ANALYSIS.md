# Systematic Function Analysis - Complete Audit

**Date**: 2025-11-05
**Purpose**: Identify ALL unused functions with evidence
**Method**: Systematic grep analysis of each function

---

## Methodology

For each function:
1. ✅ **Grep for calls** - Search entire codebase for function name
2. ✅ **Trace call chain** - Verify if called by active code path
3. ✅ **Check main.py** - Verify if entry point exists
4. ✅ **Document status** - USED or UNUSED with evidence

---

## Entry Points (Called from main.py)

These are the functions called directly from Flask routes:

| Function | Line in main.py | Status |
|----------|----------------|--------|
| `answer_question()` | 85 | ✅ ACTIVE (POST /api/chat) |
| `get_wikidata_for_entity()` | 66 | ✅ ACTIVE (GET /api/entity/.../wikidata) |
| `fetch_wikidata_info()` | 71 | ✅ ACTIVE (GET /api/entity/.../wikidata) |
| `get_all_entities()` | 128 | ✅ ACTIVE (initialization check) |
| `initialize()` | 122, 145 | ✅ ACTIVE (startup) |

---

## Analysis Status: IN PROGRESS

Checking all 44 functions systematically...
