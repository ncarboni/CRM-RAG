# Summary of All Changes — MAH Pipeline Evaluation

**Date**: 2026-02-09
**Branch**: `retrieval` (all changes uncommitted)
**Last commit**: `daf94cf` ("update gitignore")
**Dataset**: MAH museum collection (localhost:3030/MAH/sparql, ~467K entity documents)

---

## 1. Modified Files (tracked, uncommitted)

### 1a. `universal_rag_system.py` (+1480 lines, -362 deleted)

This is the bulk of the changes. Eight distinct features were added or modified:

| Feature | Lines (approx) | What it does | Why |
|---|---|---|---|
| **FR traversal integration** | ~40 | Imports `FRTraversal`, adds `_init_fr_traversal()`, uses FR-based document generation instead of BFS | BFS entity documents diluted entity identity — an artwork's doc contained its creator's biography. FR paths from Tzompanaki & Doerr (2012) keep documents entity-centric. |
| **Satellite absorption** | ~20 | Uses `classify_satellite()` to absorb E41_Appellation, E55_Type, E52_Time-Span into parent docs | These satellite entities produced tiny near-empty documents that cluttered retrieval (606 to 286 docs for Asinou). |
| **Target enrichments** | ~30 | `build_target_enrichments()` + `_format_targets()` inline P2_has_type and K14_has_attribute | Panel descriptions were missing type annotations (e.g., "Christological cycle"). |
| **Triples enrichment** | ~234 | New `_build_triples_enrichment()` method | Retrieved docs lacked structured relationship data. This pulls raw triples from `_triples_index` (built from edges.parquet), prioritizes inter-document and key CIDOC-CRM triples (temporal, creator, location, exhibition, depiction), formats as readable text. Caps at 15 triples per entity, 5000 chars total. |
| **SPARQL augmentation** | ~320 | 6 new methods: `_is_sparql_amenable()`, `_generate_sparql_query()`, `_extract_sparql_from_response()`, `_execute_sparql_augmentation()`, `_format_sparql_results()`, `_run_sparql_augmentation()` | Aggregation/listing/ranking questions ("which Swiss artists?", "top 10") cannot be answered by FAISS similarity retrieval alone. Text-to-SPARQL via LLM with retry fills this gap. Includes multilingual filter guidance (French/German/Italian/English) and one retry with feedback if first query returns empty. |
| **Dual retrieval for follow-ups** | ~40 | `answer_question()` now runs both a contextualized query (previous user message + current) and raw FAISS query, then interleaved merge of results | Follow-up questions like "When did it happen?" lost context with only raw FAISS. |
| **Vague follow-up detection** | ~20 | Detects pronoun-dominated short questions (zero content words after stopword removal, <=10 words), skips raw FAISS branch | Questions like "When they took place?" retrieved garbage via raw FAISS because no content words exist to match embeddings against. |
| **MMR diversity penalty** | ~30 | `DIVERSITY_PENALTY = 0.2` in `compute_coherent_subgraph()`, cosine similarity matrix pre-computed, penalizes `0.2 * max_sim_to_selected` | Near-identical entities (e.g., multiple E55_Type for "painting") flooded retrieval slots. |
| **Graph context** | ~50 | `_build_graph_context()`, `GRAPH_CONTEXT_MAX_NEIGHBORS=5`, `GRAPH_CONTEXT_MAX_LINES=50` | Gives LLM visibility into neighboring entities not directly retrieved, showing broader knowledge graph structure. |
| **Chat history support** | ~20 | `answer_question()` accepts `chat_history` parameter, uses it for contextualized queries and SPARQL generation | Enables conversational follow-up support end-to-end. |

### 1b. `main.py` (+3 lines)

- Passes `chat_history` from the HTTP request body to `answer_question()`.
- Changed: `result = current_rag.answer_question(question)` to `result = current_rag.answer_question(question, chat_history=chat_history)`.

### 1c. `static/js/chat.js` (+25 lines)

- Added `chatHistory` array to track conversation client-side.
- Sends last 6 messages (3 exchanges) as `chat_history` in API requests.
- Tracks assistant responses in history after receiving them.
- Minor fix: Wikidata source detection uses `source.wikidata_id` instead of `source.type === "wikidata"`.
- Added Wikidata button in source display UI.

### 1d. `scripts/bulk_generate_documents.py` (+215 lines)

- Integrated FR traversal into the bulk document generation pipeline (used for cluster/MAH processing).
- Added `_init_fr_traversal()`, satellite classification, target enrichment formatting.
- Both bulk and universal_rag now use the same FR-based document generation approach.

### 1e. `fundamental_relationships_cidoc_crm.json` (+2116 lines)

- Expanded from ~237 paths to ~437 paths by adding sub-property variants.
- Added paths via P94i_was_created_by, P108i_was_produced_by, P95i_was_formed_by, P13i_was_destroyed_by, and other inverse/sub-properties from the CRM taxonomy.
- Ensures FR path matching works with actual predicates found in real datasets (which use sub-properties, not just the base properties defined in the standard).

---

## 2. New Untracked Files

| File | Size | Purpose |
|---|---|---|
| `fr_traversal.py` | ~28 KB | FR path matcher module — core of the new document generation strategy. Implements Tzompanaki & Doerr (2012) Fundamental Relationships. |
| `config/fc_class_mapping.json` | ~4.5 KB | Maps 168 CRM classes to 6 Fundamental Categories (Thing, Actor, Place, Event, Concept, Time). |
| `scripts/evaluate_pipeline.py` | ~4.1 KB | Evaluation harness — runs 11 MAH questions as a sequential conversation, logs retrieval details to JSON. |
| `reports/evaluation_baseline.json` | — | Baseline evaluation results (before any fixes). |
| `reports/evaluation_round1.json` | — | Round 1 results (triples enrichment + SPARQL v1 + vague follow-up detection). |
| `reports/evaluation_round2.json` | — | Round 2 results (improved SPARQL with multilingual filters + retry). |
| `evaluation_report.md` | — | Partial evaluation report (incomplete). |
| `logs/` | — | Application logs from evaluation runs. |
| `CIDOC2VEC.pdf` | — | Reference paper (not code-related). |

---

## 3. Evaluation Results

### Test Questions (MAH museum dataset)

11 questions tested as a sequential conversation to simulate real usage:

1. Which pieces from Swiss Artists are in the Musee d'art et d'histoire in Geneva?
2. Which paintings depict Geneva?
3. Tell me more about Ferdinand Hodler and its paintings
4. Tell me more about Guerrier au morgenstern
5. Guerrier au morgenstern and Fondation Pierre Gianadda, it has been exhibited in which exhibition?
6. When it happened and with which other pieces?
7. In which exhibitions and where the work of Hodler has been featured?
8. When they took place?
9. Aside from Hodler, are there any relevant Swiss Artist?
10. What Hans Schweizer did?
11. Which are the top 10 Swiss Artist in the Musee d'art et d'histoire?

### Results Comparison

| Q# | Question (abbreviated) | Baseline | Round 2 | Change |
|---|---|---|---|---|
| Q1 | Swiss Artists in museum? | PARTIAL — names 2 works | PARTIAL — names exhibitions not works | Sideways (SPARQL over-constrains with museum location filter) |
| Q2 | Paintings depicting Geneva? | PARTIAL — 1 painting | PARTIAL — slightly better | Slight improvement |
| Q3 | Ferdinand Hodler? | PASS | PASS | Stable |
| Q4 | Guerrier au morgenstern? | PARTIAL — thin | PARTIAL — more detail via triples | Improved |
| Q5 | Guerrier + Fondation Pierre Gianadda? | PARTIAL — vague | PARTIAL — names more exhibitions | Improved |
| Q6 | When + which other pieces? | PARTIAL | PARTIAL | Stable |
| Q7 | Hodler exhibitions and where? | PARTIAL | PARTIAL — more venues named | Improved |
| Q8 | When they took place? | FAIL — garbage retrieval | PARTIAL — uses chat context | Major improvement |
| Q9 | Other Swiss artists? | FAIL — "no information" | PASS — names 5+ artists via SPARQL (1545 results) | Major improvement |
| Q10 | What Hans Schweizer did? | FAIL — only "from Herisau" | FAIL — same thin answer | No change |
| Q11 | Top 10 Swiss Artists? | FAIL — can't rank | FAIL — SPARQL generates malformed GROUP BY | No change |

### Aggregate Scores

| Metric | Baseline | Round 2 |
|---|---|---|
| PASS | 1 | 2 |
| PARTIAL | 3 | 7 |
| FAIL | 7 | 2 |

---

## 4. Known Issues in Current Code

1. **SPARQL for Q1**: Over-constrains with `P55_has_current_location` museum location filter that doesn't match the data model. Also uses English "geneva" but data uses French "Geneve".
2. **SPARQL for Q11**: LLM generates syntactically malformed `GROUP BY` / `COUNT` queries that Fuseki rejects as `QueryBadFormed`.
3. **Q10 (Hans Schweizer)**: The production chain `P108i_was_produced_by / P14_carried_out_by` doesn't connect to this agent in the data — may need different SPARQL path patterns or the data simply doesn't link this artist to works.
4. **Response times increased**: Approximately 2x slower due to dual retrieval + SPARQL augmentation (estimated 14s to 28s for complex queries).

---

## 5. Ground Truth (from direct SPARQL queries)

Key facts verified against the actual data:

- **Swiss artists**: Nationality is modeled via `P107i_is_current_or_former_member_of` linking to a group with label containing "suisse".
- **Top Swiss artists by work count**: Hodler (28 works), Alfred Dumont (26), Stephane Brunner (25), Jean-Antoine Linck (16), Jean-Etienne Liotard (12).
- **Works to exhibitions**: Linked via `P16i_was_used_for` (not `P12i_was_present_at`).
- **Guerrier au morgenstern** (work/49580): Has 3 exhibitions via `P16i_was_used_for`, including exhibition/5369 at Fondation Pierre Gianadda in Martigny.
- **Paintings depicting Geneva**: 5 works linked via `P62_depicts`.

---

## 6. How to Revert

All tracked file changes (universal_rag_system.py, main.py, chat.js, bulk_generate_documents.py, FR JSON):

```bash
git checkout -- .
```

New untracked files:

```bash
rm -rf fr_traversal.py config/fc_class_mapping.json scripts/evaluate_pipeline.py reports/ logs/ evaluation_report.md CIDOC2VEC.pdf
```

---

## 7. Origin of Changes

Some changes were developed during an earlier Asinou evaluation session and carried forward uncommitted into this MAH session:

**From Asinou session** (pre-existing uncommitted):
- FR traversal (`fr_traversal.py`, `fundamental_relationships_cidoc_crm.json`, `config/fc_class_mapping.json`)
- Satellite absorption
- Target enrichments
- MMR diversity penalty
- Dual retrieval for follow-ups
- `bulk_generate_documents.py` FR integration

**Added during this MAH session**:
- SPARQL augmentation (all 6 methods)
- Triples enrichment from edges.parquet
- Vague follow-up detection
- Graph context
- Chat history passing (main.py + chat.js)
- `scripts/evaluate_pipeline.py`
- All evaluation reports
