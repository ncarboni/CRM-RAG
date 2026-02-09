# Evaluation Progress Report — 2026-02-08

## Overview

Eight evaluation rounds were run on the Asinou dataset (CIDOC-CRM cultural heritage, ~286 entity documents) using 6 sequential conversational questions. Each round tested the full retrieval-to-answer pipeline after applying incremental fixes.

**LLM**: GPT-4o | **Embeddings**: OpenAI text-embedding-3-small | **Dataset**: Asinou (Cyprus churches, frescoes, iconography)

## Evaluation Questions

The 6 questions are asked sequentially as a conversation (each builds on chat history):

1. Where is Asinou located?
2. Tell me more about the panel of Saint Anastasia
3. Who is Anastasia Saramalina?
4. Are there any other donors in the church?
5. Are there any other depictions of donor aside from the ones in Asinou?
6. What's the headdress of Anastasia Saramalina?

## Ground Truth Criteria

| Q# | PASS requires | Key facts |
|----|---------------|-----------|
| Q1 | Cyprus + Nikitari | Location specificity |
| Q2 | South lunette + narthex + Pharmakolytria + donor Saramalina | Panel identification + content |
| Q3 | Donor role + 2+ visual attributes (T-shaped garment, headdress/crespine/barbette, porphyria, neck patches) | Character description |
| Q4 | Nikephoros Ischyrios + donor couple | Enumeration of donors in Asinou |
| Q5 | 2+ non-Asinou donor depictions (Ioannes/Irene at Moutoullas, donor couple at St. Nicholas, Katzouroumpos at St. Dimitrianos) | Cross-church recall |
| Q6 | Crespine + barbette | Specific attribute recall |

Note: Latin Donor is IN Asinou (via Virgin of Mercy panel), so it should NOT be listed for Q5.

## Scorecard

| # | File | Time | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | PASS | PARTIAL | FAIL |
|---|------|------|----|----|----|----|----|----|----|---------|------|
| 1 | 01_baseline_bfs | 19:20 | PARTIAL | PARTIAL | PASS | FAIL | FAIL | PASS | 2 | 2 | 2 |
| 2 | 02_post_target_enrichments | 19:27 | PARTIAL | PARTIAL | PASS | FAIL | FAIL | PASS | 2 | 2 | 2 |
| 3 | 03_post_dual_retrieval | 19:35 | PARTIAL | PARTIAL | PASS | FAIL | FAIL | PASS | 2 | 2 | 2 |
| 4 | 04_baseline_v2_regression | 19:48 | FAIL | PARTIAL | PARTIAL | FAIL | FAIL | PASS | 1 | 2 | 3 |
| 5 | 05_post_fr_traversal | 19:54 | **PASS** | **PASS** | PARTIAL | FAIL | PARTIAL | PASS | 3 | 2 | 1 |
| 6 | 06_post_satellite_absorption | 20:14 | PASS | PASS | **PASS** | **PARTIAL** | PARTIAL | PASS | 4 | 2 | 0 |
| 7 | 07_baseline_v4_context_fix | 20:57 | PASS | PASS | PASS | PARTIAL | **PASS** | PASS | 5 | 1 | 0 |
| 8 | 08_iteration1_prompt_fix | 21:07 | PASS | PASS | PASS | PARTIAL | PARTIAL | PASS | 5 | 1 | 0 |

## Phase-by-Phase Analysis

### Phase 1: BFS-based Document Generation (Rounds 1-4)

**Rounds 1-3** (baseline, target enrichments, dual retrieval) all scored 2/6 PASS. The BFS-based document generation produced entity documents with diluted identity — one real-world entity's properties were scattered across generic BFS traversal paths, making it hard for embeddings to match and for the LLM to synthesize.

- Q1 consistently said "Cyprus" but never "Nikitari" — the spatial containment chain (Asinou → Nikitari → Cyprus) wasn't captured
- Q2 never mentioned "Pharmakolytria" — the saint's epithet was lost in BFS noise
- Q4/Q5 consistently FAIL — donor entities not retrieved because their documents didn't embed well for "donor" queries

**Round 4** (baseline_v2) was a regression: Q1 answered "Wikidata code Q229" instead of "Cyprus". This was the worst-performing round.

### Phase 2: FR-based Document Generation (Rounds 5-6)

The switch to Fundamental Relationships (Tzompanaki & Doerr, 2012) for document generation was the largest single improvement.

**Round 5** (FR traversal): Q1 jumped to PASS (Nikitari found via FR spatial paths), Q2 jumped to PASS (Pharmakolytria surfaced). Q5 improved to PARTIAL (donor couple at St. Nicholas found).

**Round 6** (satellite absorption + MMR diversity): Q3 restored to PASS, Q4 improved to PARTIAL (Nikephoros found). Zero FAILs for the first time.

Key changes:
- FR paths follow CIDOC-CRM Fundamental Relationships instead of ad-hoc BFS
- Satellite entities (E41_Appellation, E55_Type, E52_Time-Span) absorbed into parent documents (606 → ~286 docs)
- MMR diversity penalty (0.2) prevents near-identical entities from flooding retrieval slots

### Phase 3: Context Query Fix (Rounds 7-8)

**Round 7** (baseline_v4): Fixed the contextualized query duplication bug and removed type-aware injection. Q5 improved to PASS (both St. Nicholas and Moutoullas donors mentioned).

The context query fix addressed two bugs:
1. Current question was duplicated in the embedding (already in chat_history + appended again)
2. Multiple previous topics polluted the embedding (e.g., Q3 about Anastasia panel was contaminated by Q2's Last Judgment context)

Type-aware injection was removed as dataset-specific — on MAH (866k entities), label words like "painting" or "the" would match thousands of documents, injecting random noise.

**Round 8** (iteration1): Added prompt instruction "Include entities even if you cannot determine their exact location." No measurable improvement — Q5 actually regressed to PARTIAL due to LLM non-determinism (identical sources, different answer).

## Key Improvements Over Time

```
Round 1 (BFS baseline):     ██░░░░░░░░  2/6 PASS
Round 5 (FR traversal):     █████░░░░░  3/6 PASS  (+Q1, +Q2)
Round 6 (satellite+MMR):    ████████░░  4/6 PASS  (+Q3, Q4→PARTIAL)
Round 7 (context fix):      █████████░  5/6 PASS  (+Q5)
```

## Remaining Limitations

See `final_report.md` for detailed analysis of remaining Q4/Q5 issues.
