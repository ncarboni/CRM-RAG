# MAH Evaluation — Baseline Diagnostic Report (v2)

**Date**: 2026-02-15
**Dataset**: MAH (Musée d'art et d'histoire, Geneva)
**Scale**: 467,881 entity documents, ~4.4M RDF triples
**LLM**: GPT-4o (OpenAI)
**Embeddings**: text-embedding-3-small (OpenAI)
**Iterations**: 1 (baseline + fixes applied for Issues 1, 2, 5; PPR graph retrieval added)

---

## Executive Summary

We evaluated a CIDOC-CRM RAG pipeline against 13 sequential conversational questions covering entity lookup, enumeration, exhibition navigation, press/article discovery, and quantitative ranking over the MAH museum collection. This evaluation uses a **new question set** (13 questions vs. 11 in previous evaluations) designed to test a broader conversation trajectory: Swiss artists → Geneva paintings → Hodler deep dive → exhibition navigation → article/press discovery → ranking.

**Session-level scores:**

| Metric | Score |
|---|---|
| **Retrieval Completeness** | 0.51 |
| **Faithfulness** | 0.65 |
| **Answer Completeness** | 0.33 |
| **Coherence** | 0.75 |
| **Continuity** | 0.72 |
| **Overall** | 0.59 |

The pipeline performs well on **targeted exhibition questions** (Q4-Q5: exhibitions and exhibition contents) and **specific article lookup** (Q11), but fails on **attribute-filtered enumeration** (Q1, Q2: "Swiss artists", "depict Geneva"), **completeness challenges** (Q6-Q7: 3/9 exhibitions found), and **quantitative ranking** (Q12-Q13: fabricated top-10 list). Six systemic issues were diagnosed, of which three are persistent from previous evaluations and three are newly identified.

---

## Evaluation Questions

| # | Question | Type | Topic |
|---|---|---|---|
| Q1 | Which pieces from Swiss Artists are in the Musée d'art et d'histoire in Geneva? | ENUMERATION | Swiss artworks |
| Q2 | Which paintings depict Geneva? | ENUMERATION | Geneva paintings |
| Q3 | Tell me more about Ferdinand Hodler and its paintings | SPECIFIC | Artist + works |
| Q4 | Give me the list of exhibitions that Hodler participated in | ENUMERATION | Exhibition list |
| Q5 | What other pieces were exhibited in Hodler // Parallélisme? | ENUMERATION | Exhibition contents |
| Q6 | Where the painting Le Lac Léman et le Salève avec cygnes was exhibited? | SPECIFIC | Exhibition lookup |
| Q7 | Only this exhibitions? | SPECIFIC | Follow-up challenge |
| Q8 | Are there any articles talking about the exhibitions of Hodler? | SPECIFIC | Article discovery |
| Q9 | Are there any articles referencing Ferdinand Hodler? | ENUMERATION | Article enumeration |
| Q10 | Are there more articles? | SPECIFIC | Follow-up expansion |
| Q11 | Tell me about the article "Une exposition Hodler". | SPECIFIC | Specific article |
| Q12 | Aside from Hodler, are there any relevant Swiss Artist? | ENUMERATION | Artist enumeration |
| Q13 | Which are the top 10 Swiss Artists in the Musée d'art et d'histoire? | AGGREGATION | Quantitative ranking |

Questions Q1-Q2 start new topics; Q3-Q7 form a 5-turn conversational chain about Hodler, exhibitions, and a specific painting; Q8-Q11 explore the article/press dimension; Q12-Q13 return to the Swiss artist ranking topic.

---

## Baseline Results

### Per-Question Scores

| Q# | Question (abbreviated) | RC | F | AC | C | CT | Mean |
|---|---|---|---|---|---|---|---|
| Q1 | Swiss artist pieces | 0.20 | 0.30 | 0.10 | 0.60 | 1.00 | 0.44 |
| Q2 | Paintings depicting Geneva | 0.30 | 0.70 | 0.10 | 0.70 | 0.80 | 0.52 |
| Q3 | Hodler and his paintings | 0.40 | 0.80 | 0.40 | 0.80 | 0.70 | 0.62 |
| Q4 | Hodler exhibitions | 0.80 | 0.80 | 0.70 | 0.90 | 0.80 | 0.80 |
| Q5 | Parallélisme pieces | 0.70 | 0.90 | 0.50 | 0.90 | 0.80 | 0.76 |
| Q6 | Le Lac Léman exhibitions | 0.50 | 0.90 | 0.30 | 0.80 | 0.70 | 0.64 |
| Q7 | Only these exhibitions? | 0.30 | 0.30 | 0.20 | 0.70 | 0.80 | 0.46 |
| Q8 | Articles on Hodler exhibitions | 0.60 | 0.80 | 0.40 | 0.80 | 0.60 | 0.64 |
| Q9 | Articles referencing Hodler | 0.40 | 0.70 | 0.20 | 0.60 | 0.50 | 0.48 |
| Q10 | More articles? | 0.50 | 0.60 | 0.30 | 0.70 | 0.60 | 0.54 |
| Q11 | "Une exposition Hodler" article | 0.80 | 0.80 | 0.60 | 0.80 | 0.70 | 0.74 |
| Q12 | Other Swiss artists | 0.60 | 0.50 | 0.30 | 0.70 | 0.70 | 0.56 |
| Q13 | Top 10 Swiss artists | 0.50 | 0.30 | 0.20 | 0.70 | 0.70 | 0.48 |
| **Mean** | | **0.51** | **0.65** | **0.33** | **0.75** | **0.72** | **0.59** |

**Metric definitions:**
- **RC** (Retrieval Completeness): Were the key entities from the reference answer retrieved?
- **F** (Faithfulness): Is the answer consistent with the retrieved sources? (No hallucination)
- **AC** (Answer Completeness): How complete is the answer relative to the SPARQL reference?
- **C** (Coherence): Is the answer well-structured and logically sound?
- **CT** (Continuity): Does the answer maintain conversation context appropriately?

### Performance Tiers

**Strong (mean ≥ 0.70)**:
- Q4 (0.80): Found 9 Hodler exhibitions. Best performance in the session.
- Q5 (0.76): Found 17/37 Parallélisme works. Good but incomplete.
- Q11 (0.74): Found the specific article entity and described it accurately.

**Moderate (0.55-0.69)**:
- Q3 (0.62): Described Hodler paintings but missed the agent entity (agent/8679).
- Q6 (0.64): Found 3/9 exhibitions for the specific painting.
- Q8 (0.64): Found 5 press clippings about Hodler exhibitions.

**Weak (< 0.55)**:
- Q1 (0.44): Retrieved press clippings about Swiss art instead of actual artworks. 0/152 works found.
- Q2 (0.52): Found only 4/460 paintings depicting Geneva.
- Q7 (0.46): Incorrectly confirmed 3 exhibitions as complete (9 exist).
- Q9 (0.48): Repeated Q8 answer nearly verbatim instead of expanding scope.
- Q10 (0.54): Listed collection folders rather than individual articles. 5/131 documents.
- Q12 (0.56): Listed artists from École suisse member list in document order, not by relevance.
- Q13 (0.48): Fabricated "top 10" ranking from document-order member list. 1/10 correct.

---

## Per-Question Analysis

### Q1: Which pieces from Swiss Artists are in the Musée d'art et d'histoire in Geneva?
**Reference**: 152 works by Swiss artists (École suisse members). Top: Hodler (28), Dumont (26), Brunner (25).
**Answer**: Listed 4 exhibition/article titles ("Des artistes genevois au Musée d'art et d'histoire", etc.) — not actual artworks.
**Diagnosis**: The pipeline retrieves press clippings and exhibition catalog titles that textually match "Swiss Artists" + "Musée d'art et d'histoire" far better than individual artwork entity documents. The FC boost for "Thing" applies to 88/90 candidates (98%), providing zero discrimination. No actual work entities by Swiss artists appear in the 20 sources. **Root cause**: embedding similarity mismatch — archive titles are more lexically similar to the query than artwork entity documents.

### Q2: Which paintings depict Geneva?
**Reference**: 460 paintings via P62_depicts → entities labeled "Genève"/"Geneva".
**Answer**: Found 4 paintings (Genève vue de la campagne, Geneva from Colgini, Genève vue de Cologny, Le Quai des Pâquis). Also retrieved many press clippings.
**Diagnosis**: Only paintings with "Geneva/Genève" in their own titles are found. The P62_depicts relationship to a Place entity labeled "Genève" is not discoverable via embedding similarity — the query "paintings depict Geneva" doesn't match an entity document about a painting that depicts Geneva but whose title doesn't mention Geneva. **Structural limitation**: attribute-based filtering (P62_depicts → specific Place) requires graph traversal, not embedding similarity.

### Q3: Tell me more about Ferdinand Hodler and its paintings
**Reference**: Agent entity agent/8679, 28 works listed.
**Answer**: Described 5-6 Hodler paintings correctly. Did not retrieve the Hodler agent entity.
**Diagnosis**: Exhibitions named "Ferdinand Hodler" (4 of 10 sources) rank higher than the agent entity in FAISS because their titles are exact matches. The TYPE_CHANNEL_POOL_FRACTION_SPECIFIC (0.3) is insufficient to surface the agent entity with k=10. See Issue 6.

### Q4: Give me the list of exhibitions that Hodler participated in
**Reference**: SPARQL found only 2 via P108i→P14 chain (too narrow). Pipeline found 9 exhibitions.
**Answer**: Listed 9 exhibitions with descriptions. Strong performance.
**Diagnosis**: The pipeline outperformed the SPARQL reference because BM25 keyword matching on "Hodler" found many exhibition entities named after him. The ENUMERATION query type (k=20, pool=120) with Event type channel provided good recall. **This is the pipeline's sweet spot**: named-entity enumeration where keyword matching is effective.

### Q5: What other pieces were exhibited in Hodler // Parallélisme?
**Reference**: 37 works in the exhibition.
**Answer**: Listed 17 works correctly.
**Diagnosis**: Good retrieval of both exhibition entities (exhibition/9234, exhibition/5441) and many work entities via connectivity. 17/37 works found. The exhibition entity documents list associated works, and coherent subgraph extraction pulls in connected works. Remaining 20 works are not retrieved because k=20 and the pool is split between exhibition entities and work entities.

### Q6: Where the painting Le Lac Léman et le Salève avec cygnes was exhibited?
**Reference**: 9 exhibitions.
**Answer**: Only 3 exhibitions listed.
**Diagnosis**: The painting entity (work/41697) is retrieved and its document lists all 9 exhibitions in the "Used in:" field. However, three exhibitions share the label "Ferdinand Hodler" and three share "Bleu, la couleur du Modernisme" — the LLM collapses same-labeled exhibitions as duplicates. Additionally, query analysis classified this as categories=['Place'] instead of ['Event'], directing the type channel to search among 4,575 Place entities instead of 28,236 Event entities. See Issues 2 and 5.

### Q7: Only this exhibitions?
**Reference**: Same 9 exhibitions as Q6.
**Answer**: "Yes" — incorrectly confirms only 3 exhibitions exist.
**Diagnosis**: The pipeline has no mechanism to detect that the previous answer was incomplete. The follow-up retrieval doesn't surface additional exhibition entities because the dual retrieval query inherits the Q6 misclassification. Faithfulness suffers because the LLM confidently asserts completeness based on incomplete context.

### Q8: Are there any articles talking about the exhibitions of Hodler?
**Reference**: 131 documents reference Hodler via P67_refers_to.
**Answer**: Found 5 press clipping articles about Hodler exhibitions.
**Diagnosis**: Good topical retrieval via BM25 keyword matching on "Hodler" + "exposition". The pipeline found actual press clippings (archive/item entities) that are relevant. Moderate score because only 5 of many possible articles were found. This is a new question category not tested in previous evaluations — the pipeline handles it reasonably via keyword matching.

### Q9: Are there any articles referencing Ferdinand Hodler?
**Reference**: 131 documents via P67_refers_to.
**Answer**: Repeated Q8's 5 articles nearly verbatim.
**Diagnosis**: The broader question ("referencing Hodler" vs "about exhibitions of Hodler") should surface more documents, but the pipeline returns essentially the same answer. The dual retrieval mechanism reuses the contextualized query from Q8, and the raw query is too similar to produce diverse results. **Continuity problem**: the answer should expand on Q8, not repeat it.

### Q10: Are there more articles?
**Reference**: 131 total documents.
**Answer**: Listed 8 collection/folder categories rather than specific articles.
**Diagnosis**: The follow-up "are there more?" triggers retrieval of archival holding entities (Curated Holding type) — the folders that contain articles — rather than individual articles. The pipeline correctly identifies the archival structure but cannot enumerate the 131 individual documents because they are P67_refers_to linkages buried in the knowledge graph, not discoverable via embedding search.

### Q11: Tell me about the article "Une exposition Hodler"
**Reference**: SPARQL couldn't find this as E73_Information_Object (typed as HMO in the data).
**Answer**: Found archive/item/2198481, described dimensions, collection, and related works.
**Diagnosis**: **Pipeline outperformed SPARQL reference**. BM25 exact title matching retrieved the article entity directly. The answer provides physical dimensions, collection membership, and associated works. The SPARQL reference incorrectly concluded the article doesn't exist because it searched for E73_Information_Object type, but the entity is typed as Human-Made Object.

### Q12: Aside from Hodler, are there any relevant Swiss Artist?
**Reference**: 382 Swiss artists; top by work count: Dumont (26), Brunner (25), Bodmer (9), Disler (9).
**Answer**: Listed 13 artists from the École suisse member list in document order.
**Diagnosis**: The pipeline retrieves the École suisse group entity (384 triples) which contains a flat member list without work counts or relevance indicators. The LLM reads the first members in the document and presents them as "relevant" artists — but these are simply the first names in the list, not the most prolific ones. See Issue 4.

### Q13: Which are the top 10 Swiss Artists in the Musée d'art et d'histoire?
**Reference**: Hodler (28), Dumont (26), Brunner (25), Bodmer (9), Disler (9), Falconnet (9), Vallotton (6), Lory (6), Aberli (5), Gygi (4).
**Answer**: Hodler #1 (correct), then 9 artists from Q12's document-order list (all incorrect rankings).
**Diagnosis**: Only Hodler is correctly placed. The remaining 9 are fabricated from the École suisse member list order. The aggregation context (actor work-count index) exists in the pipeline but does not filter by Swiss school membership, so the LLM sees global top actors and cannot correlate them with École suisse. **Faithfulness: 0.3** — the LLM presents a fabricated ranking with confidence.

---

## Diagnosed Systemic Issues

### Issue 1: FC Boosting Is Non-Discriminating for Majority FC Categories (PERSISTENT)

**Pipeline stage**: Coherent subgraph extraction (scoring phase)

**Description**: The FC-aware boost adds +0.1 to candidates matching the query's target FC categories. When the target is "Thing" — 116,347 of 175,277 documents (66%) — the boost applies to nearly every candidate, providing zero discrimination.

**Evidence**:
- Q1: `FC-aware boosting: 88/90 candidates match target categories ['Thing'] (+0.1)` — 98% receive the boost
- Q2: `FC-aware boosting: 88/90 candidates match target categories ['Thing'] (+0.1)` — identical
- In contrast, Q12 (target Actor): 60/109 candidates — some discrimination but still broad

**Falsification test**: On any CIDOC-CRM dataset where physical objects (E22/E24) vastly outnumber other entity types, queries targeting "Thing" FC would see near-universal FC boosting. This holds for museum collections (dominated by HMO entities) and church datasets (dominated by architectural features).

**Existing mechanisms**: FC boost was fixed in Feb 11 eval (expanded naming conventions). Works for minority FCs (Actor, Place, Event) but structurally ineffective for the majority FC.

**Recommended approach**: Scale the boost inversely with category size: `boost = BASE_BOOST * (1 / log(category_size))`. Alternatively, replace the flat +0.1 with a TF-IDF-style specificity score where rare FC categories receive stronger boosts.

---

### Issue 2: Query Analysis Misclassifies Exhibition-Location Questions (NEW)

**Pipeline stage**: Query analysis (`_analyze_query()`, LLM-based)

**Description**: When a user asks "Where was painting X exhibited?", the word "where" triggers the LLM to classify the answer category as `Place` instead of `Event`. The question asks for exhibitions (Events), not geographic locations. This directs the type-filtered channel to search among 4,575 Place entities instead of 28,236 Event entities.

**Evidence**:
- Q6: `Query analysis: type=SPECIFIC, categories=['Place'], context=['Thing']`
- Q6 type channel: `Type-filtered channel: 4575 docs in target FCs ['Place']`
- Only 2 of 10 sources are exhibition entities; the rest are works and places

**Falsification test**: On any cultural heritage dataset, "Where was [artifact] displayed/shown?" would trigger the same misclassification. The ambiguity between "where" (location) and "where" (in which event) is inherent to exhibition questions.

**Existing mechanisms**: The `QUERY_ANALYSIS_PROMPT` provides examples but none cover the "where was X exhibited" ambiguity. No post-processing corrects known ambiguous patterns.

**Recommended approach**: Add disambiguation examples to the query analysis prompt for "exhibited/displayed/shown" contexts. Alternatively, return **multiple FC candidates** (both Place and Event) when the question contains exhibition-related vocabulary.

---

### Issue 3: Subgraph Connectivity Amplifies Cluster Effects (PERSISTENT)

**Pipeline stage**: Coherent subgraph extraction (`compute_coherent_subgraph()`)

**Description**: Once the first few selected entities share a common artist, production event, or institution, the connectivity signal (α=0.3) creates positive feedback: entities connected to the growing cluster receive progressively higher connectivity scores, even if they have low semantic relevance. This pulls in structurally connected but thematically irrelevant entities.

**Evidence**:
- Q1 selection: "Mimesi n2" (relevance 0.601, connectivity 0.820), "6 gravures: Sans titre" (relevance 0.490, connectivity 1.000), "Block XXXV" (relevance 0.563, connectivity 0.879) — all irrelevant to "Swiss artists"
- Mean connectivity grows from 0.082 (round 2) to 0.466 (round 16) within a single extraction
- Q1 final sources include 5 entirely irrelevant HMO entities pulled in by production chain connectivity

**Falsification test**: On any dataset with densely connected subgraphs (e.g., works by the same artist, objects in the same collection), a query targeting a semantic attribute ("Swiss", "depicting Geneva") would see the connectivity signal pull in structurally connected but thematically unrelated entities.

**Existing mechanisms**: The MMR diversity penalty (0.2) penalizes embedding similarity but not topical redundancy. Entities that are structurally connected but have different embeddings receive low diversity penalties but high connectivity. The mega-entity penalty addresses hub entities but not cluster amplification.

**Recommended approach**: Implement connectivity decay — cap the connectivity contribution growth per selection round, or discount connectivity from entities that were not in the initial top-k by relevance alone. Alternatively, weight connectivity only from direct (1-hop) neighbors rather than including 2-hop virtual edges in the subgraph scoring.

---

### Issue 4: Group Entities Provide Flat Member Lists Without Quantitative Attributes (PERSISTENT)

**Pipeline stage**: Answer generation (LLM context assembly)

**Description**: Group entities like "École suisse" (384 triples) contain flat member lists — names and professional roles — but no per-member work counts or relevance indicators. When the LLM is asked to rank or identify "relevant" artists, it reads the member list and presents names in document order, fabricating a ranking.

**Evidence**:
- Q13 answer lists "Auguste-Henry Berthoud, Albert-Jakob Welti, Alex. Cingria" as top artists — these are the first members in the École suisse document, not the most prolific
- Reference: top by work count are Hodler (28), Dumont (26), Brunner (25) — none of Q13's answers (except Hodler) match
- The aggregation context (work-count index) exists but contains all 2,534 actors globally — it cannot filter by Swiss school membership

**Falsification test**: On any dataset with group/school/organization entities that have membership lists, "top N" or "most notable" queries would produce document-order lists rather than quantitative rankings. E.g., "Who are the most important donors to the church?" with a donor group entity.

**Existing mechanisms**: The `actor_work_counts()` method in KnowledgeGraph provides global counts. The Feb 13 iteration built a work-count index. However, filtering by group membership requires combining the work-count index with P107i_is_current_or_former_member_of — this join is not currently implemented.

**Recommended approach**: At context assembly time, when a Group entity is in the source set and the query is AGGREGATION type, intersect the group's membership with the work-count index to produce a ranked sub-list. This is a context enrichment operation (similar to temporal enrichment in the Feb 13 eval).

---

### Issue 5: Same-Label Disambiguation Missing in Entity Documents (NEW)

**Pipeline stage**: Document generation / Answer generation

**Description**: When an entity has multiple relationships of the same type to entities with identical labels, the entity document lists them identically. The LLM interprets these as duplicates and collapses them. Work/41697 has 9 exhibitions, but 3 are labeled "Ferdinand Hodler" and 3 are labeled "Bleu, la couleur du Modernisme" — the document's "Used in:" field shows them as repetitions.

**Evidence**:
- Work/41697 document "Used in:" line: `Bleu, la couleur du Modernisme, ..., Ferdinand Hodler, Ferdinand Hodler, Ferdinand Hodler, Bleu, la couleur du Modernisme, Hodler // Parallélisme, Ferdinand Hodler et Genève..., Bleu, la couleur du Modernisme`
- Q6 answer: only 3 unique exhibition names (collapsing 6 same-name exhibitions into 3)
- SPARQL: 9 distinct exhibition URIs with different venues/dates but same labels

**Falsification test**: On any dataset where multiple distinct entities share the same rdfs:label (common in museum data — e.g., multiple exhibitions titled "Byzantine Art" at different venues), the document's inline listing would display them identically, causing the LLM to report fewer distinct entities.

**Existing mechanisms**: Target enrichments add P2_has_type and K14_has_attribute inline. No mechanism adds venue, date, or URI suffix to disambiguate same-label entities.

**Recommended approach**: During document generation, when multiple targets of the same predicate share a label, append a disambiguating suffix: venue name, date, or truncated URI. Example: "Ferdinand Hodler (Zurich, 1917)" vs "Ferdinand Hodler (Brussels, 1921)".

---

### Issue 6: Agent Entity Not Retrieved for SPECIFIC Queries About Prolific Named Entities (PERSISTENT)

**Pipeline stage**: Retrieval (type-filtered channel for SPECIFIC queries)

**Description**: Q3 asks about "Ferdinand Hodler and his paintings" but the Hodler agent entity (agent/8679, 315 triples) does not appear in the 10 sources. Exhibitions named "Ferdinand Hodler" rank higher in both FAISS and BM25 because their titles are exact matches. The 30% type channel pool fraction for SPECIFIC queries is insufficient.

**Evidence**:
- Q3 sources: 4 exhibitions named "Ferdinand Hodler", 4 work entities, 0 agent entities
- Q4 (ENUMERATION, k=20): agent/8679 still absent
- Q13 (AGGREGATION, k=25, 50% type fraction): agent/8679 IS retrieved — larger pool and higher fraction make the difference

**Falsification test**: On any dataset with a prolific entity that has many derived entities sharing its name (exhibitions, publications, catalogs), a SPECIFIC query about that entity with k=10 would prioritize the named-after entities over the entity itself.

**Existing mechanisms**: The type channel was extended to SPECIFIC queries in Feb 13 with 30% pool fraction. Explicitly identified as potentially insufficient in that report.

**Recommended approach**: Increase TYPE_CHANNEL_POOL_FRACTION_SPECIFIC from 0.3 to 0.5, or implement entity name matching as a direct retrieval signal — when the query mentions a specific entity name, boost exact-match agent/actor entities in the final scoring.

---

## Issue Prioritization

### Tier 1: Highest Structural Impact (affects multiple question types, any dataset)

| # | Issue | Status | Effort | Expected Impact |
|---|---|---|---|---|
| 2 | Query analysis misclassification | NEW | Low | Q6-Q7 improvement (+0.3 AC each) |
| 5 | Same-label disambiguation | NEW | Medium | Q6-Q7 improvement (recover 6 hidden exhibitions) |

### Tier 2: Significant Impact (harder to fix, broader implications)

| # | Issue | Status | Effort | Expected Impact |
|---|---|---|---|---|
| 1 | FC boost non-discrimination | PERSISTENT | Low | Q1-Q2 modest improvement |
| 3 | Connectivity cluster amplification | PERSISTENT | Medium | Q1 improvement, reduces noise across all queries |
| 6 | Agent entity not retrieved for SPECIFIC | PERSISTENT | Low | Q3 improvement |

### Tier 3: Architectural Changes Required

| # | Issue | Status | Effort | Expected Impact |
|---|---|---|---|---|
| 4 | Group entities lack quantitative attributes | PERSISTENT | High | Q12-Q13 improvement (requires context enrichment join) |

---

## Proposed Solutions

This section provides concrete, dataset-agnostic implementation plans for each diagnosed issue, ordered by recommended implementation sequence (quick wins first, then structural changes).

### Solution A: Fix Query Analysis for Exhibition-Location Ambiguity (Issue 2)

**Target**: `src/crm_rag/rag_system.py`, lines 59–89 (`QUERY_ANALYSIS_PROMPT`)
**Effort**: Low (prompt-only change)
**Expected impact**: Q6 AC +0.3, Q7 AC +0.2

**Problem**: The prompt's examples don't cover the "where was X exhibited/displayed?" pattern, so the LLM maps "where" → Place instead of Event.

**Implementation**: Add disambiguation examples to `QUERY_ANALYSIS_PROMPT` that teach the LLM to distinguish location queries from exhibition-history queries:

```
Additional examples to insert:
- "Where was painting X exhibited?" → categories=["Event"], context=["Thing"]
  (asks for exhibitions/activities, not geographic locations)
- "In which exhibitions was the artwork shown?" → categories=["Event"], context=["Thing"]
- "Where is the church located?" → categories=["Place"], context=["Thing"]
  (asks for geographic location)
```

Add a clarifying note after the category definitions:
```
Note: "Where was X exhibited/displayed/shown?" asks for exhibitions (Event),
not geographic locations (Place). Use Place only when the question asks for
a physical location, city, or region.
```

**Why this works**: The query analysis LLM call is the first stage; a misclassification here propagates to the type-filtered channel (4,575 Place docs vs 28,236 Event docs) and FC boosting. Correcting this single prompt fixes the entire downstream chain for Q6-Q7.

**Anti-pattern check**: The added examples use generic patterns ("where was X exhibited"), not dataset-specific entities or URIs.

---

### Solution B: Same-Label Disambiguation in Document Generation (Issue 5)

**Target**: `src/crm_rag/fr_traversal.py`, `_format_targets()` method (lines 756–824)
**Effort**: Medium (document generation change — requires re-generating documents)
**Expected impact**: Q6 AC +0.3, Q7 AC +0.2 (combined with Solution A)

**Problem**: `_format_targets()` currently deduplicates identical formatted labels with an `x{N}` count suffix: three exhibitions all labeled "Ferdinand Hodler" become `"Ferdinand Hodler x3"`. The LLM treats this as a single exhibition appearing three times, not three distinct exhibitions.

**Implementation**: When multiple targets of the same predicate produce identical formatted labels, disambiguate them by appending a differentiating suffix derived from available metadata. The priority order for disambiguation:

1. **Time-span date** (if the target entity has a resolved P4_has_time-span): `"Ferdinand Hodler (1917)"` vs `"Ferdinand Hodler (1921)"`
2. **Place/venue** (if the target entity has a P7_took_place_at or similar location predicate): `"Ferdinand Hodler (Zürich)"` vs `"Ferdinand Hodler (Brussels)"`
3. **Truncated URI suffix** (fallback): `"Ferdinand Hodler [/exhibition/5711]"` vs `"Ferdinand Hodler [/exhibition/8490]"`

Concretely, modify `_format_targets()`:
- After formatting all targets, group by formatted string
- For groups with `len > 1`, attempt disambiguation in the priority order above
- The disambiguation data is already available: `time_span_dates` is passed as a parameter, and target enrichments carry type/attribute data
- For venue information, extend `build_target_enrichments()` to also collect P7_took_place_at labels for Event-typed targets

**Note**: This requires re-running document generation (`bulk_generate_documents.py`). The change is dataset-agnostic — it applies wherever same-label collisions occur.

---

### Solution C: Increase Type Channel Fraction for SPECIFIC Queries (Issue 6)

**Target**: `src/crm_rag/rag_system.py`, `RetrievalConfig` (line 295)
**Effort**: Low (constant change)
**Expected impact**: Q3 AC +0.15

**Problem**: `TYPE_CHANNEL_POOL_FRACTION_SPECIFIC = 0.3` reserves only 30% of pool (18 out of 60 slots) for type-matching candidates. For prolific named entities like "Ferdinand Hodler" — where 4+ exhibitions share the name — the agent entity (agent/8679) gets outranked by same-name exhibitions in both FAISS and BM25.

**Implementation**: Increase `TYPE_CHANNEL_POOL_FRACTION_SPECIFIC` from `0.3` to `0.5`:

```python
# In RetrievalConfig:
TYPE_CHANNEL_POOL_FRACTION_SPECIFIC = 0.5  # Was 0.3
```

This gives SPECIFIC queries the same type channel allocation as ENUMERATION/AGGREGATION. Evidence that this works: Q13 (AGGREGATION, k=25, 50% fraction) successfully retrieves agent/8679. The same fraction at k=10 should surface it for Q3.

**Risk**: Slightly reduces general relevance diversity for SPECIFIC queries by giving more weight to the type channel. This is an acceptable trade-off because the type channel already targets the correct FC categories.

**Alternative (complementary)**: Implement a **name-match boost** in the type-filtered channel. When the query contains a recognized entity label (detected via BM25 exact match), boost that entity's score by +0.2 in the final scoring. This is more targeted but requires a matching step.

---

### Solution D: Scaled FC Boosting by Category Size (Issue 1)

**Target**: `src/crm_rag/rag_system.py`, FC boost block in `compute_coherent_subgraph()` (lines 2285–2301)
**Effort**: Low (scoring change)
**Expected impact**: Q1 RC +0.1, Q2 RC +0.1 (modest — this alone doesn't solve the attribute-filtering ceiling)

**Problem**: The flat `FC_BOOST = 0.10` is applied uniformly. When the target FC is "Thing" (116,347 docs, 66% of total), 98% of candidates receive the same boost, providing zero discrimination.

**Implementation**: Scale the boost inversely with category prevalence using IDF-style weighting:

```python
# Replace flat FC_BOOST = 0.10 with:
fc_type_index = self.document_store.fc_type_index  # {fc_name: set(doc_ids)}
total_docs = len(self.document_store.documents)

for fc_name in query_analysis.categories:
    fc_size = len(fc_type_index.get(fc_name, set()))
    # IDF-inspired: rare categories get stronger boost
    idf_weight = math.log(total_docs / max(fc_size, 1))
    # Normalize: max boost ~0.15 for rare FCs, min ~0.02 for "Thing"
    fc_boost = 0.15 * (idf_weight / math.log(total_docs))
```

With MAH data:
- **Place** (4,575 docs): `idf = log(175277/4575) ≈ 3.64` → boost ≈ 0.15 × 3.64/12.07 ≈ **0.045**
- **Actor** (25,898 docs): `idf ≈ 1.91` → boost ≈ **0.024**
- **Thing** (116,347 docs): `idf ≈ 0.41` → boost ≈ **0.005** (effectively negligible)

This means Place and Event candidates get meaningful discrimination while Thing queries no longer waste a boost on nearly everyone.

---

### Solution E: Connectivity Decay in Subgraph Extraction (Issue 3)

**Target**: `src/crm_rag/rag_system.py`, connectivity scoring in `compute_coherent_subgraph()` (lines 2348–2376)
**Effort**: Medium (algorithm change)
**Expected impact**: Q1 AC +0.1, general noise reduction across all queries

**Problem**: Connectivity score grows progressively: as more entities are selected from a dense cluster, new candidates connected to the cluster get higher and higher connectivity scores. Mean connectivity rises from 0.082 (round 2) to 0.466 (round 16). Irrelevant but structurally connected entities (e.g., "Mimesi n2", "Block XXXV") get pulled in.

**Implementation**: Two complementary mechanisms:

#### E.1: Connectivity eligibility threshold
Only allow connectivity contributions from candidates that were in the top-50% by relevance alone. Entities below this threshold can still be selected (via high relevance) but cannot benefit from connectivity boosting:

```python
# Before the greedy loop:
relevance_median = np.median(relevance_scores)

# Inside the loop, when computing connectivity for candidate idx:
if relevance_scores[idx] < relevance_median:
    connectivity = 0.0  # No connectivity boost for low-relevance candidates
```

#### E.2: Diminishing connectivity weight per round
Reduce the connectivity weight (α) as selection progresses, so early selections are connectivity-aware but later selections prioritize relevance:

```python
# Inside the greedy loop, for selection round r:
effective_alpha = alpha + (1 - alpha) * (r / k)  # Starts at 0.7, grows toward 1.0
# Round 1: effective_alpha = 0.73 (still uses connectivity)
# Round 10: effective_alpha = 1.0 (pure relevance)
base_score = effective_alpha * relevance + (1 - effective_alpha) * connectivity
```

This ensures the first 3-5 selections build a coherent cluster, but later selections don't pull in weakly-related entities just because they share production chains.

**Anti-pattern check**: Both mechanisms use relative thresholds (median, round number), not absolute values tied to any dataset.

---

### Solution F: Group-Membership-Aware Aggregation Context (Issue 4)

**Target**: `src/crm_rag/rag_system.py`, `_build_aggregation_context()` (lines 2688–2807) and `src/crm_rag/knowledge_graph.py`, `actor_work_counts()` (lines 451–490)
**Effort**: High (requires new method + context enrichment logic)
**Expected impact**: Q12 AC +0.3, Q13 AC +0.5

**Problem**: When the LLM context includes a Group entity (like "École suisse") with 383 members AND the global actor work-count index (2,534 actors), it cannot join them. The group's flat member list dominates the LLM's attention, and it presents members in document order.

**Implementation**: Add a **group-filtered work-count enrichment** to the context assembly, similar to how temporal enrichment was implemented in the Feb 13 evaluation.

#### Step 1: New KnowledgeGraph method `actor_work_counts_for_group(group_uri)`

```python
def actor_work_counts_for_group(self, group_uri: str) -> Dict[str, int]:
    """Compute work counts only for actors who are members of the given group.

    Uses P107i_is_current_or_former_member_of to find group members,
    then intersects with the production chain (P108i → P14).
    """
    # Find all members via P107i_is_current_or_former_member_of
    members = set()
    try:
        group_vid = self._name_to_vid[group_uri]
    except KeyError:
        return {}

    for e in self._graph.es.select(_source=group_vid):
        if "P107i" in e["predicate"]:
            members.add(self._graph.vs[e.target]["name"])
    # Also check reverse direction
    for e in self._graph.es.select(_target=group_vid):
        if "P107i" in e["predicate"]:
            members.add(self._graph.vs[e.source]["name"])

    # Get global work counts, filter to members only
    all_counts = self.actor_work_counts()
    return {uri: count for uri, count in all_counts.items() if uri in members}
```

#### Step 2: Enrich context in `_build_aggregation_context()`

When a Group entity is in the retrieved sources and the query is AGGREGATION or ENUMERATION:

```python
# In _build_aggregation_context(), after the actor_work_counts block:
group_sources = [doc for doc in retrieved_docs
                 if doc.metadata.get("type") == "Group"]
for group_doc in group_sources:
    group_uri = group_doc.metadata.get("uri")
    if group_uri:
        group_counts = kg.actor_work_counts_for_group(group_uri)
        if group_counts:
            sorted_members = sorted(group_counts.items(), key=lambda x: -x[1])[:15]
            labeled = [f"{kg.get_label(uri)} ({count} works)"
                       for uri, count in sorted_members]
            lines.append(f"Top members of {group_doc.metadata.get('label', 'group')} "
                         f"by work count: {', '.join(labeled)}")
```

This gives the LLM a quantitative ranking of group members instead of a flat list, directly addressing Q12 ("relevant Swiss artists") and Q13 ("top 10 Swiss artists").

**Note**: This follows the same architectural pattern as the temporal enrichment (Feb 13): detect relevant entity types in the retrieved set → targeted graph/SPARQL query → inject structured data into the LLM context. No document regeneration needed.

---

### Solution Summary and Recommended Implementation Order

| Order | Solution | Issue | Effort | Requires Rebuild | Expected AC Gain |
|---|---|---|---|---|---|
| 1 | A: Query analysis prompt | Issue 2 | Low | No | Q6 +0.3, Q7 +0.2 |
| 2 | C: Type channel 0.3→0.5 | Issue 6 | Low | No | Q3 +0.15 |
| 3 | D: Scaled FC boost | Issue 1 | Low | No | Q1-Q2 +0.1 each |
| 4 | E: Connectivity decay | Issue 3 | Medium | No | Q1 +0.1, noise reduction |
| 5 | F: Group-filtered counts | Issue 4 | High | No | Q12 +0.3, Q13 +0.5 |
| 6 | B: Same-label disambig | Issue 5 | Medium | **Yes** (docs regen) | Q6-Q7 +0.2 each |

Solutions A-E are retrieval-time changes that don't require document regeneration. Solution B requires re-generating entity documents but can be deferred if Solutions A+C already recover most of Q6-Q7's missing exhibitions.

**Combined expected impact**: If all 6 solutions are applied, the projected session-level Answer Completeness would improve from **0.33 → ~0.48** (+45%), primarily driven by Q6-Q7 (exhibition completeness), Q12-Q13 (ranking accuracy), and Q3 (agent entity retrieval).

---

## Architectural Ceiling Solutions

Beyond the 6 targeted fixes, three fundamental limitations require deeper architectural changes:

### Ceiling 1: Attribute-Based Filtering (Q1, Q2)

**Problem**: "Swiss artist pieces" and "paintings depicting Geneva" require filtering by RDF property chains that embedding similarity cannot discover.

**Proposed approach — Hybrid SPARQL-augmented retrieval**:

When query analysis detects an ENUMERATION query with attribute-based constraints (e.g., nationality, depicted subject), generate a lightweight SPARQL query at runtime to fetch matching entity URIs, then intersect these with the embedding-based candidate pool.

```
Query analysis detects: ENUMERATION + attribute constraint ("Swiss artists")
    ↓
Generate SPARQL: SELECT ?work WHERE {
    ?actor P107i_is_current_or_former_member_of ?group .
    ?work P108i_was_produced_by/P14_carried_out_by ?actor .
}
    ↓
Execute against SPARQL endpoint → set of matching URIs
    ↓
Intersect with FAISS/BM25 candidate pool OR inject as additional candidates
    ↓
Continue with standard coherent subgraph extraction
```

This requires: (a) a SPARQL query template generator guided by query analysis categories and context, and (b) the SPARQL endpoint to be available at query time (already the case for temporal enrichment).

**Effort**: High. The SPARQL template generation is the main challenge — it needs to be general enough to handle diverse attribute queries without hardcoding property paths.

### Ceiling 2: Quantitative Ranking (Q12, Q13)

**Addressed by Solution F** (group-filtered work-count enrichment). The remaining gap is when the user asks for rankings that combine multiple criteria (e.g., "most exhibited Swiss artists" — requiring both exhibition count and nationality filtering). This would need a composable aggregation engine, which is beyond the current architecture.

### Ceiling 3: Completeness Verification (Q7)

**Problem**: The pipeline cannot detect that its own answer is incomplete.

**Proposed approach — Confidence-calibrated responses**:

When the LLM generates an enumeration answer, compare the count of items listed with the cardinality information available in the source documents. If a source entity lists "Used in: 9 items including X, Y, Z" but the answer only names 3, inject a caveat:

```
After LLM generation:
    ↓
Parse answer for enumeration count (regex: numbered list items)
    ↓
Check source documents for cardinality hints ("N items including", "... and N more")
    ↓
If answer_count < source_cardinality:
    Append: "Note: the source data indicates N total items; additional items
    may exist beyond those listed here."
```

This doesn't fix the retrieval gap but prevents the LLM from confidently affirming completeness when the evidence suggests otherwise.

---

## Comparison with Previous Evaluations

### Cross-Evaluation Metric Comparison

| Metric | Feb 11 Baseline | Feb 11 Final | Feb 13 Baseline | Feb 13 Final | **Feb 15 v2 Baseline** |
|---|---|---|---|---|---|
| Retrieval Completeness | 0.31 | 0.38 | 0.36 | 0.41 | **0.51** |
| Faithfulness | 0.61 | 0.67 | 0.68 | 0.72 | **0.65** |
| Answer Completeness | 0.25 | 0.31 | 0.35 | 0.39 | **0.33** |
| Coherence | 0.60 | 0.66 | 0.67 | 0.72 | **0.75** |
| Continuity | 0.59 | 0.65 | 0.71 | 0.74 | **0.72** |

**Notes on comparability**: The question sets differ (11 questions in Feb 11/13, 13 questions in Feb 15 v2), so direct numerical comparison is approximate. The Feb 15 v2 set includes new question categories (article discovery Q8-Q11, follow-up challenge Q7) that are harder than the previous sets. The higher baseline RC (0.51 vs 0.36) partially reflects that the new questions include more keyword-matchable topics (Hodler exhibitions, article titles).

### Issue Tracking Across Evaluations

| Issue | Feb 11 | Feb 13 | Feb 15 v2 Baseline | Feb 15 v2 Iter 1 |
|---|---|---|---|---|
| FC boost broken (naming mismatch) | **Diagnosed & Fixed** | Resolved | Resolved | Resolved |
| FC boost non-discriminating (majority FC) | Identified (symptom) | Persists | **Persists** (Issue 1) | **Fixed** — FC boost removed (redundant with type channel) |
| Type channel missing for SPECIFIC | — | **Diagnosed & Fixed** (30%) | Persists at 30% (Issue 6) | Persists (deferred) |
| Agent entity not retrieved | — | Identified (symptom) | **Persists** (Issue 6) | Persists (deferred, may improve via PPR) |
| Temporal predicates absent | — | **Diagnosed & Fixed** | Resolved | Resolved |
| Mega-entity dominance | — | **Diagnosed & Fixed** | Resolved | Resolved |
| Embedding similarity mismatch (Q1) | Identified | Persists | **Persists** (Issues 1+3) | **Partially addressed** — PPR graph retrieval added |
| Connectivity cluster amplification | — | — | **NEW** (Issue 3) | Persists (deferred) |
| Query analysis misclassification | — | — | **NEW** (Issue 2) | **Fixed** — prompt examples added to prompts.yaml |
| Same-label disambiguation | — | — | **NEW** (Issue 5) | **Fixed** — URI-aware disambiguation with date/venue/type |
| Group entity flat lists | — | Identified (symptom) | **Persists** (Issue 4) | Persists (deferred) |
| Work-count index global (no group filter) | — | Built index | **Persists** (Issue 4) | Persists (deferred) |

### Newly Tested Question Categories

The Feb 15 v2 evaluation introduced three new question patterns:

1. **Follow-up challenge (Q7 "Only this exhibitions?")**: Tests whether the pipeline can detect incomplete answers. Score: 0.46 — the pipeline incorrectly affirms completeness.

2. **Article/press discovery (Q8-Q11)**: Tests retrieval of archival press clippings and documents. Scores: 0.48-0.74. BM25 keyword matching is effective for finding specific article titles (Q11: 0.74) but poor for comprehensive enumeration (Q9-Q10: 0.48-0.54).

3. **Specific article lookup (Q11)**: The pipeline outperformed the SPARQL reference — BM25 found the article entity typed as HMO, which the SPARQL reference missed because it searched only for E73_Information_Object type.

---

## Iteration 1: Implemented Fixes

**Commit**: `95db3c1` on `network` branch
**Date**: 2026-02-15
**Files modified**: `rag_system.py`, `fr_traversal.py`, `knowledge_graph.py`, `config/prompts.yaml`

### Fix 1: FC Boost Removal (Issue 1 — Solution D replacement)

**Approach changed**: Instead of scaling the FC boost by category size (Solution D), the FC boost was removed entirely. The type-filtered channel already provides FC-aware candidate injection at the pool level, making the downstream FC boost in `compute_coherent_subgraph()` redundant. Evidence: 98% of candidates received the boost for Thing queries, providing zero discrimination.

**Changes**:
- Deleted FC boost block (~17 lines) from `compute_coherent_subgraph()`
- Removed `query_analysis` parameter from the method signature, docstring, and call site
- TYPE_SCORE_MODIFIERS per-type logic (positive boosts for HMO/Actor, penalties for Linguistic Object) retained — these provide per-type discrimination and remain valuable

**Rationale**: Simpler than IDF-scaling and addresses the root cause — the type-filtered channel handles FC-level discrimination at pool construction time, so applying it again at scoring time is double-counting.

### Fix 2: Query Analysis Prompt for Exhibition-Location Ambiguity (Issue 2 — Solution A)

**Changes**:
- Extracted the query analysis prompt to `config/prompts.yaml` for easier editing
- Added three disambiguation examples covering the "where was X exhibited?" pattern:
  - `"Where was painting X exhibited?" → categories=["Event"], context=["Thing"]`
  - `"In which exhibitions was the artwork shown?" → categories=["Event"], context=["Thing"]`
  - `"Where is the church located?" → categories=["Place"], context=["Thing"]`

**Expected impact**: Q6 should now classify as `categories=["Event"]` instead of `["Place"]`, directing the type-filtered channel to search among 28,236 Event entities instead of 4,575 Place entities. Q7 (follow-up) should inherit the corrected classification.

### Fix 3: Same-Label Disambiguation (Issue 5 — Solution B)

**Changes in `fr_traversal.py`**:
- Added `_VENUE_PRED_LOCALS = {"P7_took_place_at", "P7i_witnessed"}` for venue collection
- Extended `build_target_enrichments()` to collect venue labels via P7_took_place_at
- Enrichment dict now includes `"venue"` key alongside `"type_tag"` and `"attributes"`
- Rewrote `_format_targets()` deduplication: replaced `Counter` + `"x{N}"` with URI-aware disambiguation
  - First pass: build `(uri, base_label, formatted_label)` tuples
  - Second pass: group by `formatted_label` to detect collisions
  - Collision resolution priority: type tag + date + venue combined
  - Fallback: URI suffix `[exhibition/5711]` when no metadata available

**Changes in `rag_system.py`**:
- Added target time-span resolution in `_create_fr_document_from_prefetched()` after `build_target_enrichments()` and before `format_fr_document()`
- For each target URI, checks `fr_outgoing` for P4_has_time-span, resolves the E52 via satellite dates or `knowledge_graph.resolve_time_span()` fallback
- Time-span dates are now available for disambiguation even when the target entity is an Event (not just Time-Span satellites)

**Expected output examples**:
- Before: `"Ferdinand Hodler x3"` or `"Ferdinand Hodler, Ferdinand Hodler, Ferdinand Hodler"`
- After: `"Ferdinand Hodler (Exhibition, 2017-01-26 to 2017-06-18, Zürich)"`, `"Ferdinand Hodler (Exhibition, 1917)"`, etc.

**Requires rebuild**: Yes — entity documents must be regenerated to apply the new formatting.

### Fix 4: PPR-Based Graph Retrieval (New — addresses Q1/Q2 architectural ceiling)

**Rationale**: Queries like "Swiss artist pieces" (Q1) and "paintings depicting Geneva" (Q2) require traversing RDF property chains (Actor → Production Event → Work, or Work → P62_depicts → Place) that embedding similarity cannot discover. Personalized PageRank on the knowledge graph can propagate relevance from constraint entities (e.g., "École suisse" actor) through the graph topology to discover connected entities.

**Changes in `knowledge_graph.py`**:
- Added `personalized_pagerank(seed_uris, damping, top_n, doc_only)` method
- Wraps igraph's `personalized_pagerank(reset_vertices=seed_vids, damping=damping, weights="weight")`
- Filters to `is_doc=True` vertices, excludes seed URIs, returns top N by PPR score

**Changes in `rag_system.py`**:
- Added `PPR_SEED_MAX = 5` and `PPR_DAMPING = 0.85` to `RetrievalConfig`
- Added `_rrf_fuse_multi(ranked_lists, pool_size, k_rrf)` — generalizes 2-way RRF to N-way
- Added `_ppr_retrieval(query, query_analysis, pool_size)`:
  1. Runs BM25 on query (k=50)
  2. Filters results by `context_categories` FC match using `document_store._fc_doc_ids`
  3. Takes top 5 matching entities as PPR seeds
  4. Runs `knowledge_graph.personalized_pagerank(seed_uris)` to discover connected entities
- Integrated into `retrieve()`: when `query_analysis.context_categories` is non-empty and PPR finds seeds, uses 3-way RRF (FAISS + BM25 + PPR) instead of 2-way

**Activation conditions** (all must be true):
- `query_analysis.context_categories` is non-empty
- Knowledge graph is loaded (`vertex_count > 0`)
- At least one BM25 result matches a context category FC

**Expected impact**: For Q1 ("Swiss artist pieces"), PPR seeds should include "École suisse" or similar Actor entities from context_categories=["Actor"]. PPR propagation through the production chain (Actor → Production Event → Work) should surface actual artworks by Swiss artists that embedding similarity missed.

### Fixes Not Yet Implemented

| Issue | Solution | Status | Reason |
|---|---|---|---|
| Issue 3: Connectivity cluster amplification | Solution E: Connectivity decay | Deferred | Medium effort; want to measure PPR impact first |
| Issue 4: Group entity flat lists | Solution F: Group-filtered counts | Deferred | High effort; requires new KG method + context enrichment |
| Issue 6: Agent entity not retrieved | Solution C: Type fraction 0.3→0.5 | Deferred | Low effort but want to evaluate PPR's effect on agent retrieval first |

---

## Appendix: Reference Answer Sources

Reference answers were built from the SPARQL endpoint (`localhost:3030/MAH/sparql`) using targeted queries for each question. Full SPARQL queries and reference data are in `reference_answers.json`. Note that the SPARQL reference for Q4 (exhibitions) was too narrow — it only traversed P108i_was_produced_by → P14_carried_out_by → P16i_was_used_for, finding only 2 exhibitions. The actual number of Hodler-related exhibitions in the dataset is ~80 (entities named "Ferdinand Hodler" or containing Hodler works).

---

*Generated by RAG Pipeline Evaluation Orchestrator, 2026-02-15*
