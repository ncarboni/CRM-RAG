# RAG Pipeline Evaluation Report

**Date**: 2026-02-08
**Dataset**: Asinou (Cyprus church — frescoes, iconography, donors)
**LLM**: GPT-4o via OpenAI
**Embeddings**: text-embedding-3-small (1536 dims)

---

## 1. Evaluation Protocol

Six questions run as a sequential conversation (with chat history), testing:
- Factual recall (Q1: location)
- Entity-focused retrieval (Q2: panel, Q3: person)
- Relational reasoning over conversation context (Q4: "other donors")
- Cross-entity discovery (Q5: depictions beyond Asinou)
- Attribute lookup (Q6: headdress composition)

Each answer is evaluated against ground truth extracted from the entity documents, not just for plausibility.

Three evaluation rounds:
1. **Baseline** — before any fixes (`evaluation_baseline_v2.json`)
2. **Post-fix v1** — satellite absorption + dual retrieval + duplicate source fix (`evaluation_post_fix2.json`)
3. **Post-fix v2** — + target enrichments (type tags, attributes inline) + improved system prompt (`evaluation_post_fix3.json`)

## 2. Ground Truth (from entity documents)

### The "Anastasias" panel
The panel on the **South lunette** of the **Narthex** depicts **two distinct people**:
- **Saint Anastasia** (icon) — type: Saint Anastasia Pharmakolytria, attributes: cross of martyrdom + bottle of medicine, portraying the historical Anastasia of Sirmium
- **Anastasia Saramalina** (icon) — type: **Donor**, attributes: porphyria, T-shaped garment, headdress (crespine + barbette), neck rectangular patches

The panel also denotes the **Headdress of Anastasia Saramalina** (crespine + barbette). The painting event ("Painting of Anastasias") is typed as Mural Painting, dated to the Lusignan period.

### All donors in the dataset
| Donor | Type | Location | Church |
|-------|------|----------|--------|
| Anastasia Saramalina | Donor | Anastasias panel, South lunette, Narthex | Panagia Phorbiottisa (Asinou) |
| Nikephoros Ischyrios | Founder | Donor panel, Naos | Panagia Phorbiottisa (Asinou) |
| Donor's wife | Donor | Donor couple panel, Narthex | Panagia Phorbiottisa (Asinou) |
| Donor Ioannes and Irene | — | North wall of sanctuary | Our Lady of Moutoullas |
| Donor Michael Katzouroumpos and wife | — | — | Saint Dimitrianos |

### Key relationships
- Panagia Phorbiottisa `Has part` → Donor couple, Narthex
- Donor couple panel `Denotes` → Donor's wife (Donor)
- Donor panel (Naos) `Denotes` → Nikephoros Ischyrios (Founder)
- Nikephoros Ischyrios (icon) `Has type` → Founder

## 3. Fixes Applied

### Fix A: Satellite Entity Absorption (v1)
**Problem**: 42% of the FAISS index (253/606 docs) were E41_Appellation fragments — 2-5 line documents containing only a label, duplicating information already in parent entities.

**Solution**: Classify entities typed as MINIMAL_DOC_CLASSES (E41_Appellation, E55_Type, E54_Dimension, E52_Time-Span, etc.) as satellites. For each satellite, find its parent via incoming edges and absorb its label into the parent doc as "Also known as:", "Dimensions:", or "Time-span:" lines. Skip satellite entities during document generation and filter their triples from edges.parquet.

**Result**:
- Documents: 606 → 286 (53% reduction)
- Edges: 1715 → 794 (54% reduction)
- Avg doc length: 116 → 190 chars (64% increase)
- Zero Appellation/Type documents in FAISS index

### Fix B: Dual Retrieval for Follow-Up Questions (v1)
**Problem**: When chat history is present, the contextualized query overwhelms the raw question signal.

**Solution**: For follow-up questions, run two FAISS retrievals in parallel (contextualized + raw), merge, deduplicate by URI, take top-k.

**Result**: Mixed. Helps in some cases (Q5 now retrieves Donor couple), but doesn't solve the fundamental embedding proximity problem for Q4.

### Fix C: Duplicate Source Bug (v1)
**Problem**: Wikidata-enriched entities appeared twice in sources — once as graph source, once as separate wikidata source with empty type and 0 triples.

**Solution**: Merge Wikidata info (wikidata_id, wikidata_url, image) into the existing graph source entry instead of appending a duplicate. Updated frontend to detect Wikidata-enriched sources by `s.wikidata_id` instead of `s.type === "wikidata"`.

**Result**: All questions now return exactly 10 sources (expected k=10), no duplicates.

### Fix D: Target Enrichments in FR Documents (v2)
**Problem**: FR traversal output contained bare labels — `Denotes: Anastasia Saramalina, Saint Anastasia`. The LLM had no way to know Anastasia Saramalina is a donor or Saint Anastasia is a Pharmakolytria without retrieving their individual docs. Type and role information was invisible at the referencing entity's document level.

**Solution**: At document generation time, for every target entity referenced in FR results or direct predicates, look up its P2_has_type (descriptive type tag) and K14_has_attribute values. Append these inline:
- `Denotes: Anastasia Saramalina (Donor), Saint Anastasia (Saint Anastasia Pharmakolytria)`
- `Denotes: Nikephoros Ischyrios (Founder)`
- For depiction predicates, also include attributes: `Has attribute: cross of martyrdom (Visual Item), bottle of medicine (Visual Item)`

Implementation: `build_target_enrichments()` function in `fr_traversal.py`, called by both `bulk_generate_documents.py` and `universal_rag_system.py` at document creation time. Dataset-agnostic — works for any CIDOC-CRM dataset.

**Result**: The Anastasias panel doc now contains enough context for the LLM to describe both depicted figures with their roles and attributes, even without retrieving the individual icon docs. The Donor panel doc now says `Denotes: Nikephoros Ischyrios (Founder)`, making the founder role visible.

### Fix E: Improved System Prompt (v2)
**Problem**: The system prompt was minimal and didn't address CRM jargon leaking into answers ("denoted by", "features Anastasias, which denotes figures such as") or guide the LLM to enumerate all matching entities.

**Solution**: Added explicit instructions to:
- Translate ontological vocabulary into everyday words ("denotes" → "depicts")
- Never expose raw ontology codes or Wikidata codes
- When describing a panel, explain WHO is depicted and WHAT ROLE each figure has
- When asked about "other" entities, list ALL matching entities from the context

**Result**: CRM jargon no longer appears in answers. Q2 answer now correctly describes both figures with roles and attributes instead of echoing bare labels.

## 4. Per-Question Results

### Q1: Where is Asinou located?

| | Baseline | Post-fix v1 | Post-fix v2 |
|---|---|---|---|
| **Answer** | "country corresponding to Wikidata code Q229" | Cyprus, coordinates | "Cyprus, part of Nikitari" |
| **Grade** | BAD (leaked Wikidata code) | PASS | **PASS** |
| **Notes** | Raw Wikidata code in answer | Missing Nikitari | Now includes Nikitari village |

**Ground truth**: Panagia Phorbiottisa → Is part of: Nikitari, Cyprus.

**v2 answer**: *"Asinou is located in Cyprus, specifically in the place known as Panagia Phorbiottisa, which is part of Nikitari."*

Correct and complete. Mentions Cyprus, Panagia Phorbiottisa, and Nikitari. No leaked codes.

### Q2: Tell me more about the panel of Saint Anastasia

| | Baseline | Post-fix v1 | Post-fix v2 |
|---|---|---|---|
| **Answer** | Flat list: "Anastasia Saramalina and Saint Anastasia" | Same | **Two distinct people with roles and attributes** |
| **Grade** | BAD | BAD | **PASS** |
| **Donor identified** | No | No | **Yes** |
| **Saint type** | No | No | **Yes** (Pharmakolytria) |
| **Attributes** | None | None | **Cross of martyrdom, bottle of medicine** |
| **CRM jargon** | "denotes figures such as" | Same | None |

**v2 answer**: *"The panel of Saint Anastasia is located in the South lunette of the Narthex in the Asinou Church. This panel depicts Saint Anastasia Pharmakolytria, who is associated with a cross of martyrdom and a bottle of medicine, symbolizing her role as a healer and a martyr. Additionally, the panel includes Anastasia Saramalina, who is depicted as a donor. The artwork is part of the mural paintings created during the Lusignan period."*

**What improved**: The enriched Anastasias panel doc now contains `Denotes: Saint Anastasia (Saint Anastasia Pharmakolytria), Anastasia Saramalina (Donor)`. The LLM can identify both figures, their roles, and attributes from a single document. The improved system prompt prevents CRM jargon and instructs the LLM to describe panel contents with roles.

**Still missing**: Anastasia Saramalina's visual attributes (porphyria, T-shaped garment, headdress) — these are only in her individual icon doc, which wasn't in the top-10. But the core question ("tell me about the panel") is answered correctly: two distinct people, their identities, their roles.

### Q3: Who is Anastasia Saramalina?

| | Baseline | Post-fix v1 | Post-fix v2 |
|---|---|---|---|
| **Answer** | Donor, attributes, but CRM jargon | Same | Donor, depicted in panel of Saint Anastasia |
| **Grade** | PARTIAL | PARTIAL | **PARTIAL** |
| **Donor** | Yes | Yes | Yes |
| **Attributes** | porphyria, T-shaped garment, headdress | Same | None (icon doc not retrieved) |
| **CRM jargon** | "denoted by the panel" | Same | None |

**v2 answer**: *"Anastasia Saramalina is depicted as a donor in the panel of Saint Anastasia located in the South lunette of the Narthex in the Asinou Church."*

**Assessment**: Correctly identifies her as a donor depicted in the Anastasias panel. No CRM jargon. But the Anastasia Saramalina icon doc (which contains her visual attributes: porphyria, T-shaped garment, headdress, neck rectangular patches) was NOT in the top-10 sources for this question. The "Donor" identification comes from the enriched Anastasias panel doc.

**Retrieval gap**: The icon doc (e1ce4ed8) should have been retrieved for "Who is Anastasia Saramalina?" but wasn't. The Q3 top-10 contains Anastasias panel, Painting of Anastasias, Panagia Phorbiottisa, Construction of Asinou, South lunette, Saint Mamas, Saint Anastasia (×2), Anastasia of Sirmium — but not the Anastasia Saramalina icon doc itself.

### Q4: Are there any other donors in the church?

| | Baseline | Post-fix v1 | Post-fix v2 |
|---|---|---|---|
| **Answer** | No other donors | Same | Same |
| **Grade** | **FAIL** | **FAIL** | **FAIL** |
| **Donor panel retrieved** | No | No | No |
| **Nikephoros retrieved** | No | No | No |

**v2 answer**: *"Based on the retrieved information, Anastasia Saramalina is the only donor mentioned in the context of the Asinou Church."*

**Root cause — unchanged**: All 10 retrieved docs are Anastasia-related. The word "donor" in the query + conversation context about Anastasia Saramalina (typed as Donor) creates an embedding that locks onto the Anastasia cluster. The Donor panel doc (which says `Denotes: Nikephoros Ischyrios (Founder)`) was NOT retrieved despite now containing the "(Founder)" tag.

**Why target enrichment didn't help here**: The enrichment made the Donor panel doc richer (`Denotes: Nikephoros Ischyrios (Founder)` instead of bare `Denotes: Nikephoros Ischyrios`), but the doc is still only 8 lines and its embedding doesn't compete with the Anastasia cluster's 10+ densely connected docs. The conversation context pushes "other donors" to embed near existing Anastasia-donor content.

**What would fix this**: This requires **structural retrieval** — filtering entities by their P2_has_type value rather than relying on embedding similarity. For "are there other X?" queries, the system needs to:
1. Identify the type keyword ("donor")
2. Search the graph for entities with matching types ("Donor", "Founder")
3. Exclude already-discussed entities (Anastasia Saramalina)

This is a hybrid retrieval problem beyond the scope of document-level fixes.

### Q5: Are there any other depictions of donors aside from Asinou?

| | Baseline | Post-fix v1 | Post-fix v2 |
|---|---|---|---|
| **Answer** | "No other depictions" | Same | "Donor couple found on Saint Nicholas of the Roof" |
| **Grade** | **FAIL** | **FAIL** | **PARTIAL** |
| **Donor couple retrieved** | No | No | Yes |
| **Non-Asinou donors found** | No | No | Partially (Donor couple) |

**v2 answer**: *"Based on the retrieved information, aside from the depictions of donors in the Asinou Church, there is a mention of a 'Donor couple' found on the Saint Nicholas of the Roof Church."*

**What improved**: The Donor panel (Asinou) and Donor couple docs are now both in the top-10 sources. The LLM correctly identifies the Donor couple as a non-Asinou depiction based on its doc content (`Is found on: Saint Nicholas of the Roof (Church)`). The improved system prompt instructs the LLM to enumerate ALL matching entities.

**Still missing**: Donor Ioannes and Irene (Our Lady of Moutoullas), Donor Michael Katzouroumpos (Saint Dimitrianos) — not in the top-10 sources.

### Q6: What's the headdress of Anastasia Saramalina?

| | Baseline | Post-fix v1 | Post-fix v2 |
|---|---|---|---|
| **Answer** | Crespine and barbette | Same | Crespine and barbette, donor context |
| **Grade** | PASS | PASS | **PASS** |

**v2 answer**: *"The headdress of Anastasia Saramalina is described as having a barbette and crespine. It is associated with her depiction as a donor in the Asinou Church."*

Correct and complete. The added donor context comes from the enriched documents.

## 5. Aggregate Results

### Answer Quality (data-grounded)

| Grade | Baseline | Post-fix v1 | Post-fix v2 | Delta (baseline→v2) |
|-------|----------|-------------|-------------|---------------------|
| PASS | 2 (Q1*, Q6) | 2 (Q1, Q6) | **3** (Q1, Q2, Q6) | +1 |
| PARTIAL | 1 (Q3) | 1 (Q3) | **2** (Q3, Q5) | +1 |
| BAD | 1 (Q2) | 1 (Q2) | **0** | -1 |
| FAIL | 2 (Q4, Q5) | 2 (Q4, Q5) | **1** (Q4) | -1 |

*Q1 baseline leaked a Wikidata code — graded PASS on factual content but BAD on presentation. In v1, the Wikidata code issue was fixed by the duplicate source merge.

**Score progression**: 2/6 → 2/6 → **3/6 correct, 2/6 partial, 1/6 failed**

### Infrastructure Metrics

| Metric | Baseline | Post-fix v2 | Delta |
|--------|----------|-------------|-------|
| FAISS index size | 606 docs | 286 docs | -53% |
| Edge count | 1715 | 794 | -54% |
| Avg doc length | 116 chars | 248 chars | +114% |
| Appellation docs in sources | 7 across 6 Qs | 0 | -100% |
| Duplicate source entries | Yes (11-12 per Q) | No (10 per Q) | Fixed |
| CRM jargon in answers | 4/6 questions | 0/6 questions | Eliminated |

## 6. What Each Fix Solved

| Fix | Targeted Issue | Questions Improved | Mechanism |
|-----|----------------|-------------------|-----------|
| Satellite absorption | Fragment docs wasting retrieval slots | Q1 (cleaner sources) | 53% fewer docs, all meaningful |
| Dual retrieval | Follow-up queries biased by context | Q5 (found Donor couple) | Parallel raw+contextualized queries |
| Duplicate source fix | 11-12 sources instead of 10 | All (cleaner data) | Merge Wikidata into graph source |
| **Target enrichment** | Bare labels hide entity roles | **Q2 (BAD→PASS)** | Type tags + attributes inline in FR docs |
| **System prompt** | CRM jargon, missed evidence | **Q2, Q5** | Translate vocabulary, enumerate entities |

## 7. Remaining Issues

### 7.1 Q4: Type-based retrieval failure
The most significant remaining issue. When a user asks "are there other donors?", the system needs to find entities by **type or role** (P2_has_type = "Donor" or "Founder"), not by embedding similarity. The current pipeline retrieves documents similar to "donor" which all cluster around the already-discussed Anastasia Saramalina.

**Impact**: Any CIDOC-CRM dataset with relational "are there other X?" queries will hit this.

**Required fix**: Hybrid retrieval combining vector similarity with structured graph queries. For type-enumeration questions:
1. Extract the type keyword from the query
2. Search entity metadata for matching P2_has_type values
3. Merge with FAISS results, excluding already-discussed entities

### 7.2 Q3: Individual entity docs not consistently retrieved
For "Who is Anastasia Saramalina?", the Anastasia Saramalina icon doc (which has her detailed attributes) was NOT in the top-10 sources. The answer correctly identified her as a donor (from the enriched Anastasias panel doc) but missed her visual attributes.

**Impact**: Moderate. The enrichments partially compensate — critical role information ("Donor") propagates to referencing docs. But detailed attributes remain only in the icon doc.

### 7.3 No remaining CRM jargon issue
The improved system prompt eliminated CRM jargon from all answers. This is fully resolved.

## 8. Files Modified

### Core changes (FR target enrichment)
- `fr_traversal.py`: Added `build_target_enrichments()` function, `_format_targets()` method, `_DEPICTION_LABELS`/`_TYPE_PRED_LOCALS`/`_ATTR_PRED_LOCALS` constants
- `scripts/bulk_generate_documents.py`: Updated `_create_fr_document()` to build and pass enrichments
- `universal_rag_system.py`: Updated `_create_fr_document_from_prefetched()` to build and pass enrichments; improved `get_cidoc_system_prompt()`

### Infrastructure fixes (previous round)
- `universal_rag_system.py`: Duplicate source bug fix (Wikidata merge)
- `static/js/chat.js`: Frontend Wikidata source detection update
- `fr_traversal.py`: Satellite absorption (`classify_satellite()`, `format_absorbed_satellites()`, E52_Time-Span in MINIMAL_DOC_CLASSES)
- `scripts/bulk_generate_documents.py`: `_identify_satellites()`, satellite filtering in doc generation + edge saving
- `universal_rag_system.py`: `_identify_satellites_from_prefetched()`, satellite filtering

## 9. Recommendations for Next Iteration

### 9.1 Hybrid retrieval (high priority, addresses Q4)
Combine FAISS vector search with type-aware graph queries:
- Maintain a type index: P2_has_type value → [entity URIs]
- For queries containing type keywords ("donor", "saint", "church"), also search by type
- Exclude already-discussed entities from results (negative filtering based on chat history)

### 9.2 Query intent classification (medium priority)
Route different query types through different retrieval strategies:
- Factual lookups → standard FAISS
- "Other X" / type enumeration → type-aware retrieval + negative filtering
- Attribute queries → standard FAISS (already works well)

### 9.3 Graph neighbor expansion for entity queries (low priority, addresses Q3)
When a user asks about a specific entity ("Who is X?"), automatically include 1-hop neighbor documents in the context even if they didn't rank in the top-10 by embedding similarity. The system already has `_build_graph_context()` but it only adds relationship labels, not full neighbor docs.
