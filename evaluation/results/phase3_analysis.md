# Phase 3 — System Evaluation Analysis

**Financial Multi-Agent System** · Filippo Maria Incecchi · Università degli Studi di Brescia
Analytics and Data Science for Economics and Management · Prof. Angela Locoro

---

## 1. Method

- **Dataset:** 100 questions — 20 each for the RAG, Market and Macro agents, and 10 each for the four `multi` combinations (rag+market, rag+macro, market+macro, rag+market+macro). English only; 9 indexed tickers, so no on-demand downloads were triggered.
- **Generation:** the full system (router → agents → synthesizer) on `gemma3:4b`, run in isolation per question (no conversational memory), producing a trace of route, agents fired, final answer and the contexts used. All 100 collected with **0 runtime errors**.
- **Judge:** Claude Opus 4.8 applied the RAGAS metric definitions (faithfulness, answer relevancy, context precision) **holistically in-session**, reading each question's answer against the context the system actually used. This is *LLM-as-judge with the RAGAS rubric*, not the `ragas` Python package — directionally equivalent and zero-cost, but not a third-party-reproducible script. Scores are reported per category as means.
- **Routing/selection accuracy** is exact (computed against the labelled expected route/agent set).
- Macro data window: **24 months** for monthly series, **8 quarters** for GDP — confirming the frequency-aware history change works as intended.

---

## 2. Headline results

| Dimension | Score |
|---|---|
| Routing accuracy | **97 / 100** |
| Agent-selection accuracy | **88 / 100** |
| Faithfulness (mean, judge) | **≈ 0.84** |
| Answer relevancy (mean, judge) | **≈ 0.90** |
| Context precision (mean, judge) | **≈ 0.83** |

The system is **strong on routing and on data-grounded answering (Market, RAG)**; the weak spots are **the Macro agent's numeric reasoning over 24 data points** and **agent selection for the market+macro combination**.

> **Update — re-run after fixes (see §9):** the §7 recommendations were
> implemented and the **macro** and **market+macro** categories re-run. Macro
> routing improved **17/20 → 20/20** and macro faithfulness **≈0.72 → ≈0.92**;
> the market+macro dropped-agent cascade was eliminated (needed-agent recall
> **6/10 → 10/10**). Details and remaining caveats in §9.

---

## 3. Routing & agent selection

| Category | Route acc. | Selection acc. |
|---|---|---|
| rag | 20/20 | 20/20 |
| market | 20/20 | 20/20 |
| macro | 17/20 | 17/20 |
| multi_rag_market | 10/10 | 10/10 |
| multi_rag_macro | 10/10 | 10/10 |
| multi_market_macro | 10/10 | **1/10** |
| multi_all | 10/10 | 10/10 |

**Routing (97%).** Only 3 misses, all pure-macro questions over-classified as `multi`:
- `macro_10` "relationship between inflation and unemployment" → multi (added rag).
- `macro_17` "Has the Fed been raising or cutting rates recently?" → multi (rag+market).
- `macro_19` "How do interest rates compare to a year ago?" → multi (market+macro).

Certain macro phrasings ("relationship between…", "compare to a year ago", "raising or cutting") nudge the 4B router toward `multi`. The most damaging is **`macro_17`**: routed to rag+market, it **never queried FRED**, leaned on stale 10-K boilerplate, and concluded the Fed is *"raising"* rates — the opposite of the data (5.33% → 3.63%). A routing error here cascades directly into a factually wrong answer.

**Selection (88%), dominated by one category.** Routing to `multi` is essentially perfect, and three of the four combinations select agents perfectly (rag+market, rag+macro, all-three: 10/10 each). The market+macro combination is the outlier (1/10): when a question names a company but asks only about its valuation + macro, the selector almost always **adds RAG** (because a ticker is present) and is **inconsistent about dropping a genuinely needed agent**:
- 5/10 → rag+market+macro (rag added, harmless but imprecise)
- 3/10 → rag+macro (**market dropped** — e.g. `mm_03` then can't assess valuation)
- 1/10 → rag+market (**macro dropped** — e.g. `mm_01` fabricates a "hypothetical 4.5% rate")
- 1/10 → market+macro (the only exact match, `mm_10`)

Framed as precision/recall: selection **recall is high** (rarely misses an agent the question needs in the rag/market/all cases) but **precision drops** on market+macro, where the presence of a company name over-triggers RAG and occasionally crowds out the agent that was actually required.

---

## 4. Per-agent findings

### Market agent — strongest component (faithfulness ≈ 0.95, relevancy ≈ 0.98, precision ≈ 1.0)
Every one of the 20 answers extracted the correct ticker and reported the exact figure from its data block. Issues:
- **`market_11`** asserts Apple is "the largest publicly traded company in the world" — not in the context and contradicted by it (NVDA $5,023B > AAPL $4,395B). A clean faithfulness violation (injected world knowledge).
- **META returns show `+nan%`** (`market_06`, `market_15`) — a data-quality bug in the Market agent's return computation (likely a split/history gap), not an answer error; the model correctly ignored it.
- Mild editorialising ("overvalued", "premium") but tied to the provided P/E and returns.

### RAG agent — excellent faithfulness, retrieval is the bottleneck (faithfulness ≈ 0.93, relevancy ≈ 0.85, precision ≈ 0.70)
- **Faithfulness is the system's best quality.** When retrieval failed, the agent **did not hallucinate** — `rag_08` (Intel segments) and `rag_18` (Intel foundry) honestly answered "the provided context does not contain information…", and `rag_09` (Palantir) stuck to the retrieved financial-note chunks. The strict context-only prompt is working.
- **Retrieval precision is the weak link.** For some filings — Intel especially — the embedding (all-MiniLM) surfaced boilerplate (signatures, subsidiary lists, certifications) instead of the business/segment sections, producing "not found" answers to answerable questions. The deterministic ticker filter guarantees the *right company*, but **within-company section relevance** is inconsistent. This is a chunking/embedding issue, not an LLM issue.
- Strong groundings elsewhere: `rag_04` (Amazon segments), `rag_05` (MSFT cloud), `rag_07` (Alphabet revenue table), `rag_12` (Amazon fulfilment) are accurate and well-cited.

### Macro agent — correct data, shaky numeric reasoning (faithfulness ≈ 0.72, relevancy ≈ 0.90, precision ≈ 0.90)
The FRED data pulled is correct and well-formatted (24 months / 8 quarters). The **4B model's reading of it is the problem**:
- **Date misattribution** (recurring): `macro_01` "As of April 2026 … 4.27%" (4.27% is May); `macro_03` "As of February 29, 2026 … 3.63%" (a non-existent date; 3.63% is May).
- **Quarter mislabelling:** `2024-04` repeatedly called "Q4 2024" (it is Q2).
- **Fabricated data point:** `macro_13` invents a "Q2 2026 / February" GDP of `31,985.46` that does not exist in the context (GDP is quarterly and the series ends at 2026-01).
- **Trend-direction error:** `macro_20` calls the rate path a "tightening cycle" when it is plainly a *cutting* cycle.

A plausible interpretation worth a sentence in the thesis: **widening the window from 8 to 24 points improved data coverage but increased the 4B model's attribution/arithmetic errors** — more rows to confuse it. There is a coverage-vs-reasoning trade-off at this model size.

---

## 5. Multi-combination findings

- **rag+market (≈0.82 faithfulness)** and **rag+macro (≈0.85)**: coherent syntheses that genuinely combine sources. Occasional unsupported claims — `rm_02` calls a Tesla revenue *decline* ($97.7B→$94.8B) an "increase"; `rm_07` cites a $402.8B Alphabet total that looks inflated; **`rm_09` confidently attributes Intel's +462% return to its "foundry strategy" although RAG retrieved nothing on it.**
- **Synthesis can erode faithfulness.** In the pure RAG run the agent honestly said "not in the filings" (`rag_18`); inside `multi`, the synthesizer instead *manufactured* a causal story (`rm_09`). The combiner does not inherit the RAG agent's "don't invent" discipline.
- **market+macro (≈0.68 — weakest):** the selection noise cascades. Dropping macro made `mm_01` invent a "hypothetical 4.5%" rate; dropping market made `mm_03` unable to judge valuation; and `mm_07` misreads CPI as "3.8%" (vs 4.27%). Several answers openly state "we lack the data".
- **all-three (≈0.85):** consistently the richest answers — strategy + valuation + macro woven into analyst-style summaries (`all_01`–`all_10`), with only minor generic causal claims and small number drift (e.g. `all_10` P/E 371.2x vs the Market agent's 364.6x).

---

## 6. Cross-cutting themes

1. **Grounded data → reliable answers.** Where the context is a clean data block (Market) or a well-retrieved chunk (most RAG), faithfulness is high. The system's architecture is sound.
2. **The 4B model is the ceiling, not the plumbing.** Routing prompts, ticker resolution, retrieval filtering and data fetching all work; the residual errors are gemma3:4b's reasoning — date attribution, multi-row arithmetic, and over-eager agent selection.
3. **Routing errors cascade.** A single misroute (`macro_17`) produced a flatly wrong answer. Routing is 97% but the 3% failure is high-impact.
4. **Selection precision is the main gap.** market+macro is the only weak category; the selector treats "a company is named" as "RAG is needed."
5. **Retrieval precision limits RAG.** The ticker filter fixed *which company*; the open problem is surfacing the right *section* within a noisy SGML 10-K.

---

## 7. Recommendations (prioritised)

1. **Macro numeric reliability** (biggest quality lever): have the agent compute and pass the "latest period", the YoY/period delta and the trend direction *in code*, and tell the model to quote those fields verbatim — removing the date/arithmetic reasoning from the 4B model (the same deterministic-guardrail philosophy already used for tickers).
2. **Selection precision for market+macro:** add a rule that a company name alone does not require RAG — RAG is for business/strategy/risk questions, not pure valuation; or post-filter the selection so a needed market/macro agent is never dropped.
3. **Macro routing:** the 3 misroutes are the deterministic-guardrail candidate previously discussed (macro keyword + no company/market term ⇒ `macro`).
4. **RAG retrieval precision:** improve 10-K cleaning (strip signatures/exhibits/subsidiary tables before chunking) and/or try a stronger embedding; consider section-aware chunking.
5. **Synthesizer faithfulness:** instruct the synthesizer to preserve "not available in the sources" rather than inventing causal links, and to avoid recomputing figures the agents already provided.
6. **Market data hygiene:** guard the `nan` return case (META) and drop the unsupported "largest company" type claims.

---

## 8. Conclusion

On 100 questions the system routed correctly **97%** of the time, selected the right agent set **88%** of the time, and produced answers that are **relevant (≈0.90)** and **well-grounded (≈0.84 faithfulness)**, with context precision **≈0.83**. The **Market and RAG agents are reliable**; **RAG never hallucinated even when retrieval failed**, which is the single most important property for a financial assistant. The measurable weaknesses are concentrated and explainable: the **Macro agent's reading of multi-row data**, **agent selection on market+macro**, and a small number of **high-impact routing misses** — all attributable to the 4B generator rather than the orchestration, and all addressable with the deterministic-guardrail pattern already proven elsewhere in the system.


---

## 9. Post-fix re-evaluation (macro + market+macro)

After implementing the §7 recommendations, the **macro** (20) and
**market+macro** (10) categories were re-run with `--collect --only` (same
`gemma3:4b` generator, same live data window). The fixes exercised here are the
deterministic macro **SUMMARY** block, the **macro-route guardrail**, and the
multi-agent **selection floor**.

### 9.1 Routing & selection — before → after

| Metric | Before | After |
|---|---|---|
| Macro — routing accuracy | 17/20 | **20/20** |
| Macro — selection accuracy | 17/20 | **20/20** |
| market+macro — routing accuracy | 10/10 | 10/10 |
| market+macro — exact selection `{market, macro}` | 1/10 | **5/10** |
| market+macro — both needed agents present (recall) | 6/10 | **10/10** |

- The three macro questions that previously leaked to `multi` — `macro_10`
  (inflation vs unemployment), `macro_17` (Fed raising/cutting), `macro_19`
  (rates vs a year ago) — now all route to `macro`. The guardrail downgrades a
  pure-macro question (macro term present, no company, no market term) that the
  4B router over-classified as `multi`.
- For market+macro, the selection **floor eliminated every dropped-agent case**:
  all ten now run both the Market and Macro agents. The five non-exact cases now
  merely *add* RAG (harmless over-inclusion) instead of *dropping* a needed
  agent — a far healthier failure mode, and the one that previously caused
  fabrication.

### 9.2 Faithfulness — before → after (judge)

| Category | Before | After |
|---|---|---|
| Macro | ≈ 0.72 | **≈ 0.92** |
| market+macro | ≈ 0.68 | **≈ 0.82** |

**Macro.** Quoting the precomputed `SUMMARY` removed the error class that
dominated the baseline:

- `macro_17` ("Has the Fed been raising or cutting rates?") now answers
  **"cutting / falling"** — correct — versus the baseline's flatly wrong
  "raising". This was the single worst error in the first run; it is fixed.
- `macro_13` no longer **fabricates** a Q2-2026 GDP figure; it quotes the real
  latest value (2026-01 = 31,819.46) and the computed deltas.
- `macro_01` / `macro_03` now report the correct latest **period and value**
  (2026-05) rather than mislabelled dates ("April… 4.27%", "February 29, 2026").
- The "tightening cycle" misread (`macro_20`) is gone — the rate path is
  correctly described as falling / easing.
- *Residual (cosmetic):* the model occasionally still derives an *extra*
  percentage from the absolute deltas with a small slip (e.g. `macro_13` quotes
  ~+1.78% QoQ where it is ≈+1.26%); the SUMMARY figures it quotes are correct.
  A minor interpretive slip also remains ("lower rates → higher borrowing
  costs"). Both are negligible next to the eliminated errors.

**market+macro.** With both agents now guaranteed present, the cascade is gone:

- `mm_01` ("Is Nvidia's P/E justified given rates?") no longer invents a
  "hypothetical 4.5% rate" — it has the real Macro data and uses the actual P/E.
- `mm_03` ("Is Tesla's valuation stretched?") now includes the Market agent and
  can actually report the valuation (price $404.66, market cap $1.52T,
  P/E 371.2x) instead of pleading missing data.
- *Residual (honest limitation):* a few answers (`mm_05`, `mm_10`) still conclude
  that a direct correlation is "impossible to determine". Both agents' data is
  present; the system simply does not *statistically join* the two time series.
  This is an honest capability limit — not a fabrication or a routing/selection
  error — and would require a dedicated cross-series comparison step (future work).

### 9.3 Takeaway

The deterministic-guardrail strategy delivered the predicted gains: macro routing
reached **20/20**, macro faithfulness rose **≈0.72 → ≈0.92**, and the market+macro
cascade (dropped agents → invented figures) was eliminated (needed-agent recall
**6/10 → 10/10**). The residual gaps are now either cosmetic (a stray derived
percentage) or honest capability limits (no cross-series correlation), rather than
the high-impact errors of the baseline. This confirms the §6 conclusion: at this
model size, **moving arithmetic, date attribution, and agent-set decisions out of
the 4B model and into deterministic code is the most effective reliability lever.**

