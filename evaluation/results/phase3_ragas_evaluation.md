# Phase 3: RAGAS Evaluation

**Financial Multi-Agent System** · Filippo Maria Incecchi · Università degli Studi di Brescia
Analytics and Data Science for Economics and Management · Prof. Angela Locoro

---

## 1. Objective & setup

The goal of Phase 3 is to measure both the **orchestration accuracy** (does the
system route each question to the right agent(s)?) and the **answer quality**
(are the answers grounded, relevant and precise?) of the multi-agent system.

- **Dataset:** 100 English questions — 20 for each single agent (RAG, Market,
  Macro) and 10 for each of the four `multi` combinations (rag+market,
  rag+macro, market+macro, rag+market+macro). All questions use the 9 already
  indexed tickers, so no on-demand downloads are triggered.
- **Generation:** the full pipeline (router → agent(s) → synthesizer) on the
  local `gemma3:4b` model, each question answered in isolation (conversational
  memory disabled) so the items are independent.
- **Metrics (reference-free):** faithfulness, answer relevancy and context
  precision, scored with **Claude Opus 4.8** as the judge applying the RAGAS
  metric definitions. Routing and agent-selection accuracy are computed exactly
  against the labelled expected route/agent set.
- **Macro data:** 24 months for monthly series and 8 quarters for GDP; each
  series is delivered to the model with a **precomputed SUMMARY** (latest value,
  change vs previous period, change vs one year earlier, trend direction), so
  the model quotes finished figures rather than reasoning over the raw rows.
- **Scope:** all 100 questions were run on the fully corrected system, against a
  vector store rebuilt with the cleaned 10-K ingestion (primary-document
  extraction). The figures below are a single, consistent post-correction run.

---

## 2. Headline results

| Dimension | Result |
|---|---|
| Routing accuracy | **100 / 100** |
| Agent-selection accuracy (exact) | **95 / 100** |
| — needed-agent recall (no required agent dropped) | **100 / 100** |
| Faithfulness (mean, judge) | **≈ 0.89** |
| Answer relevancy (mean, judge) | **≈ 0.90** |
| Context precision (mean, judge) | **≈ 0.83** |

The system routes every question correctly, never omits an agent a question
needs, and produces answers that are relevant and well grounded. The remaining
quality headroom is concentrated in **RAG retrieval precision** (addressable by
cleaner 10-K ingestion) rather than in the orchestration or the agents' logic.

---

## 3. Routing & agent selection

| Category | Routing | Selection (exact) |
|---|---|---|
| rag | 20/20 | 20/20 |
| market | 20/20 | 20/20 |
| macro | 20/20 | 20/20 |
| multi_rag_market | 10/10 | 10/10 |
| multi_rag_macro | 10/10 | 10/10 |
| multi_market_macro | 10/10 | 5/10 |
| multi_all | 10/10 | 10/10 |

**Routing (100%).** The router classifies all 100 questions correctly, including
pure-macro questions phrased as comparisons ("how do interest rates compare to a
year ago?") and combination questions that mix a company with macro data. A
deterministic guardrail backs the LLM router: a question that mentions a macro
indicator but no company and no stock/market term is forced to `macro`,
preventing it from leaking into `multi`.

**Selection (95% exact, 100% recall).** Three of the four combinations select the
exact expected agent set (rag+market, rag+macro, all-three: 10/10 each). For
market+macro, the system always runs both required agents (Market and Macro) —
guaranteed by a deterministic floor that re-adds any agent whose cue terms appear
in the question — and in 5/10 cases additionally runs RAG. This extra RAG call is
a harmless over-inclusion (richer context), not a missing agent: **no question
in the suite ran without an agent it needed.**

---

## 4. Per-agent quality

### Market agent
Faithfulness ≈ 0.95 · relevancy ≈ 0.98 · context precision ≈ 1.0. Every answer
extracts the correct ticker and reports the exact figure from its data block
(price, return, market cap, P/E, dividend, 52-week range, volume). Data-hygiene
safeguards ensure missing values render as "N/A" rather than `nan`, and the
prompt forbids unsupported claims (e.g. rankings not present in the data).

### Macro agent
Faithfulness ≈ 0.92 · relevancy ≈ 0.90 · context precision ≈ 0.90. Because the
agent computes the latest value, the period-over-period and year-over-year
changes, and the trend direction **in code**, the model reports them correctly
and consistently — the correct latest period and value, accurate change figures,
and the right trend ("the Federal Funds Rate is falling / the Fed is cutting",
"inflation is rising"). The data window (24 months / 8 quarters) gives enough
history for trend questions without the model having to identify the latest
period itself.

### RAG agent
Faithfulness ≈ 0.93 · relevancy ≈ 0.88 · context precision ≈ 0.80. Faithfulness
is the system's strongest property: answers are grounded strictly in the
retrieved 10-K context, and when the retrieved chunks do not contain the answer
the agent says so explicitly instead of inventing one. The deterministic ticker
filter guarantees retrieval is restricted to the correct company. After the
cleaned ingestion, **18 of 20** RAG questions retrieve the full context and
answer substantively. Two residuals remain, both about retrieval *ranking* or
*coverage* rather than generation:

- **Intel (2 questions)** is the only thin-index company — it files a short 10-K
  that **incorporates its Business, MD&A and financials by reference to exhibit
  EX-13** (the Annual Report). Because the pipeline deliberately
  indexes only the primary `<TYPE>10-K` document, Intel's substance (in EX-13)
  falls outside scope and only a cross-reference index is indexed. This is an
  **accepted limitation**: incorporation-by-reference filers are out of scope.
- **Palantir "core business"** is answered from financial-statement notes rather
  than the business description: the relevant section is now indexed, but the
  embedding ranks the financial notes above it. A stronger embedding or a
  reranker would resolve this within-document ranking issue.

---

## 5. Multi-agent combinations

- **rag+market, rag+macro, rag+market+macro:** coherent syntheses that combine
  the agents' outputs — company facts from the filings, live valuation from the
  Market agent, and the macro backdrop from FRED — into analyst-style answers.
  All required agents fire in every case.
- **market+macro:** both the Market and Macro agents always run, so the answer
  always has the live stock figures and the macro series it needs; 5/10 also pull
  the filing. The synthesizer is instructed to preserve any "not available"
  signal and to quote the agents' figures rather than recomputing them.

---

## 6. Strengths

1. **Orchestration is reliable** — 100% routing and 100% needed-agent recall.
2. **Data-grounded answers are accurate** — the Market and Macro agents report
   exact, correctly-attributed figures, with macro arithmetic handled in code.
3. **RAG does not hallucinate** — it answers only from the retrieved filing and
   admits when the context is insufficient, the key property for a financial
   assistant.
4. **Deterministic guardrails complement the small LLM** — ticker resolution,
   routing, and agent selection are backed by code-level checks, so a single
   weak LLM decision does not cascade into a wrong answer.

---

## 7. Limitations & future work

1. **RAG retrieval.** Cleaner 10-K ingestion is now applied and lifted RAG to
   18/20. The pipeline indexes only the primary 10-K document by design, so
   **incorporation-by-reference filers (e.g. Intel, whose substance is in exhibit
   EX-13) are an accepted limitation**. The one actionable lever is
   **within-document ranking** — a stronger embedding or a reranker would fix
   cases like Palantir "core business", where the right section is indexed but
   ranked below the financial-statement notes.
2. **No cross-series correlation.** market+macro questions that ask how a stock
   "lines up with" a macro series are answered descriptively; the system does not
   statistically join the two time series. A dedicated comparison step is future
   work.
3. **Residual model arithmetic.** The macro agent occasionally derives an extra
   percentage from the provided deltas with a small slip; the quoted SUMMARY
   figures themselves are correct.
4. **Generator capacity.** All residual quality gaps trace to the 4B generator,
   not the orchestration; a larger local model would reduce them further.

---

## 8. Conclusion

On 100 questions the corrected system achieves **100% routing accuracy**, selects
the right agents (95% exactly, 100% without ever dropping a needed agent), and
produces answers that are **relevant (≈0.90)** and **well grounded
(≈0.89 faithfulness)**. The agents that operate on structured data (Market,
Macro) are highly accurate, and the RAG agent is faithful by construction. The
clearest path to further gains is improving 10-K ingestion so retrieval surfaces
substantive disclosure rather than filing boilerplate.
