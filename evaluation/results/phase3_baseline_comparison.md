# Phase 3: Baseline Comparison — Bare LLM vs Multi-Agent System

**Financial Multi-Agent System** · Filippo Maria Incecchi · Università degli Studi di Brescia
Analytics and Data Science for Economics and Management · Prof. Angela Locoro

---

## 1. Objective

This ablation isolates the contribution of the multi-agent architecture itself.
The same 100 questions used in the system evaluation are answered by the **bare
`gemma3:4b` model** — with no router, no specialized agents, no RAG retrieval and
no access to live market or macroeconomic data — and compared against the full
system. The question it answers is: *what does the orchestration + tools + RAG
actually add over the language model alone?*

---

## 2. Method

- **Same 100-question benchmark**, run through the bare model with a minimal
  prompt ("answer as accurately as you can; if unsure of a figure, say so"). No
  tools, no retrieval, no current data — the model can only draw on its training
  memory. Saved separately to `phase3_baseline.jsonl`.
- **Metric reframing.** RAGAS faithfulness / context precision do **not** apply
  to the bare model — there is no retrieved context for it to be faithful to. The
  meaningful measures here are **abstention vs fabrication** (does the model
  admit it cannot know, or invent a figure?) and **accuracy against the real
  values** the system reports.

---

## 3. Headline finding

The bare model is **confidently and silently wrong on everything that requires
current data.** It does not abstain — it fabricates, with false precision, and
it is temporally frozen at its training cutoff without realising it.

- **96 / 100** bare answers are anchored to **2023** (e.g. "As of today,
  November 2, 2023…"); only **1 / 100** referenced the actual current period.
- Across the **40 market + macro questions** — all of which need live data — the
  bare model issued a data disclaimer in only **1 / 40** cases. The other 39
  produced specific, confident, out-of-date figures.
- It even **mimics grounding**: inventing exact timestamps ("3:30 PM PST") and
  citing sources it never queried (Yahoo Finance, the BLS, the Federal Reserve).
  This is arguably worse than abstaining — it looks authoritative while being
  fabricated.

### Abstention vs fabrication, by category

| Category (n) | Explicit data disclaimer | Gave a specific figure |
|---|---|---|
| market (20) | 0 | 17 |
| macro (20) | 1 | 19 |
| rag (20) | 2 | 9 |
| multi_rag_market (10) | 0 | 8 |
| multi_rag_macro (10) | 1 | 9 |
| multi_market_macro (10) | 0 | 10 |
| multi_all (10) | 0 | 10 |
| **Total (100)** | **4** | **— ** |

---

## 4. Bare model vs reality

Every bare figure below is stated as current "as of November 2, 2023" — the
model's training cutoff — and is wrong against the real, current value the
system retrieves (June 2026 data window):

| Question | Bare `gemma3:4b` | System (real, current) | Gap |
|---|---|---|---|
| Apple stock price | $177.75 | $299.24 | −41% |
| Nvidia market capitalization | ~$867B | $5,023B | ~6× too low |
| Microsoft P/E ratio | 32.3 | 23.5 | wrong |
| Tesla P/E ratio | 68.4 | 364.6 | ~5× too low |
| US inflation (CPI YoY) | 3.2% (Oct 2023) | 4.27% (May 2026) | stale |
| Federal Funds rate | 5.25–5.50% | 3.63% | missed the entire 2024–26 cutting cycle |
| Latest US GDP | "Q3 2023, +4.9%" | $31,819B (2026-01) | stale |

The system answers all of these with the correct, current figures because it
fetches them from yfinance and FRED at query time; the bare model cannot, and
does not know that it cannot.

---

## 5. Interpretation

The architecture's value is **not better reasoning** — it is **access to
current, verifiable data and the discipline to use only it.**

- The multi-agent system grounds every quantitative claim in a live source
  (market data, FRED series) or a retrieved filing, and — by design — refuses to
  answer beyond that context.
- The bare model, lacking any data, defaults to confident fabrication anchored to
  a two-and-a-half-year-old world state. For a financial assistant this is the
  worst possible failure mode: not "I don't know", but a precise, plausible,
  wrong number presented as fact.

This is the central justification for the whole system: a small local LLM is
**unsafe to use directly** for data-grounded financial questions, and the
agent + RAG architecture is what makes it usable.

---

## 6. An honest nuance

The bare model is **not** useless everywhere. On the **RAG business-model
questions** it performs reasonably from general memory — it describes Apple's
hardware-plus-services ecosystem (`rag_01`) or Google's advertising-driven model
(`rag_07`) correctly in qualitative terms. What it cannot do is supply the
**current, specific figures** the system cites (e.g. Alphabet's $294.7B 2025
advertising revenue). So the architecture's advantage is **starkest for live
market/macro data and exact filing figures, and more modest for stable
qualitative knowledge**. Stating this plainly keeps the comparison rigorous
rather than overclaiming.

---

## 7. Conclusion

On the same 100 questions, the bare `gemma3:4b` model fabricated current data in
**39 of 40** market/macro questions and anchored **96 of 100** answers to its
2023 training cutoff, with only **4** honest disclaimers across the whole set.
The multi-agent system reports the correct current values instead. The
comparison is the clearest empirical justification for the architecture: the
contribution of the router, the specialized agents and the RAG pipeline is
precisely to replace the bare model's confident, stale fabrication with grounded,
current, verifiable answers.
