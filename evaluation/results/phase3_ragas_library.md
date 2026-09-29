# Phase 3: RAGAS Library Scoring (Reproducible)

**Financial Multi-Agent System** · Filippo Maria Incecchi · Università degli Studi di Brescia
Analytics and Data Science for Economics and Management · Prof. Angela Locoro

---

## 1. Objective

The main evaluation used an LLM-as-judge applying the RAGAS *rubric* in-session.
To obtain **script-reproducible** metrics for the methodology, the same 100
system traces were scored with the actual **`ragas` Python library**.

---

## 2. Method

- **Library:** `ragas` (v0.2), reference-free metrics — Faithfulness,
  Answer Relevancy, and LLM Context Precision (without reference).
- **Judge LLM:** Claude **Haiku 4.5** via the Anthropic API.
- **Embeddings:** local `nomic-embed-text` (Ollama) for Answer Relevancy.
- **Input:** the 100 saved traces (`phase3_traces.jsonl`) — same answers and
  contexts the system produced; the generator is not re-run.

---

## 3. Results (RAGAS library, Haiku judge)

| Category | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|
| rag | 0.965 | 0.764 | 0.386 |
| market | 0.851 | 0.842 | 1.000 |
| macro | 0.512 | 0.808 | 1.000 |
| multi_rag_market | 0.402 | 0.314 | 0.058 |
| multi_rag_macro | 0.274 | 0.444 | 0.067 |
| multi_market_macro | 0.324 | 0.075 | 0.073 |
| multi_all | — (n/a) | 0.490 | 0.000 |
| **OVERALL** | **0.663** | **0.615** | **0.497** |

Faithfulness means exclude cells the judge could not compute: **20 of 100
returned `NaN`**, concentrated in the long multi-agent answers — **all 10
`multi_all`** (so its faithfulness is effectively unmeasured), 4
`multi_market_macro`, 3 `multi_rag_macro`, 1 `multi_rag_market`, 2 `macro`.
Answer Relevancy and Context Precision computed on all 100.

---

## 4. Library vs. rubric-based judgment

| Metric (overall) | Rubric judge (holistic) | RAGAS library (Haiku) |
|---|---|---|
| Faithfulness | ≈ 0.89 | 0.66 |
| Answer Relevancy | ≈ 0.90 | 0.62 |
| Context Precision | ≈ 0.83 | 0.50 |

The library is **systematically stricter**, exactly as expected: it decomposes
each answer into atomic claims and checks every one against the context, and it
judges the relevance of every retrieved chunk individually — whereas the rubric
judge scored each answer holistically. The gap is not a contradiction; it
reflects two different granularities of the same metrics.

---

## 5. Interpretation

**The pattern across categories is the real result.**

- **Single-agent RAG faithfulness is very high (0.965).** Short, extractive,
  strictly-grounded answers score well even under claim-level checking.
- **Market (0.851) and macro (0.512) are lower because the answers add
  interpretation.** By design the Market and Macro prompts ask for "investor
  implications" / analysis ("businesses may…", "investors could…"). RAGAS counts
  each such advisory sentence as a claim that is *not* in the FRED/market data,
  so faithfulness drops — even though the underlying **figures are correct**
  (verified manually: prices, FRED values and filing numbers are accurate).
- **The multi-agent syntheses score lowest (0.27–0.40, plus many NaN).** The
  synthesizer produces long, analyst-style answers that elaborate well beyond
  the retrieved context and weave in connections and recommendations — strict
  claim-level faithfulness penalizes this heavily, and the length also caused the
  judge to fail (NaN) on the longest answers.
- **Context Precision confirms retrieval is the weak link.** RAG = 0.386: the
  top-5 chunks often include tangential passages (culture, compensation,
  boilerplate) alongside the relevant one. Market/macro = 1.000 (a single,
  always-relevant data block). Multi ≈ 0.06: the context is dominated by the
  many RAG chunks, most tangential to a broad multi-part question.
- **Answer Relevancy** is solid for single agents (0.76–0.84) and low for multi
  (0.07–0.49): the long, multi-faceted syntheses drift from the specific question.

**The crucial distinction for the thesis:** a faithfulness of 0.66 does **not**
mean the system invents data one third of the time. It does not fabricate
figures — the baseline comparison and manual checks show its numbers are
correct and current. The strict score is lowered by two things: (a) the
*interpretive* sentences the system is deliberately prompted to add, which the
metric treats as ungrounded claims, and (b) 20 unmeasured (NaN) long answers.
"No hallucinated data" and "every sentence is a context-grounded claim" are
different bars; the system clears the first strongly and the second only
partially.

---

## 6. Caveats

- **20 NaN cells** (all of `multi_all`) mean multi faithfulness is partly
  unmeasured — a limitation of a small, cheap judge on very long answers, not a
  system failure. A stronger judge or shorter answers would recover them.
- The metric **penalizes interpretation**: the low market/macro/multi scores are
  driven largely by advisory analysis the system is designed to provide, not by
  factual errors.
- Reference-free Context Precision is strict on multi contexts because the RAG
  chunks (many, some tangential) dominate the context list.

---

## 7. What it points to

Two concrete, defensible improvements follow directly from the library scores:

1. **Tighten the synthesizer / agent prompts** to stay closer to the retrieved
   facts and add less free-standing interpretation — this is the single biggest
   lever on strict faithfulness and answer relevancy for the multi categories.
2. **Improve retrieval precision** (stronger embedding or a reranker) so the
   top-k chunks are on-topic — this lifts Context Precision, most of all for RAG
   and the multi combinations.

---

## 8. Conclusion

The reproducible `ragas` library, judged by Claude Haiku 4.5, reports overall
faithfulness **0.66**, answer relevancy **0.62** and context precision **0.50** —
lower than the holistic rubric judgment because it scores at the atomic-claim
and per-chunk level. The breakdown is the value: **single-agent, data-grounded
answers are strong** (RAG faithfulness 0.965), while the **verbose multi-agent
syntheses and imprecise retrieval** are where strict metrics fall — pointing to a
tighter synthesizer and better retrieval as the next improvements. Importantly,
the low strict-faithfulness reflects added *interpretation*, not fabricated data:
the system's figures remain correct and current.
