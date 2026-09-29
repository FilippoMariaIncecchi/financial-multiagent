# Phase 3: Conversational-Memory Ablation

**Financial Multi-Agent System** · Filippo Maria Incecchi · Università degli Studi di Brescia
Analytics and Data Science for Economics and Management · Prof. Angela Locoro

---

## 1. Objective

This ablation isolates the contribution of the **conversational memory**. The
system is run over a set of multi-turn dialogues twice — once with memory ON and
once with memory OFF — while everything else is held fixed, so any difference is
attributable to memory alone.

---

## 2. Method

- **8 dialogues, 22 turns, 14 follow-up turns.** Each dialogue opens with an
  explicit company question ("What is Apple's business model?") and continues
  with follow-ups that refer to that company **only through a reference** —
  "And its current stock price?", "How has it performed over the past year?" —
  never by name. Resolving such a follow-up therefore *requires* the history.
- **Metric — reference-resolution accuracy.** For each follow-up the expected
  ticker (the referent) must appear among the tickers the system actually
  resolves and uses. Scored on the follow-up turns, memory ON vs OFF.
- **Fair isolation.** The dialogues avoid follow-ups that name a *second*
  company ("compare it to Microsoft"), because the deterministic ticker resolver
  short-circuits on an explicit name and would not exercise the reference — that
  would confound the test. Pure-reference follow-ups are the clean measure.

---

## 3. Result

| Reference resolution (follow-up turns) | Memory ON | Memory OFF |
|---|---|---|
| Resolved correctly | **14 / 14 (100%)** | **0 / 14 (0%)** |

The **routes chosen are identical** in both modes (the router still sends "its
stock price?" to the Market agent); the only thing that changes is whether the
system can recover *which* company the reference points to. With memory it
recovers the correct ticker every time; without it, never.

Illustrative turns (expected → resolved):

| Follow-up | Expected | Memory ON | Memory OFF |
|---|---|---|---|
| "And what is its current stock price?" (after Apple) | AAPL | AAPL ✅ | — ❌ |
| "What is its current P/E ratio?" (after Tesla) | TSLA | TSLA ✅ | — ❌ |
| "What is its market capitalization?" (after Nvidia) | NVDA | NVDA ✅ | — ❌ |
| "And its 52-week high?" (after Meta) | META | META ✅ | TSLA ❌ |
| "Is it overvalued given current interest rates?" (after Tesla) | TSLA | TSLA ✅ | "YOU" ❌ |

---

## 4. Interpretation

Two points stand out.

**Memory is necessary and sufficient for follow-up resolution.** Because routing
is unchanged, the jump from 0/14 to 14/14 is caused purely by the memory
injecting the prior turns so the ticker extractor can resolve "it"/"its". This
is a clean, quantitative demonstration that the feature does exactly what it is
meant to do.

**Without memory the failure is not merely "can't answer" — sometimes it is
"answers about the wrong company".** In most OFF cases the ticker extractor
correctly returns nothing (the reference is unresolvable), but in three cases it
*fabricated* a ticker — resolving to `TSLA` twice and to the non-ticker `YOU`
once. A memoryless multi-turn assistant is therefore not just unhelpful on
follow-ups; it can silently attribute an answer to the wrong entity, which for a
financial assistant is a genuine correctness risk. Memory removes this failure
mode entirely in the tested dialogues.

---

## 5. Limitations

- **Scope of the reference type.** By design the follow-ups are single-entity
  references. Multi-entity follow-ups that introduce a new company by name
  ("compare it to Microsoft") are out of scope, because the deterministic
  resolver prioritises the explicit name over the remembered referent — a
  known, documented behaviour rather than a memory failure.
- **Sample size.** Eight dialogues / fourteen follow-ups is a focused
  demonstration, not a large-scale benchmark; the 100% vs 0% split is
  categorical enough to be convincing at this size, but a larger dialogue set
  would tighten the estimate.

---

## 6. Conclusion

On 14 memory-dependent follow-up turns, the system resolves the referenced
company **100% of the time with conversational memory and 0% without it**, with
routing held constant — an unambiguous isolation of the memory contribution.
Beyond the accuracy gap, memory also eliminates a dangerous failure mode in
which the memoryless system fabricates or mis-attributes the company a follow-up
is about. Conversational memory is thus what makes the assistant usable across
turns rather than only in single-shot question answering.
