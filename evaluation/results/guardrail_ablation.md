# Guardrail ablation — raw LLM decision vs deterministic layer

**Financial Multi-Agent System** · Filippo Maria Incecchi · Università degli Studi di Brescia

Same 100-question benchmark, same model (`gemma3:4b`, temperature 0). The columns differ only in whether the deterministic guardrails are applied to the model's decision.

## Overall

| Decision | Raw LLM | After guardrails | Δ |
|---|---|---|---|
| Routing accuracy | 97/100 | 100/100 | +3 |
| Agent selection (exact) | 95/100 | 95/100 | +0 |
| Needed-agent recall | 99/100 | 100/100 | +1 |

## By category

| Category | n | Routing raw → final | Selection raw → final | Recall raw → final |
|---|---|---|---|---|
| rag | 20 | 20 → 20 | 20 → 20 | 20 → 20 |
| market | 20 | 20 → 20 | 20 → 20 | 20 → 20 |
| macro | 20 | 17 → 20 | 20 → 20 | 20 → 20 |
| multi_rag_market | 10 | 10 → 10 | 10 → 10 | 10 → 10 |
| multi_rag_macro | 10 | 10 → 10 | 10 → 10 | 10 → 10 |
| multi_market_macro | 10 | 10 → 10 | 5 → 5 | 9 → 10 |
| multi_all | 10 | 10 → 10 | 10 → 10 | 10 → 10 |

## Cases modified by the guardrails (4)

| id | question | raw | final |
|---|---|---|---|
| macro_10 | What is the relationship between recent US inflation and une | route=multi agents=None | route=macro agents=['macro'] |
| macro_17 | Has the Fed been raising or cutting rates recently? | route=multi agents=None | route=macro agents=['macro'] |
| macro_19 | How do current US interest rates compare to a year ago? | route=multi agents=None | route=macro agents=['macro'] |
| mm_03 | Given the Federal Funds rate, is Tesla's valuation stretched | route=multi agents=['rag', 'macro'] | route=multi agents=['rag', 'market', 'macro'] |

## Ticker resolution path

- resolved deterministically: **55/100**
- fell back to the LLM: **45/100**
