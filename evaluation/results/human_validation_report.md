# Human validation of the automatic judge

**Financial Multi-Agent System** · Filippo Maria Incecchi · Università degli Studi di Brescia

30 answers rated by hand against the evidence the system used, without sight of the judge's scores.

| Metric | n | Mean absolute difference | Agreement within 0.25 |
|---|---|---|---|
| Faithfulness | 30 | 0.14 | 26/30 |
| Answer relevancy | 30 | 0.26 | 20/30 |

## Per-answer detail

| id | category | metric | human | judge | |Δ| | judge error |
|---|---|---|---|---|---|---|
| rag_02 | rag | faithful | 1.00 | 0.00 | 1.00 |  |
| rag_02 | rag | relevanc | 1.00 | 0.78 | 0.22 |  |
| rag_14 | rag | faithful | 1.00 | 1.00 | 0.00 |  |
| rag_14 | rag | relevanc | 0.50 | 1.00 | 0.50 |  |
| rag_03 | rag | faithful | 1.00 | 1.00 | 0.00 |  |
| rag_03 | rag | relevanc | 1.00 | 0.87 | 0.13 |  |
| rag_08 | rag | faithful | 0.00 | 1.00 | 1.00 |  |
| rag_08 | rag | relevanc | 0.00 | 0.66 | 0.66 |  |
| rag_09 | rag | faithful | 1.00 | 1.00 | 0.00 |  |
| rag_09 | rag | relevanc | 0.50 | 0.91 | 0.41 |  |
| rag_19 | rag | faithful | 1.00 | 0.83 | 0.17 |  |
| rag_19 | rag | relevanc | 1.00 | 0.84 | 0.16 |  |
| market_02 | market | faithful | 1.00 | 1.00 | 0.00 |  |
| market_02 | market | relevanc | 1.00 | 0.95 | 0.05 |  |
| market_20 | market | faithful | 1.00 | 1.00 | 0.00 |  |
| market_20 | market | relevanc | 1.00 | 0.91 | 0.09 |  |
| market_03 | market | faithful | 1.00 | 1.00 | 0.00 |  |
| market_03 | market | relevanc | 1.00 | 0.94 | 0.06 |  |
| market_07 | market | faithful | 1.00 | 1.00 | 0.00 |  |
| market_07 | market | relevanc | 1.00 | 0.91 | 0.09 |  |
| market_14 | market | faithful | 1.00 | 1.00 | 0.00 |  |
| market_14 | market | relevanc | 1.00 | 1.00 | 0.00 |  |
| market_18 | market | faithful | 1.00 | 1.00 | 0.00 |  |
| market_18 | market | relevanc | 1.00 | 0.94 | 0.06 |  |
| macro_02 | macro | faithful | 1.00 | 1.00 | 0.00 |  |
| macro_02 | macro | relevanc | 1.00 | 1.00 | 0.00 |  |
| macro_15 | macro | faithful | 1.00 | 0.88 | 0.12 |  |
| macro_15 | macro | relevanc | 1.00 | 0.82 | 0.18 |  |
| macro_03 | macro | faithful | 1.00 | 0.50 | 0.50 |  |
| macro_03 | macro | relevanc | 1.00 | 0.91 | 0.09 |  |
| macro_08 | macro | faithful | 1.00 | 1.00 | 0.00 |  |
| macro_08 | macro | relevanc | 1.00 | 0.84 | 0.16 |  |
| macro_12 | macro | faithful | 1.00 | 1.00 | 0.00 |  |
| macro_12 | macro | relevanc | 1.00 | 0.78 | 0.22 |  |
| macro_19 | macro | faithful | 1.00 | 1.00 | 0.00 |  |
| macro_19 | macro | relevanc | 1.00 | 0.77 | 0.23 |  |
| rm_06 | multi_rag_market | faithful | 1.00 | 1.00 | 0.00 |  |
| rm_06 | multi_rag_market | relevanc | 1.00 | 0.98 | 0.02 |  |
| rm_05 | multi_rag_market | faithful | 1.00 | 0.83 | 0.17 |  |
| rm_05 | multi_rag_market | relevanc | 0.50 | 0.00 | 0.50 |  |
| rm_07 | multi_rag_market | faithful | 1.00 | 1.00 | 0.00 |  |
| rm_07 | multi_rag_market | relevanc | 0.00 | 0.68 | 0.68 |  |
| rmac_10 | multi_rag_macro | faithful | 0.80 | 0.60 | 0.20 |  |
| rmac_10 | multi_rag_macro | relevanc | 0.00 | 0.61 | 0.61 |  |
| rmac_02 | multi_rag_macro | faithful | 1.00 | 1.00 | 0.00 |  |
| rmac_02 | multi_rag_macro | relevanc | 1.00 | 0.73 | 0.27 |  |
| rmac_07 | multi_rag_macro | faithful | 1.00 | 0.91 | 0.09 |  |
| rmac_07 | multi_rag_macro | relevanc | 1.00 | 0.86 | 0.14 |  |
| mm_01 | multi_market_macro | faithful | 1.00 | 1.00 | 0.00 |  |
| mm_01 | multi_market_macro | relevanc | 0.00 | 0.86 | 0.86 |  |
| mm_02 | multi_market_macro | faithful | 1.00 | 0.20 | 0.80 |  |
| mm_02 | multi_market_macro | relevanc | 0.00 | 0.00 | 0.00 |  |
| mm_05 | multi_market_macro | faithful | 1.00 | 1.00 | 0.00 |  |
| mm_05 | multi_market_macro | relevanc | 0.00 | 0.52 | 0.52 |  |
| all_06 | multi_all | faithful | 1.00 | 1.00 | 0.00 |  |
| all_06 | multi_all | relevanc | 1.00 | 0.77 | 0.23 |  |
| all_04 | multi_all | faithful | 0.86 | 0.86 | 0.00 |  |
| all_04 | multi_all | relevanc | 1.00 | 0.79 | 0.21 |  |
| all_10 | multi_all | faithful | 1.00 | 0.85 | 0.15 |  |
| all_10 | multi_all | relevanc | 1.00 | 0.68 | 0.32 |  |

## Reading

This is a small sample and is not intended to validate the judge in general. It checks whether, on answers a reader can verify, the automatic scores are consistent with the intended interpretation of the metric. Disagreements are as informative as agreements and are listed individually above.

The two metrics are not equally comparable. Faithfulness was rated by counting supported claims over total claims, which is how RAGAS constructs the metric, so human and judge values are measured the same way. Answer relevancy is computed by RAGAS from embedding similarity, a procedure a human cannot reproduce; it was rated here on a three-point scale, so a larger divergence is expected by construction and should not be read as disagreement about the answers themselves.
