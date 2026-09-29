# Indice dei risultati sperimentali

**Financial Multi-Agent System** · Filippo Maria Incecchi · Università degli Studi di Brescia

Ogni esperimento produce fino a tre file con lo stesso prefisso:

| suffisso | che cosa contiene |
|---|---|
| `*.jsonl` | le tracce grezze — una riga per domanda con route, agenti, risposta ed evidenze usate |
| `*_scores.csv` | il punteggio RAGAS per singola domanda |
| `*_summary.csv` | le medie per categoria e complessive |

---

## 1. Sistema completo — il riferimento di tutto

| file | contenuto |
|---|---|
| `phase3_traces.jsonl` | 100 risposte del sistema completo, con le evidenze |
| `phase3_scores.csv` | punteggi RAGAS per domanda |
| `phase3_summary.csv` | **medie per categoria** — faithfulness 0,874 · answer relevancy 0,733 · context precision 0,536 |
| `phase3_analysis.md` | lettura dei risultati |
| `phase3_ragas_evaluation.md`, `phase3_ragas_library.md` | note sul protocollo di scoring |
| `phase3_improvement_before_after.md` | le tre revisioni successive del sistema (§6.3) |

## 2. Modello nudo — senza architettura e senza dati

| file | contenuto |
|---|---|
| `phase3_baseline.jsonl` | risposte di gemma3:4b da solo |
| `phase3_baseline_comparison.md` | 96/100 ancorate al 2023 · 39/40 dati inventati |

## 3. Agente singolo con le stesse evidenze — isola la decomposizione

| file | contenuto |
|---|---|
| `single_agent_baseline.jsonl` | 100 risposte, un'unica chiamata, tutte le evidenze insieme |
| `single_agent_baseline_scores.csv` | punteggi per domanda |
| `single_agent_baseline_summary.csv` | **faithfulness 0,909 · relevancy 0,768 · context precision 0,161** |

Confronto con il sistema completo: fedeltà equivalente, precisione di retrieval 0,161 contro 0,536.

## 4. Ablazione dei guardrail deterministici

### Guardrail sulle decisioni (router e soglia minima di agenti)

| file | contenuto |
|---|---|
| `guardrail_ablation.jsonl` | decisione grezza del modello contro decisione finale, per domanda |
| `guardrail_ablation.md` | **routing 97→100 · recall agenti 99→100** · i 4 casi corretti · ticker risolti in modo deterministico 55/100 |

### Guardrail sulle evidenze (uno alla volta)

| file | contenuto |
|---|---|
| `ablation_no_macro_summary_*` | senza riepilogo macro precalcolato: **macro 0,518** (da 0,865) · **mercato+macro 0,314** (da 0,753) |
| `ablation_no_ticker_scan_*` | senza identificazione deterministica delle aziende: **filings 0,750** (da 0,918) |
| `ablation_no_nan_safe.jsonl`, `ablation_nan_safe.md` | formattazione NaN-safe: nessun effetto misurabile (0/50 in entrambe) |
| `corpus_contamination.md` | **la contaminazione del corpus** causata dall'ablazione del ticker |
| `corpus_snapshot_before_cleanup.json`, `tickers_before_ablation.txt` | lo stato del corpus prima della pulizia, conservato come prova |

## 5. Ablazione della memoria conversazionale

| file | contenuto |
|---|---|
| `phase3_multiturn.jsonl`, `phase3_traces_prefix.jsonl` | dialoghi a più turni |
| `phase3_memory_ablation.md` | risultati con e senza memoria |
| `phase3_summary_prefix.csv` | medie della variante senza memoria |

## 6. Validazione manuale del giudice automatico

| file | contenuto |
|---|---|
| `human_validation.csv` | **i 30 giudizi manuali** — è il file canonico |
| `human_validation_30.xlsx` | la griglia compilata a mano |
| `human_validation_sheet.md` | le risposte con le evidenze, usate per valutare |
| `human_validation_report.md` | report generato da `--compare` |
| `human_validation_10_backup.csv` | i primi 10 giudizi, prima dell'estensione |

Risultato: scarto medio 0,140 sulla faithfulness (0,086 esclusi i due errori del giudice, 0,052 escludendo anche `rag_08`), 26/30 entro 0,25, 20 coincidenze esatte.

## 7. Tentativo scartato — sintetizzatore riscritto

| file | contenuto |
|---|---|
| `multi_revised_synth_*` | faithfulness complessiva 0,692 contro 0,874: peggiorativo, la modifica è stata annullata |

---

## Come rigenerare

```bash
python evaluation/phase3_ragas_eval.py                  # sistema completo
python evaluation/phase3_ragas_eval.py --baseline       # modello nudo
python evaluation/single_agent_baseline.py              # agente singolo con strumenti
python evaluation/guardrail_ablation.py                 # guardrail sulle decisioni
python evaluation/guardrail_ablation_full.py            # guardrail sulle evidenze
python evaluation/score_any.py --file <traces.jsonl> --label "<etichetta>"
python evaluation/human_validation.py --extend 30
python evaluation/human_validation.py --compare
python evaluation/corpus_cleanup.py --audit
```

Generazione in locale con `gemma3:4b` via Ollama a temperatura zero; scoring con Claude Haiku
attraverso l'API Anthropic. Le tracce sono salvate mano a mano, quindi una sessione interrotta
si può riprendere.
