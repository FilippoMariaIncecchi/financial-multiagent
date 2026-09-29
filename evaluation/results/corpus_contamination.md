# Corpus contamination from LLM ticker extraction

**Financial Multi-Agent System** · Filippo Maria Incecchi · Università degli Studi di Brescia

Snapshot taken 2026-09-15, before cleanup. 36 distinct companies, 22469 indexed chunks.

## What happened

During the ablation of the deterministic ticker guardrail, every question was routed through the LLM ticker extractor. The extractor returned ordinary English words as ticker symbols, and the on-demand ingestion pipeline resolved each one against SEC EDGAR, downloading and indexing the 10-K of whichever real company owns that symbol.

The failure is therefore not a wrong answer but a persistent corruption of the retrieval corpus: the spurious filings remain indexed and available to later queries until removed by hand.

## Corpus composition at snapshot time

| Group | Tickers | Chunks |
|---|---|---|
| Benchmark companies | 9 | 4104 |
| In the alias catalogue (legitimate on-demand downloads) | 11 | 8548 |
| Spurious (not requested by any question) | 16 | 9817 |

## Spurious entries

| Ticker | Company the symbol belongs to | Chunks |
|---|---|---|
| `A` | Agilent Technologies | 803 |
| `ACI` | Albertsons | 19 |
| `APPS` | Digital Turbine | 515 |
| `BE` | Bloom Energy | 885 |
| `HAS` | Hasbro | 589 |
| `IOT` | Samsara | 746 |
| `KEY` | KeyCorp | 770 |
| `LAKE` | Lakeland Industries | 474 |
| `MU` | Micron Technology | 509 |
| `NET` | Cloudflare | 826 |
| `OPEN` | Opendoor | 736 |
| `RUN` | Sunrun | 695 |
| `S` | SentinelOne | 659 |
| `T` | AT&T | 472 |
| `VSTS` | Vestis | 543 |
| `YOU` | Clear Secure | 576 |

Most of these symbols are common English words appearing in the questions themselves, which is what makes the failure mode systematic rather than incidental.

## Action taken

The spurious entries were removed and the corpus restricted to the nine benchmark companies, so that the results reported in Chapter 6 remain reproducible. This snapshot preserves the evidence.
