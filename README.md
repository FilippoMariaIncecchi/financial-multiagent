# A Multi-Agent LLM Architecture for Financial Question Answering Based on a Local Model

Code and evaluation data for the master's thesis *A Multi-Agent LLM Architecture for
Financial Question Answering Based on a Local Model* — Filippo Maria Incecchi,
Analytics and Data Science for Economics and Management, Università degli Studi di
Brescia, supervisor Prof. Angela Locoro.

The system answers financial questions from three kinds of evidence — SEC 10-K filings,
live market data and macroeconomic series — using a four-billion-parameter open-weight
model running locally. Nothing leaves the machine except the calls to the data providers.

## What this repository is for

The thesis reports that the multi-agent decomposition is **not** what makes a small local
model faithful: a single agent handed the same evidence answers just as faithfully. What
does make it dependable is the *deterministic preparation of the evidence* — resolving the
company without asking the model, precomputing macroeconomic summaries, enforcing which
agents a question requires. The decomposition earns its place elsewhere: comparable
faithfulness from less than half the material, at more than three times the retrieval
precision.

This repository holds the system those measurements were taken on, the harness that took
them, and every raw result, so the numbers in the thesis can be checked rather than taken
on trust.

## Architecture

```
question
   │
   ▼
router ──────────────► rag agent      SEC 10-K filings, ChromaDB + nomic-embed-text
   │  (+ macro-route   market agent   Yahoo Finance
   │   guardrail)      macro agent    FRED
   │
   ▼
agent selection ─────► synthesiser ──► answer
   (+ required-agent floor)
```

Orchestration is an explicit LangGraph state machine, so every routing and selection
decision is inspectable rather than emergent. Five **deterministic guardrails** sit
alongside the model: ticker resolution against an alias catalogue, a macro-route
guardrail, a required-agent floor, precomputed macroeconomic summaries and NaN-safe
formatting. Section 4.8 of the thesis describes each.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) with `gemma3:4b` and `nomic-embed-text`
- A [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) (free)
- An email address for the SEC EDGAR user agent (required by SEC policy)
- An Anthropic API key — **only** to re-run the RAGAS scoring; the system itself does not
  need it

About 8 GB of RAM is enough to run the system; the full hundred-question evaluation takes
a few hours on consumer hardware, and the scoring stage costs a few euros in API calls.

## Setup

```bash
git clone https://github.com/FilippoMariaIncecchi/financial-multiagent.git
cd financial-multiagent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull gemma3:4b
ollama pull nomic-embed-text

cp .env.example .env      # then fill in the three values
```

`.env` holds three variables and is never committed:

```
SEC_EMAIL=your.name@example.com
FRED_API_KEY=...
ANTHROPIC_API_KEY=...
```

## Running

```bash
python main.py            # command line; ingests the filings on first run
streamlit run app.py      # web interface
```

The first run downloads the 10-K filings of the companies listed in `config.py`
(`SEC_COMPANIES`) from SEC EDGAR and indexes them. The filings are not included here:
they are public, and re-downloading them is part of the pipeline. The benchmark uses nine
companies — the seven in `SEC_COMPANIES` plus `INTC` and `PLTR`, which are fetched on
demand the first time a question mentions them.

## Reproducing the evaluation

```bash
python evaluation/phase3_ragas_eval.py                  # the full system, 100 questions
python evaluation/phase3_ragas_eval.py --baseline       # the same model with no tools
python evaluation/single_agent_baseline.py              # one agent, the same evidence
python evaluation/guardrail_ablation.py                 # the two decision guardrails
python evaluation/guardrail_ablation_full.py            # the three evidence guardrails
python evaluation/score_any.py --file <traces.jsonl> --label "<name>"

python evaluation/human_validation.py --extend 30       # sample answers to rate by hand
python evaluation/human_validation.py --compare         # agreement with the judge

python evaluation/corpus_cleanup.py --audit             # corpus composition report
```

Generation runs locally at temperature zero, so it is reproducible up to the data the
providers return on the day. Traces are written as they are produced, so an interrupted
run can be resumed. Scoring uses Claude Haiku through the Anthropic API — a different
model family from the generator, which is what makes the judgement independent.

Every result the thesis reports is already in `evaluation/results/`, with
[an index](evaluation/results/README.md) mapping each file to the experiment it belongs to
and the section that discusses it.

## Layout

| path | contents |
|---|---|
| `agents/` | orchestrator (router, selection, synthesiser) and the three specialised agents |
| `rag/` | ingestion, chunking, vector store and retrieval |
| `evaluation/` | the benchmark, the harness and every experiment script |
| `evaluation/results/` | raw traces, per-question scores, summaries and reports |
| `figures/` | the figures used in the thesis |
| `docs/` | architecture notes and screenshots of the interface |
| `tests/` | routing tests |
| `config.py` | model, chunking, retrieval and alias-catalogue settings |

## Notes on the results

Two things in `evaluation/results/` are worth flagging because they are findings rather
than artefacts.

`corpus_contamination.md` documents what happened when the deterministic ticker guardrail
was disabled: the model returned ordinary English words as ticker symbols, and the
on-demand ingestion downloaded and indexed the 10-K of whichever real company owns each
symbol. The corpus stayed polluted after the run. It is the strongest single piece of
evidence in the thesis for doing that resolution deterministically.

`human_validation.csv` holds thirty answers rated by hand against the evidence the system
used, without sight of the automatic scores. Where the two disagree the judge is the
stricter of them on nine occasions out of ten, which is why the reported faithfulness is a
lower bound. The one exception is an answer that asserts nothing at all and receives the
highest possible score.

## Licence and citation

Released under the MIT licence. If you use this work, please cite it with the DOI in
`CITATION.cff`.
