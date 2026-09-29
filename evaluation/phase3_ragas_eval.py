# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Phase 3 — RAGAS evaluation harness (two stages).

  Stage 1  (--collect) : runs the multi-agent system over the 100 questions and
                         saves a trace per question (route, agents, final answer,
                         retrieved contexts). Needs Ollama + the populated vector
                         store, but NO API key. This is the slow, local pass.

  Stage 2  (--score)   : loads the traces and scores them with RAGAS using
                         Claude Haiku 4.5 as the judge LLM (Anthropic API) and
                         the local nomic-embed-text model for embeddings. Needs an
                         ANTHROPIC_API_KEY. Writes per-question and per-category
                         CSVs + a summary.

Usage:
    python evaluation/phase3_ragas_eval.py               # both stages
    python evaluation/phase3_ragas_eval.py --collect     # stage 1 only
    python evaluation/phase3_ragas_eval.py --score       # stage 2 only (reuse traces)
    python evaluation/phase3_ragas_eval.py --collect --limit 7   # quick smoke test
    python evaluation/phase3_ragas_eval.py --collect --fresh     # restart, ignore saved traces
    python evaluation/phase3_ragas_eval.py --collect --only macro,multi_market_macro  # re-run 2 categories
    python evaluation/phase3_ragas_eval.py --baseline            # bare LLM, no system (ablation)
    python evaluation/phase3_ragas_eval.py --multiturn          # conversational-memory ablation

Resume: --collect skips questions already collected and APPENDS new ones, so
you can stop any time (Ctrl+C) and re-run --collect to continue where you left
off. Progress is flushed after every question, so nothing is lost.

Metrics (reference-free):
    • faithfulness          — is the answer grounded in the retrieved context?
    • answer relevancy      — does the answer address the question?
    • context precision     — are the retrieved contexts relevant? (LLM, no reference)

Setup (one-time, on the machine that runs it):
    pip install ragas langchain-anthropic
    ollama pull nomic-embed-text
    echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from config import EMBEDDING_MODEL                 # noqa: E402
from evaluation.questions import EXPECTED_COUNTS, QUESTIONS  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────

JUDGE_MODEL  = "claude-haiku-4-5-20251001"   # RAGAS judge (Anthropic API).
                                   # Cheap: ~$1/M input, $5/M output tokens.
                                   # Set ANTHROPIC_API_KEY in .env before --score.
RESULTS_DIR  = ROOT / "evaluation" / "results"
TRACES_PATH  = RESULTS_DIR / "phase3_traces.jsonl"
SCORES_PATH  = RESULTS_DIR / "phase3_scores.csv"
SUMMARY_PATH = RESULTS_DIR / "phase3_summary.csv"
BASELINE_PATH = RESULTS_DIR / "phase3_baseline.jsonl"
MULTITURN_PATH = RESULTS_DIR / "phase3_multiturn.jsonl"


# ── STAGE 1: COLLECT TRACES (local, no API) ───────────────────────────────────

def _load_traces(path: Path = TRACES_PATH) -> list[dict]:
    """
    Load saved traces, de-duplicated by id. Appended resume runs may write the
    same id more than once (e.g. an earlier error then a later success); a
    successful record always wins, otherwise the latest one is kept.
    """
    if not path.exists():
        return []
    by_id: dict[str, dict] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cur = by_id.get(r["id"])
        if cur is None or not r.get("error") or cur.get("error"):
            by_id[r["id"]] = r
    return list(by_id.values())


def collect(limit: int | None = None, fresh: bool = False,
            only: set[str] | None = None) -> None:
    from agents.orchestrator import Orchestrator

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = QUESTIONS
    if only:
        questions = [q for q in questions if q["category"] in only]
    if limit:
        questions = questions[:limit]

    # ── Resume / re-run support ───────────────────────────────────────────────
    # Normally already-collected (non-error) ids are skipped and new results are
    # APPENDED, so an interrupted run (Ctrl+C) never loses progress. With --only,
    # the selected categories are ALWAYS re-run and their new records overwrite
    # the old ones (the loader keeps the latest); other categories stay untouched.
    # --fresh wipes everything and starts over.
    done_ids = set()
    if only:
        pass                                   # force re-run of the selected categories
    elif fresh:
        if TRACES_PATH.exists():
            TRACES_PATH.unlink()
    else:
        done_ids = {r["id"] for r in _load_traces() if not r.get("error")}

    remaining = [q for q in questions if q["id"] not in done_ids]

    print("=" * 70)
    print("  STAGE 1 — COLLECTING TRACES" + (f"  (only: {', '.join(sorted(only))})" if only else ""))
    print(f"  {len(done_ids)} already done · {len(remaining)} to run · {len(questions)} selected")
    print("=" * 70)

    if not remaining:
        print("\n[Stage 1] Nothing to do — all requested questions already collected.")
        _print_routing_report(_load_traces())
        return

    orch = Orchestrator()

    with open(TRACES_PATH, "a", encoding="utf-8") as fh:   # append → safe resume
        for i, q in enumerate(remaining, 1):
            print(f"\n[{i}/{len(remaining)}] ({q['category']}) {q['question']}")
            try:
                trace = orch.run_traced(q["question"])
                route_ok  = trace["route"] == q["expected_route"]
                agents_ok = set(trace["agents"]) == set(q["expected_agents"])
                record = {
                    "id":              q["id"],
                    "category":        q["category"],
                    "question":        q["question"],
                    "expected_route":  q["expected_route"],
                    "route":           trace["route"],
                    "route_ok":        route_ok,
                    "expected_agents": q["expected_agents"],
                    "agents":          trace["agents"],
                    "agents_ok":       agents_ok,
                    "answer":          trace["answer"],
                    "contexts":        trace["contexts"],
                    "error":           "",
                }
                print(f"    route={trace['route']} ({'✅' if route_ok else '❌'})  "
                      f"agents={trace['agents']} ({'✅' if agents_ok else '❌'})")
            except Exception as e:  # never lose the whole run over one question
                record = {
                    "id": q["id"], "category": q["category"], "question": q["question"],
                    "expected_route": q["expected_route"], "route": "", "route_ok": False,
                    "expected_agents": q["expected_agents"], "agents": [], "agents_ok": False,
                    "answer": "", "contexts": [], "error": f"{type(e).__name__}: {e}",
                }
                print(f"    ⚠️  ERROR: {record['error']}")

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()

    _print_routing_report(_load_traces())
    print(f"\n[Stage 1] Traces saved → {TRACES_PATH}")


# ── STAGE 2: SCORE WITH RAGAS (needs ANTHROPIC_API_KEY) ───────────────────────

def score(only: set[str] | None = None) -> None:
    import os

    if not TRACES_PATH.exists():
        print(f"⚠️  No traces found at {TRACES_PATH}. Run --collect first.")
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set. Add it to .env to run the Claude judge.")
        return

    records = [r for r in _load_traces() if not r["error"]]  # deduped, skip failures
    if only:
        records = [r for r in records if r["category"] in only]
    if not records:
        print("⚠️  No usable traces to score.")
        return

    print("=" * 70)
    print(f"  STAGE 2 — RAGAS SCORING ({len(records)} traces, judge={JUDGE_MODEL})")
    print("=" * 70)

    try:
        import pandas as pd
        from langchain_anthropic import ChatAnthropic
        from langchain_ollama import OllamaEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}.  Run: pip install ragas langchain-anthropic")
        return

    # Build the RAGAS metrics (class API in 0.2; context precision needs no reference).
    metrics = _build_metrics()

    # Judge = Claude Opus 4.8; embeddings = local nomic-embed-text.
    judge = LangchainLLMWrapper(ChatAnthropic(model=JUDGE_MODEL, temperature=0.0, max_tokens=2048))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBEDDING_MODEL))

    samples = [{
        "user_input":         r["question"],
        "response":           r["answer"] or "(no answer produced)",
        "retrieved_contexts": r["contexts"] or ["(no context retrieved)"],
    } for r in records]

    dataset = EvaluationDataset.from_list(samples)

    try:
        result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings)
    except Exception as e:
        print(f"⚠️  RAGAS evaluation failed: {type(e).__name__}: {e}")
        return

    df = result.to_pandas()
    # Attach our metadata by row order (RAGAS preserves sample order).
    df.insert(0, "id",       [r["id"] for r in records])
    df.insert(1, "category", [r["category"] for r in records])

    metric_cols = [c for c in df.columns
                   if c not in {"id", "category", "user_input", "response",
                                "retrieved_contexts", "reference"}
                   and pd.api.types.is_numeric_dtype(df[c])]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_PATH, index=False)

    # Per-category summary (NaN-safe means).
    summary = df.groupby("category")[metric_cols].mean(numeric_only=True)
    order = [c for c in EXPECTED_COUNTS if c in summary.index]
    summary = summary.reindex(order)
    summary.loc["OVERALL"] = df[metric_cols].mean(numeric_only=True)
    summary.to_csv(SUMMARY_PATH)

    print("\n── RAGAS scores by category (mean) ─────────────────────────────────")
    print(summary.round(3).to_string())
    n_nan = int(df[metric_cols].isna().sum().sum())
    if n_nan:
        print(f"\n  note: {n_nan} metric cell(s) came back NaN (judge parse/sampling) "
              f"— means are computed over the valid cells.")
    _print_routing_report(records)
    print(f"\n[Stage 2] Per-question scores → {SCORES_PATH}")
    print(f"[Stage 2] Category summary    → {SUMMARY_PATH}")


def _build_metrics():
    """RAGAS metric instances, tolerant to minor version naming differences."""
    from ragas.metrics import Faithfulness
    metrics = [Faithfulness()]
    try:                                   # answer/response relevancy (embeddings)
        from ragas.metrics import ResponseRelevancy
        metrics.append(ResponseRelevancy())
    except ImportError:
        from ragas.metrics import AnswerRelevancy
        metrics.append(AnswerRelevancy())
    try:                                   # reference-free context precision (RAG)
        from ragas.metrics import LLMContextPrecisionWithoutReference
        metrics.append(LLMContextPrecisionWithoutReference())
    except ImportError:
        pass
    return metrics


# ── SHARED: ROUTING / SELECTION ACCURACY ──────────────────────────────────────

def _print_routing_report(records: list[dict]) -> None:
    done = [r for r in records if not r["error"]]
    if not done:
        return
    route_hits  = sum(r["route_ok"] for r in done)
    agent_hits  = sum(r["agents_ok"] for r in done)
    print("\n── Routing / selection accuracy (cross-check) ──────────────────────")
    print(f"  Routing:           {route_hits}/{len(done)}")
    print(f"  Agent selection:   {agent_hits}/{len(done)}")

    cats = {}
    for r in done:
        c = cats.setdefault(r["category"], [0, 0])
        c[0] += r["route_ok"]; c[1] += 1
    for cat in EXPECTED_COUNTS:
        if cat in cats:
            hit, tot = cats[cat]
            print(f"    {'✅' if hit == tot else '⚠️ '} {cat:<20} route {hit}/{tot}")
    errs = [r for r in records if r["error"]]
    if errs:
        print(f"  ⚠️  {len(errs)} question(s) errored during collection.")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

BASELINE_PROMPT = """You are a financial analyst assistant. Answer the user's
question as accurately and specifically as you can, using your own knowledge.
If you are not sure of a figure, say so rather than guessing."""


def baseline(limit: int | None = None, fresh: bool = False) -> None:
    """
    Ablation: answer the same 100 questions with the BARE model — no router, no
    agents, no RAG, no live market/macro data. The model can only draw on its
    (stale) training memory, so this isolates what the multi-agent architecture
    adds. Saved to a separate file so it never mixes with the system traces.
    Resume/append + --fresh work exactly like --collect.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
    from config import LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = QUESTIONS[:limit] if limit else QUESTIONS

    done_ids = set()
    if not fresh:
        done_ids = {r["id"] for r in _load_traces(BASELINE_PATH) if not r.get("error")}
    elif BASELINE_PATH.exists():
        BASELINE_PATH.unlink()
    remaining = [q for q in questions if q["id"] not in done_ids]

    print("=" * 70)
    print("  BASELINE — bare model, NO system (no router / agents / RAG / live data)")
    print(f"  {len(done_ids)} already done · {len(remaining)} to run · {len(questions)} total")
    print("=" * 70)
    if not remaining:
        print("\n[Baseline] Nothing to do — all questions already answered.")
        return

    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, num_ctx=LLM_NUM_CTX)

    with open(BASELINE_PATH, "a", encoding="utf-8") as fh:
        for i, q in enumerate(remaining, 1):
            print(f"\n[{i}/{len(remaining)}] ({q['category']}) {q['question']}")
            try:
                resp = llm.invoke([SystemMessage(content=BASELINE_PROMPT),
                                   HumanMessage(content=q["question"])])
                rec = {"id": q["id"], "category": q["category"], "question": q["question"],
                       "route": "baseline", "agents": [], "contexts": [],
                       "answer": resp.content, "error": ""}
            except Exception as e:
                rec = {"id": q["id"], "category": q["category"], "question": q["question"],
                       "route": "baseline", "agents": [], "contexts": [],
                       "answer": "", "error": f"{type(e).__name__}: {e}"}
                print(f"    ⚠️  ERROR: {rec['error']}")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    print(f"\n[Baseline] Bare-model answers saved → {BASELINE_PATH}")


def multiturn() -> None:
    """
    Conversational-memory ablation. Runs the labelled multi-turn dialogues twice
    — once with memory ON, once with memory OFF — and measures reference-
    resolution accuracy on the follow-up turns (does the system recover the
    remembered ticker when the question only refers to it as "its" / "it"?).
    """
    from agents.orchestrator import Orchestrator
    from evaluation.dialogues import DIALOGUES, counts

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    c = counts()
    print("=" * 70)
    print("  CONVERSATIONAL-MEMORY ABLATION (multi-turn dialogues)")
    print(f"  {c['dialogues']} dialogues · {c['turns']} turns · "
          f"{c['followup_turns']} follow-up turns")
    print("=" * 70)

    orch = Orchestrator()
    records: list[dict] = []

    for memory_on in (True, False):
        print(f"\n===== {'MEMORY ON' if memory_on else 'MEMORY OFF (ablation)'} =====")
        for dlg in DIALOGUES:
            orch.memory.clear()                      # fresh, independent conversation
            for turn in dlg["turns"]:
                try:
                    trace = orch.run_traced(turn["text"], use_memory=memory_on)
                    resolved = trace["tickers"]
                    exp = turn["expected_tickers"]
                    res_ok = set(exp).issubset(set(resolved))
                    rec = {
                        "dialogue": dlg["id"], "turn": turn["id"],
                        "memory_on": memory_on, "is_followup": turn["is_followup"],
                        "text": turn["text"], "expected_tickers": exp,
                        "resolved_tickers": resolved, "route": trace["route"],
                        "resolution_ok": res_ok, "answer": trace["answer"], "error": "",
                    }
                    if turn["is_followup"]:
                        print(f"  {'✅' if res_ok else '❌'} {turn['id']:7} "
                              f"resolved={resolved} exp={exp} ({trace['route']})")
                except Exception as e:
                    rec = {
                        "dialogue": dlg["id"], "turn": turn["id"],
                        "memory_on": memory_on, "is_followup": turn["is_followup"],
                        "text": turn["text"], "expected_tickers": turn["expected_tickers"],
                        "resolved_tickers": [], "route": "", "resolution_ok": False,
                        "answer": "", "error": f"{type(e).__name__}: {e}",
                    }
                    print(f"    ⚠️  ERROR {turn['id']}: {rec['error']}")
                records.append(rec)

    with open(MULTITURN_PATH, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def acc(mode: bool):
        fu = [r for r in records if r["memory_on"] == mode and r["is_followup"] and not r["error"]]
        return sum(r["resolution_ok"] for r in fu), len(fu)
    on_h, on_n = acc(True)
    off_h, off_n = acc(False)
    print("\n" + "=" * 60)
    print("  REFERENCE-RESOLUTION ACCURACY (follow-up turns)")
    print(f"    Memory ON : {on_h}/{on_n}")
    print(f"    Memory OFF: {off_h}/{off_n}   (ablation)")
    print("=" * 60)
    print(f"\n[Multi-turn] Saved -> {MULTITURN_PATH}")


def main() -> None:
    argv = sys.argv[1:]
    limit = None
    if "--limit" in argv:
        try:
            limit = int(argv[argv.index("--limit") + 1])
        except (IndexError, ValueError):
            print("⚠️  --limit needs an integer, e.g. --limit 7")
            return

    only = None
    if "--only" in argv:
        try:
            raw = argv[argv.index("--only") + 1]
        except IndexError:
            print("⚠️  --only needs a category list, e.g. --only macro,multi_market_macro")
            return
        only = {c.strip() for c in raw.split(",") if c.strip()}
        unknown = only - set(EXPECTED_COUNTS)
        if unknown:
            print(f"⚠️  Unknown categor(ies): {', '.join(sorted(unknown))}")
            print(f"    Valid: {', '.join(EXPECTED_COUNTS)}")
            return

    if "--multiturn" in argv:
        multiturn()
        return

    if "--baseline" in argv:
        baseline(limit, fresh="--fresh" in argv)
        return

    do_collect = "--collect" in argv
    do_score   = "--score" in argv
    if not do_collect and not do_score:   # default: run both stages
        do_collect = do_score = True

    if do_collect:
        collect(limit, fresh="--fresh" in argv, only=only)
    if do_score:
        score(only=only)


if __name__ == "__main__":
    main()
