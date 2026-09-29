# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Single-agent baseline WITH the same tools and data (fair comparison for RQ1).

WHY THIS EXISTS
    The bare-model baseline (`--baseline` in phase3_ragas_eval.py) removes both
    the architecture AND the data: it shows that a model without tools cannot
    report current figures, which is true but does not isolate the contribution
    of the multi-agent decomposition.

    This script closes that gap. The model receives EXACTLY the same evidence the
    multi-agent system would gather — retrieved 10-K passages, live market data
    and macroeconomic series — but with no router, no agent selection, no
    per-source specialised prompts and no synthesiser. One model, one prompt,
    one call, all the evidence at once.

    The difference between this baseline and the full system is therefore
    attributable to the DECOMPOSITION, not to data access.

DESIGN NOTE
    Because there is no router, the single agent cannot decide which sources a
    question needs: it is given all of them. That is the honest single-agent
    counterpart — deciding what to fetch is precisely the job the architecture
    performs — and it is also the more favourable setting for the baseline,
    since no required evidence is ever missing.

USAGE
    python evaluation/single_agent_baseline.py              # all 100 questions
    python evaluation/single_agent_baseline.py --limit 5    # smoke test
    python evaluation/single_agent_baseline.py --fresh      # restart

    Then score it with the same judge and metrics as the full system:
    python evaluation/score_any.py \
        --file evaluation/results/single_agent_baseline.jsonl \
        --label "single agent + tools"

    The trace format is identical to phase3_traces.jsonl, so the comparison
    against evaluation/results/phase3_summary.csv is like-for-like.

REQUIREMENTS
    Ollama running with gemma3:4b, vector store populated, FRED key in .env.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from evaluation.questions import QUESTIONS  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"
OUT = RESULTS_DIR / "single_agent_baseline.jsonl"

SINGLE_AGENT_PROMPT = """You are a financial analyst assistant.

You are given evidence gathered from three sources: excerpts from SEC 10-K
filings, current market data, and current macroeconomic data. Some of it may be
irrelevant to the question — it is up to you to decide what to use.

Rules:
- Answer ONLY from the evidence provided below. Do not add figures from memory.
- If the evidence does not contain what is needed, say so explicitly.
- Be concise and answer the specific question asked.
- Do not add investment recommendations or generic disclaimers.
"""


def build_evidence(orch, question: str) -> tuple[str, dict]:
    """
    Gather ALL three evidence blocks for this question, exactly as the
    specialised agents would, but without any routing decision.
    """
    from rag.retriever import Retriever  # noqa: F401  (typing only)

    parts: list[str] = []
    detail = {"rag_contexts": [], "market_summary": "", "macro_summary": ""}

    # --- filings (same ticker-filtered retrieval the RAG agent uses) ---
    try:
        tickers = orch._ensure_tickers_indexed(question, "")
        chunks = orch.retriever.retrieve(question, tickers=tickers)
        ctxs = [c.text for c in chunks]
        detail["rag_contexts"] = ctxs
        if ctxs:
            parts.append("=== SEC 10-K FILING EXCERPTS ===\n"
                         + orch.retriever.format_context(chunks))
    except Exception as e:                                   # noqa: BLE001
        print(f"    [rag] {type(e).__name__}: {e}")

    # --- market data ---
    try:
        m = orch.market_agent.run(question, history="")
        if m and m.data_summary:
            detail["market_summary"] = m.data_summary
            parts.append("=== CURRENT MARKET DATA ===\n" + m.data_summary)
    except Exception as e:                                   # noqa: BLE001
        print(f"    [market] {type(e).__name__}: {e}")

    # --- macroeconomic data ---
    try:
        mac = orch.macro_agent.run(question, history="")
        if mac and mac.data_summary:
            detail["macro_summary"] = mac.data_summary
            parts.append("=== CURRENT MACROECONOMIC DATA ===\n" + mac.data_summary)
    except Exception as e:                                   # noqa: BLE001
        print(f"    [macro] {type(e).__name__}: {e}")

    return "\n\n".join(parts), detail


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    seen = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            seen[r["id"]] = r
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and OUT.exists():
        OUT.unlink()

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    done = {r["id"] for r in load(OUT) if not r.get("error")}
    remaining = [q for q in questions if q["id"] not in done]

    print("=" * 70)
    print("  SINGLE-AGENT BASELINE — same tools and data, no decomposition")
    print(f"  {len(done)} already done · {len(remaining)} to run · {len(questions)} total")
    print("=" * 70)
    if not remaining:
        print("\nNothing to do.")
        return

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama
    from agents.orchestrator import Orchestrator
    from config import LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE

    orch = Orchestrator()
    llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, num_ctx=LLM_NUM_CTX)

    with open(OUT, "a", encoding="utf-8") as fh:
        for i, q in enumerate(remaining, 1):
            print(f"\n[{i}/{len(remaining)}] ({q['category']}) {q['question']}")
            try:
                evidence, detail = build_evidence(orch, q["question"])
                messages = [
                    SystemMessage(content=SINGLE_AGENT_PROMPT),
                    HumanMessage(content=f"EVIDENCE:\n{evidence}\n\nQUESTION: {q['question']}"),
                ]
                answer = llm.invoke(messages).content.strip()

                contexts = list(detail["rag_contexts"])
                if detail["market_summary"]:
                    contexts.append(detail["market_summary"])
                if detail["macro_summary"]:
                    contexts.append(detail["macro_summary"])

                rec = {
                    "id": q["id"],
                    "category": q["category"],
                    "question": q["question"],
                    # same trace shape as phase3_traces.jsonl → same scorer works
                    "route": "single_agent",
                    "agents": ["single"],
                    "answer": answer,
                    "contexts": contexts,
                    "rag_contexts": detail["rag_contexts"],
                    "market_summary": detail["market_summary"],
                    "macro_summary": detail["macro_summary"],
                }
                print(f"    → {answer[:110]}...")
            except Exception as e:                           # noqa: BLE001
                rec = {"id": q["id"], "category": q["category"],
                       "question": q["question"],
                       "error": f"{type(e).__name__}: {e}"}
                print(f"    ERROR: {rec['error']}")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    ok = [r for r in load(OUT) if not r.get("error")]
    print("\n" + "=" * 70)
    print(f"  Collected {len(ok)}/{len(questions)} → {OUT}")
    print("  Next: score this file with the RAGAS harness and compare against")
    print("        evaluation/results/phase3_summary.csv (the full system).")
    print("=" * 70)


if __name__ == "__main__":
    main()
