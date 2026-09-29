# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Re-run the multi-agent categories after revising the synthesiser prompt.

WHY
    The evaluation showed the system scoring 0.37 answer relevancy on
    multi_rag_market, against 0.76 for a single agent given the same evidence,
    and the human validation surfaced the mechanism: the synthesiser reports the
    retrieved figures without connecting them to the question. Two sampled
    answers illustrate it — "Nvidia's current P/E ratio is 29.9x" to a question
    asking whether that ratio is justified, and a list of macro indicators to a
    question asking how a slowing economy would affect Intel.

    The synthesiser prompt was revised to require that the answer resolve the
    question type (a yes/no judgement, an explicit A→B connection, or a value)
    and that every entity named in the question appear in the answer.

WHAT THIS DOES
    Re-runs ONLY the four multi categories — the ones the change targets — and
    writes them to a separate file, so the original traces remain untouched and
    the comparison is like-for-like.

USAGE
    python evaluation/rerun_multi.py
    python evaluation/score_any.py \\
        --file evaluation/results/multi_revised_synth.jsonl \\
        --label "revised synthesiser"

    Then compare the multi rows against evaluation/results/phase3_summary.csv.
    The script prints the baseline rows at the end for convenience.

NOTE
    Only the multi categories are affected: the single-agent routes do not pass
    through the synthesiser in a way this change alters. Re-running 40 questions
    instead of 100 keeps the scoring cost to roughly one euro.
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
OUT = RESULTS_DIR / "multi_revised_synth.jsonl"
MULTI = {"multi_rag_market", "multi_rag_macro", "multi_market_macro", "multi_all"}


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
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and OUT.exists():
        OUT.unlink()

    questions = [q for q in QUESTIONS if q["category"] in MULTI]
    if args.limit:
        questions = questions[: args.limit]
    done = {r["id"] for r in load(OUT) if not r.get("error")}
    remaining = [q for q in questions if q["id"] not in done]

    print("=" * 70)
    print("  RE-RUN — multi categories with the revised synthesiser prompt")
    print(f"  {len(done)} done · {len(remaining)} to run · {len(questions)} total")
    print("=" * 70)

    if remaining:
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        with open(OUT, "a", encoding="utf-8") as fh:
            for i, q in enumerate(remaining, 1):
                print(f"\n[{i}/{len(remaining)}] ({q['category']}) {q['question']}")
                try:
                    tr = orch.run_traced(q["question"], use_memory=False)
                    rec = {
                        "id": q["id"], "category": q["category"],
                        "question": q["question"],
                        "route": tr["route"], "agents": tr["agents"],
                        "answer": tr["answer"], "contexts": tr["contexts"],
                        "rag_contexts": tr["rag_contexts"],
                        "market_summary": tr["market_summary"],
                        "macro_summary": tr["macro_summary"],
                        "tickers": tr.get("tickers", []),
                    }
                    print(f"    → {(tr['answer'] or '')[:130]}...")
                except Exception as e:                        # noqa: BLE001
                    rec = {"id": q["id"], "category": q["category"],
                           "question": q["question"],
                           "error": f"{type(e).__name__}: {e}"}
                    print(f"    ERROR: {rec['error']}")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    ok = [r for r in load(OUT) if not r.get("error")]
    print("\n" + "=" * 70)
    print(f"  {len(ok)}/{len(questions)} collected → {OUT}")
    print("\n  Score it:")
    print("    python evaluation/score_any.py \\")
    print(f"        --file {OUT.relative_to(ROOT)} --label \"revised synthesiser\"")
    print("\n  Baseline to compare against (original synthesiser):")
    summ = RESULTS_DIR / "phase3_summary.csv"
    if summ.exists():
        for line in summ.read_text(encoding="utf-8").splitlines():
            cat = line.split(",")[0]
            if cat in MULTI or cat == "category":
                print("   ", line)
    print("=" * 70)


if __name__ == "__main__":
    main()
