# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Score ANY trace file with the same RAGAS setup used for the main evaluation.

The main harness (phase3_ragas_eval.py --score) is hard-wired to the system's
own trace file. This script applies the identical metrics, judge and embeddings
to any file in the same format, so that a baseline can be compared with the
full system on exactly equal terms.

USAGE
    python evaluation/score_any.py --file evaluation/results/single_agent_baseline.jsonl
    python evaluation/score_any.py --file <path> --label "single agent + tools"

OUTPUT
    <file>_scores.csv     per-question metric values
    <file>_summary.csv    per-category and overall means

REQUIREMENTS
    ANTHROPIC_API_KEY in .env, Ollama running (for the embedding model).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from config import EMBEDDING_MODEL                              # noqa: E402
from evaluation.phase3_ragas_eval import JUDGE_MODEL, _build_metrics  # noqa: E402


def load(path: Path) -> list[dict]:
    seen = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                seen[r["id"]] = r
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    import os
    path = Path(args.file)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        sys.exit(f"File not found: {path}")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set — add it to .env")

    records = load(path)
    if not records:
        sys.exit("No usable records.")

    label = args.label or path.stem
    print("=" * 70)
    print(f"  SCORING '{label}' — {len(records)} records, judge={JUDGE_MODEL}")
    print("=" * 70)

    import pandas as pd
    from langchain_anthropic import ChatAnthropic
    from langchain_ollama import OllamaEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    judge = LangchainLLMWrapper(
        ChatAnthropic(model=JUDGE_MODEL, temperature=0.0, max_tokens=2048))
    embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=EMBEDDING_MODEL))

    dataset = EvaluationDataset.from_list([{
        "user_input":         r["question"],
        "response":           r.get("answer") or "(no answer produced)",
        "retrieved_contexts": r.get("contexts") or ["(no context retrieved)"],
    } for r in records])

    result = evaluate(dataset=dataset, metrics=_build_metrics(),
                      llm=judge, embeddings=embeddings)

    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in records])
    df.insert(1, "category", [r["category"] for r in records])

    scores_path = path.with_name(path.stem + "_scores.csv")
    df.to_csv(scores_path, index=False)

    metric_cols = [c for c in df.columns
                   if c not in ("id", "category", "user_input", "response",
                                "retrieved_contexts")]
    summary = df.groupby("category")[metric_cols].mean(numeric_only=True)
    overall = df[metric_cols].mean(numeric_only=True).to_frame().T
    overall.index = ["OVERALL"]
    summary = pd.concat([summary, overall])

    summary_path = path.with_name(path.stem + "_summary.csv")
    summary.to_csv(summary_path)

    print("\n" + summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n[per-question] {scores_path}")
    print(f"[summary]      {summary_path}")


if __name__ == "__main__":
    main()
