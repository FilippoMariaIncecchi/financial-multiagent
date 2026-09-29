# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Ablation of the remaining deterministic guardrails (completes RQ2).

guardrail_ablation.py measured the two DECISION guardrails (macro-route
downgrade, required-agent floor) by recording the decision before and after
they act. The other three guardrails cannot be measured that way, because they
do not change a decision — they change the EVIDENCE the model is given. They
therefore require re-running the affected questions with the guardrail disabled
and scoring the answers.

    G3  macro SUMMARY       the macro agent precomputes latest value, period and
                            year-over-year change and trend, so the model only
                            quotes finished figures. Disabled: the model receives
                            the raw series and must do the arithmetic itself.
                            Affects: macro + every multi category containing macro.

    G4  ticker resolution   company identity is resolved by a deterministic scan
                            of the question, with the LLM used only as a fallback.
                            Disabled: the scan always returns nothing, so every
                            question goes through the LLM extractor.
                            Affects: rag + every multi category containing rag.

    G5  NaN-safe formatting missing or non-finite provider values are rendered as
                            "N/A". Disabled: whatever the provider returned is
                            printed, so NaN and spurious values reach the model.
                            Affects: market + every multi category containing market.
                            Scored by counting, not by the judge — no API cost.

USAGE
    python evaluation/guardrail_ablation_full.py --which macro_summary
    python evaluation/guardrail_ablation_full.py --which ticker
    python evaluation/guardrail_ablation_full.py --which nan_safe
    python evaluation/guardrail_ablation_full.py --which all      # runs all three

    Only the affected categories are re-run, which keeps both the runtime and the
    scoring cost down. Each run is resumable.

THEN (for the two that need the judge)
    python evaluation/score_any.py \
        --file evaluation/results/ablation_no_macro_summary.jsonl \
        --label "no macro SUMMARY"
    python evaluation/score_any.py \
        --file evaluation/results/ablation_no_ticker_scan.jsonl \
        --label "no deterministic ticker scan"

    Compare against evaluation/results/phase3_summary.csv restricted to the same
    categories — the script prints the relevant subset for you at the end.

REQUIREMENTS
    Ollama running with gemma3:4b, vector store populated, FRED key in .env.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from evaluation.questions import QUESTIONS  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"

ABLATIONS = {
    "macro_summary": {
        "out": "ablation_no_macro_summary.jsonl",
        "label": "no macro SUMMARY",
        "categories": {"macro", "multi_rag_macro", "multi_market_macro", "multi_all"},
        "needs_judge": True,
    },
    "ticker": {
        "out": "ablation_no_ticker_scan.jsonl",
        "label": "no deterministic ticker scan",
        "categories": {"rag", "multi_rag_market", "multi_rag_macro", "multi_all"},
        "needs_judge": True,
    },
    "nan_safe": {
        "out": "ablation_no_nan_safe.jsonl",
        "label": "no NaN-safe formatting",
        "categories": {"market", "multi_rag_market", "multi_market_macro", "multi_all"},
        "needs_judge": False,
    },
}


# ── the three disabling patches ───────────────────────────────────────────────

def disable_macro_summary(orch):
    """Strip the precomputed SUMMARY block; leave the raw rows."""
    agent = orch.macro_agent
    original = agent._format_series

    def raw_format(series, description, series_id=""):
        text = original(series, description, series_id)
        out, skipping = [], False
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("SUMMARY"):
                skipping = True
                continue
            if skipping:
                # the SUMMARY block is the indented lines that follow it
                if s.startswith(("Latest:", "Change vs", "Trend over")):
                    continue
                skipping = False
            out.append(line)
        return "\n".join(out)

    agent._format_series = raw_format


PROJECT_MODULES = ("tickers", "config", "memory", "agents", "rag", "evaluation")


def _patch_symbol(name: str, replacement) -> int:
    """
    Rebind a module-level symbol wherever the project imported it by name.

    Two precautions matter here. Only this project's modules are inspected, and
    the lookup goes through `mod.__dict__` rather than `getattr`: third-party
    packages such as `transformers` define a module-level `__getattr__` for lazy
    loading, and calling `getattr` on them triggers imports of optional
    dependencies that need not be installed.
    """
    patched = 0
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        root = mod_name.split(".")[0]
        if root not in PROJECT_MODULES:
            continue
        d = getattr(mod, "__dict__", None)
        if d is not None and name in d:
            d[name] = replacement
            patched += 1
    return patched


def disable_ticker_scan(orch):
    """Force every question through the LLM ticker extractor."""
    def nothing(_question):
        return []

    n = _patch_symbol("scan_known_tickers", nothing)
    print(f"  [patch] scan_known_tickers disabled in {n} project module(s)")
    if n == 0:
        sys.exit("Patch failed: scan_known_tickers not found in any loaded module.")


def disable_nan_safe(orch):
    """Let non-finite provider values through to the model."""
    n = _patch_symbol("_finite", lambda _x: True)
    print(f"  [patch] _finite disabled in {n} project module(s)")
    if n == 0:
        sys.exit("Patch failed: _finite not found in any loaded module.")


PATCHES = {
    "macro_summary": disable_macro_summary,
    "ticker": disable_ticker_scan,
    "nan_safe": disable_nan_safe,
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    seen = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            seen[r["id"]] = r
    return list(seen.values())


SPURIOUS = re.compile(r"\bnan\b|\binf\b|N/A%|\$nan|nan%", re.I)


def run_one(which: str) -> None:
    cfg = ABLATIONS[which]
    out_path = RESULTS_DIR / cfg["out"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions = [q for q in QUESTIONS if q["category"] in cfg["categories"]]
    done = {r["id"] for r in load(out_path) if not r.get("error")}
    remaining = [q for q in questions if q["id"] not in done]

    print("=" * 70)
    print(f"  ABLATION — {cfg['label']}")
    print(f"  categories: {', '.join(sorted(cfg['categories']))}")
    print(f"  {len(done)} done · {len(remaining)} to run · {len(questions)} affected")
    print("=" * 70)

    if remaining:
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        PATCHES[which](orch)
        print(f"  [patch] {which} disabled\n")

        with open(out_path, "a", encoding="utf-8") as fh:
            for i, q in enumerate(remaining, 1):
                print(f"[{i}/{len(remaining)}] ({q['category']}) {q['question']}")
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
                    print(f"    → {(tr['answer'] or '')[:100]}...")
                except Exception as e:                        # noqa: BLE001
                    rec = {"id": q["id"], "category": q["category"],
                           "question": q["question"],
                           "error": f"{type(e).__name__}: {e}"}
                    print(f"    ERROR: {rec['error']}")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    records = [r for r in load(out_path) if not r.get("error")]
    print(f"\n[done] {len(records)} records → {out_path}")

    if which == "nan_safe":
        # measured by counting, no judge needed
        base = {r["id"]: r for r in load(RESULTS_DIR / "phase3_traces.jsonl")}
        bad_ab = [r for r in records if SPURIOUS.search(r.get("market_summary") or "")
                  or SPURIOUS.search(r.get("answer") or "")]
        bad_base = [r for r in base.values()
                    if r["id"] in {x["id"] for x in records}
                    and (SPURIOUS.search(r.get("market_summary") or "")
                         or SPURIOUS.search(r.get("answer") or ""))]
        md = [
            "# Ablation — NaN-safe formatting",
            "",
            "Non-finite provider values reach the model when the guardrail is off.",
            "",
            "| Configuration | Answers or data blocks containing a non-finite value |",
            "|---|---|",
            f"| With NaN-safe formatting (system) | {len(bad_base)}/{len(records)} |",
            f"| Without it (ablated) | {len(bad_ab)}/{len(records)} |",
            "",
            "## Affected questions",
            "",
        ]
        for r in bad_ab:
            snippet = (r.get("market_summary") or "")[:160].replace("\n", " ")
            md.append(f"- `{r['id']}` — {r['question'][:70]} → `{snippet}`")
        (RESULTS_DIR / "ablation_nan_safe.md").write_text("\n".join(md), encoding="utf-8")
        print(f"[report] {RESULTS_DIR / 'ablation_nan_safe.md'}")
        print(f"  with guardrail: {len(bad_base)} · without: {len(bad_ab)}")
    else:
        print("\nNext step — score it with the same judge:")
        print(f"  python evaluation/score_any.py --file {out_path.relative_to(ROOT)} "
              f"--label \"{cfg['label']}\"")
        print("\nThen compare against these baseline rows (full system):")
        summ = RESULTS_DIR / "phase3_summary.csv"
        if summ.exists():
            for line in summ.read_text(encoding="utf-8").splitlines():
                cat = line.split(",")[0]
                if cat in cfg["categories"] or cat in ("category", "OVERALL"):
                    print("   ", line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", required=True,
                    choices=list(ABLATIONS) + ["all"])
    args = ap.parse_args()

    targets = list(ABLATIONS) if args.which == "all" else [args.which]
    for i, w in enumerate(targets):
        if i:
            print("\n\n")
        run_one(w)
        if args.which == "all" and i < len(targets) - 1:
            print("\n⚠️  Restart the process before the next ablation: the patches "
                  "modify module-level state and must not overlap.")
            print("    Run them one at a time with --which <name>.")
            break


if __name__ == "__main__":
    main()
