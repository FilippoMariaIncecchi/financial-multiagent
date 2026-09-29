# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Guardrail ablation — raw LLM decision vs decision after the deterministic layer.

Answers the reviewer's request: report BOTH
    1) the raw LLM routing / agent-selection accuracy, and
    2) the final routing / agent-selection accuracy after the guardrails,

so the contribution of the deterministic correction layer can be quantified
instead of being asserted.

HOW IT WORKS
    The guardrails are two pure functions inside the Orchestrator:

        _guard_macro_route(question, route)   -> possibly downgraded route
        _ensure_required_agents(question, ag) -> possibly extended agent list

    This script wraps both. Each wrapper records the value it RECEIVED (the raw
    LLM decision) and the value it RETURNED (the final decision), then delegates
    to the original implementation. Nothing in the system's behaviour changes —
    the final answers are identical to a normal run.

    Ticker resolution is also instrumented: the orchestrator resolves tickers
    deterministically first and only falls back to the LLM when the scan finds
    nothing, so we record which of the two paths produced the result.

USAGE
    python evaluation/guardrail_ablation.py                # full 100 questions
    python evaluation/guardrail_ablation.py --limit 10     # quick smoke test
    python evaluation/guardrail_ablation.py --fresh        # ignore previous run

    Resumable: already-processed ids are skipped and new ones appended.

OUTPUT
    evaluation/results/guardrail_ablation.jsonl   one record per question
    evaluation/results/guardrail_ablation.md      summary tables (paste-ready)

REQUIREMENTS
    Ollama running with gemma3:4b, and the vector store already populated.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.questions import QUESTIONS  # noqa: E402

RESULTS_DIR = ROOT / "evaluation" / "results"
OUT_JSONL = RESULTS_DIR / "guardrail_ablation.jsonl"
OUT_MD = RESULTS_DIR / "guardrail_ablation.md"

CATEGORIES = [
    "rag", "market", "macro",
    "multi_rag_market", "multi_rag_macro", "multi_market_macro", "multi_all",
]


# ── instrumentation ───────────────────────────────────────────────────────────

def instrument(orch, sink: dict):
    """
    Wrap the two guardrails (and ticker resolution) so that every call records
    what came in and what went out. `sink` is cleared before each question.
    """
    orig_guard = orch._guard_macro_route
    orig_floor = orch._ensure_required_agents
    orig_tick = orch._extract_tickers_for_rag

    def guard(question, route):
        out = orig_guard(question, route)
        sink["route_raw"] = route
        sink["route_final"] = out
        return out

    def floor(question, agents):
        out = orig_floor(question, agents)
        sink["agents_raw"] = list(agents)
        sink["agents_final"] = list(out)
        return out

    def tick(question, history=""):
        out = orig_tick(question, history)
        # the deterministic scan runs first inside the orchestrator; if it
        # matched, the LLM was never called for this question
        from tickers import scan_known_tickers
        scanned = scan_known_tickers(question)
        sink["tickers_deterministic"] = scanned
        sink["tickers_final"] = list(out)
        sink["ticker_path"] = "deterministic" if scanned else "llm_fallback"
        return out

    orch._guard_macro_route = guard
    orch._ensure_required_agents = floor
    orch._extract_tickers_for_rag = tick


# ── scoring ───────────────────────────────────────────────────────────────────

def score(records: list[dict]) -> dict:
    """Accuracy of the raw LLM decision vs the final one, overall and per category."""
    def blank():
        return {
            "route_raw": 0, "route_final": 0,
            "sel_raw": 0, "sel_final": 0,
            "recall_raw": 0, "recall_final": 0,
            "n": 0,
        }

    agg = blank()
    # NOTE: must be a fresh dict per category — copying `agg` here would carry
    # over the running totals and produce cumulative per-category counts.
    per = defaultdict(blank)

    for r in records:
        cat = r["category"]
        exp_route = r["expected_route"]
        exp_agents = set(r["expected_agents"])

        # Routes: if the guardrail never fired, raw == final.
        route_raw = r.get("route_raw") or r["route_final"]
        route_final = r["route_final"]
        # Agents: the selector only runs on the multi route; elsewhere the route
        # itself determines the single agent, so raw == final by construction.
        agents_raw = set(r.get("agents_raw") or r["agents_final"])
        agents_final = set(r["agents_final"])

        for bucket in (agg, per[cat]):
            bucket["n"] += 1
            bucket["route_raw"] += route_raw == exp_route
            bucket["route_final"] += route_final == exp_route
            bucket["sel_raw"] += agents_raw == exp_agents
            bucket["sel_final"] += agents_final == exp_agents
            bucket["recall_raw"] += exp_agents.issubset(agents_raw)
            bucket["recall_final"] += exp_agents.issubset(agents_final)

    return {"overall": agg, "per_category": dict(per)}


def write_report(records: list[dict], s: dict) -> None:
    o = s["overall"]
    n = o["n"]

    def pct(x):
        return f"{x}/{n}"

    lines = [
        "# Guardrail ablation — raw LLM decision vs deterministic layer",
        "",
        "**Financial Multi-Agent System** · Filippo Maria Incecchi · "
        "Università degli Studi di Brescia",
        "",
        "Same 100-question benchmark, same model (`gemma3:4b`, temperature 0). "
        "The columns differ only in whether the deterministic guardrails are "
        "applied to the model's decision.",
        "",
        "## Overall",
        "",
        "| Decision | Raw LLM | After guardrails | Δ |",
        "|---|---|---|---|",
        f"| Routing accuracy | {pct(o['route_raw'])} | {pct(o['route_final'])} | "
        f"{o['route_final'] - o['route_raw']:+d} |",
        f"| Agent selection (exact) | {pct(o['sel_raw'])} | {pct(o['sel_final'])} | "
        f"{o['sel_final'] - o['sel_raw']:+d} |",
        f"| Needed-agent recall | {pct(o['recall_raw'])} | {pct(o['recall_final'])} | "
        f"{o['recall_final'] - o['recall_raw']:+d} |",
        "",
        "## By category",
        "",
        "| Category | n | Routing raw → final | Selection raw → final | Recall raw → final |",
        "|---|---|---|---|---|",
    ]
    for cat in CATEGORIES:
        c = s["per_category"].get(cat)
        if not c:
            continue
        lines.append(
            f"| {cat} | {c['n']} | {c['route_raw']} → {c['route_final']} "
            f"| {c['sel_raw']} → {c['sel_final']} "
            f"| {c['recall_raw']} → {c['recall_final']} |"
        )

    # cases the guardrails actually changed
    changed = [
        r for r in records
        if (r.get("route_raw") and r["route_raw"] != r["route_final"])
        or (r.get("agents_raw") and set(r["agents_raw"]) != set(r["agents_final"]))
    ]
    lines += [
        "",
        f"## Cases modified by the guardrails ({len(changed)})",
        "",
        "| id | question | raw | final |",
        "|---|---|---|---|",
    ]
    for r in changed:
        raw = f"route={r.get('route_raw')} agents={r.get('agents_raw')}"
        fin = f"route={r['route_final']} agents={r['agents_final']}"
        lines.append(f"| {r['id']} | {r['question'][:60]} | {raw} | {fin} |")

    det = sum(1 for r in records if r.get("ticker_path") == "deterministic")
    lines += [
        "",
        "## Ticker resolution path",
        "",
        f"- resolved deterministically: **{det}/{len(records)}**",
        f"- fell back to the LLM: **{len(records) - det}/{len(records)}**",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] {OUT_MD}")


# ── main ──────────────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out, seen = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            seen[r["id"]] = r          # keep the latest record per id
    return list(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from an existing run")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        recs = load(OUT_JSONL)
        if not recs:
            sys.exit("No previous run found.")
        write_report(recs, score(recs))
        return

    if args.fresh and OUT_JSONL.exists():
        OUT_JSONL.unlink()

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    done = {r["id"] for r in load(OUT_JSONL)}
    remaining = [q for q in questions if q["id"] not in done]

    print("=" * 70)
    print("  GUARDRAIL ABLATION — raw LLM decision vs deterministic layer")
    print(f"  {len(done)} already done · {len(remaining)} to run · {len(questions)} total")
    print("=" * 70)

    if remaining:
        from agents.orchestrator import Orchestrator
        orch = Orchestrator()
        sink: dict = {}
        instrument(orch, sink)

        with open(OUT_JSONL, "a", encoding="utf-8") as fh:
            for i, q in enumerate(remaining, 1):
                print(f"\n[{i}/{len(remaining)}] ({q['category']}) {q['question']}")
                sink.clear()
                try:
                    trace = orch.run_traced(q["question"], use_memory=False)
                    rec = {
                        "id": q["id"],
                        "category": q["category"],
                        "question": q["question"],
                        "expected_route": q["expected_route"],
                        "expected_agents": q["expected_agents"],
                        # raw = what the model decided; final = after the guardrails
                        "route_raw": sink.get("route_raw"),
                        "route_final": trace["route"],
                        "agents_raw": sink.get("agents_raw"),
                        "agents_final": trace["agents"],
                        "ticker_path": sink.get("ticker_path"),
                        "tickers_final": trace.get("tickers", []),
                    }
                except Exception as e:                       # noqa: BLE001
                    rec = {"id": q["id"], "category": q["category"],
                           "question": q["question"],
                           "expected_route": q["expected_route"],
                           "expected_agents": q["expected_agents"],
                           "error": f"{type(e).__name__}: {e}"}
                    print(f"    ERROR: {rec['error']}")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()                                    # safe to Ctrl+C

    recs = [r for r in load(OUT_JSONL) if not r.get("error")]
    s = score(recs)
    o = s["overall"]
    print("\n" + "=" * 70)
    print(f"  Routing   raw {o['route_raw']}/{o['n']}  →  final {o['route_final']}/{o['n']}")
    print(f"  Selection raw {o['sel_raw']}/{o['n']}  →  final {o['sel_final']}/{o['n']}")
    print(f"  Recall    raw {o['recall_raw']}/{o['n']}  →  final {o['recall_final']}/{o['n']}")
    print("=" * 70)
    write_report(recs, s)


if __name__ == "__main__":
    main()
