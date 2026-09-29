# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Human validation of the automatic judge.

WHY
    The RAGAS scores are produced by a language model acting as judge. Reporting
    them without any human check leaves open the objection that the evaluation
    is one model grading another. Rating a small stratified sample by hand and
    comparing it against the judge addresses that directly: it does not validate
    every score, but it shows whether the judge and a human reader agree on the
    cases a reader can verify.

WHAT IT DOES
    1) --sample   picks N answers stratified across the seven categories,
                  writes a scoring sheet (Markdown) for manual rating and a CSV
                  to fill in, with the judge's scores HIDDEN so the rating is
                  not anchored by them.
    2) --extend   enlarges an existing sample to N answers WITHOUT disturbing the
                  ratings already given. The answers already rated are kept, the
                  remainder is drawn at random from the answers not yet rated, and
                  the per-category quota is proportional to the benchmark itself
                  (20/20/20/10/10/10/10), so the enlarged sample mirrors the
                  composition of the hundred questions. The sheet it writes
                  contains ONLY the answers still to rate.
    3) --compare  reads the filled-in CSV and reports agreement with the judge
                  (mean absolute difference and agreement within a tolerance),
                  plus a short report ready to paste into the thesis.

    Mark a row with `y` in the `judge_error` column when the disagreement is not
    a difference of opinion but a demonstrable mistake by the judge. The report
    then gives the figures both with and without those rows, so the claim is
    reported honestly in either reading.

HOW TO RATE (the sheet repeats these instructions)
    Judge ONLY against the evidence shown — not against what you know to be true
    about the world. That is what faithfulness measures; judging against outside
    knowledge would measure something else.

    FAITHFULNESS — count, do not grade.
        RAGAS decomposes an answer into atomic claims and checks each one, so its
        values are ratios (0.857 = 6/7, 0.909 = 10/11). To be comparable, do the
        same: write how many distinct factual claims the answer makes
        (`claims_total`) and how many are supported by the evidence
        (`claims_supported`). The script computes the ratio for you.
        A claim is a single verifiable assertion — a figure, a date, a stated
        fact. Ignore hedges, restatements of the question and connective prose.

    ANSWER RELEVANCY — a coarse scale is the only honest option.
        RAGAS computes this from embedding similarity, a mechanism a human
        cannot replicate, so an exact match is not the goal. Use:
            1.0  answers the question asked, fully
            0.5  partially answers it, or answers something adjacent
            0.0  does not answer the question
        The comparison for this metric is therefore weaker than for
        faithfulness, and is reported as such.

USAGE
    python evaluation/human_validation.py --sample 10
    ... open evaluation/results/human_validation_sheet.md, rate, fill the CSV ...
    python evaluation/human_validation.py --compare

    python evaluation/human_validation.py --extend 30     # keeps the first ten
    ... the sheet now holds only the twenty new answers ...
    python evaluation/human_validation.py --compare
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "results"
TRACES = RESULTS_DIR / "phase3_traces.jsonl"
SCORES = RESULTS_DIR / "phase3_scores.csv"
SHEET_MD = RESULTS_DIR / "human_validation_sheet.md"
SHEET_CSV = RESULTS_DIR / "human_validation.csv"
REPORT = RESULTS_DIR / "human_validation_report.md"

CATEGORIES = ["rag", "market", "macro", "multi_rag_market",
              "multi_rag_macro", "multi_market_macro", "multi_all"]
SEED = 20260914          # fixed so the sample is reproducible
SEED_EXTEND = 20260921   # fixed so the enlargement is reproducible too

# how many questions of each category the benchmark itself contains; the
# enlarged sample is drawn in the same proportions
BENCHMARK_SIZE = {"rag": 20, "market": 20, "macro": 20,
                  "multi_rag_market": 10, "multi_rag_macro": 10,
                  "multi_market_macro": 10, "multi_all": 10}

CSV_COLUMNS = ["id", "category", "question", "claims_total", "claims_supported",
               "human_answer_relevancy", "judge_error", "notes"]


def load_traces() -> dict:
    seen = {}
    for line in TRACES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("error"):
                seen[r["id"]] = r
    return seen


def load_judge_scores() -> dict:
    """id -> {metric: value} from the RAGAS per-question CSV."""
    out = {}
    if not SCORES.exists():
        return out
    with open(SCORES, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rid = row.get("id")
            if not rid:
                continue
            rec = {}
            for k, v in row.items():
                kl = k.lower()
                if "faithful" in kl or "relevanc" in kl:
                    try:
                        rec[kl] = float(v)
                    except (TypeError, ValueError):
                        pass
            out[rid] = rec
    return out


def pick(traces: dict, n: int) -> list[dict]:
    """Stratified sample: spread across categories, deterministic."""
    rng = random.Random(SEED)
    by_cat = {c: [] for c in CATEGORIES}
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from evaluation.questions import QUESTIONS
    cat_of = {q["id"]: q["category"] for q in QUESTIONS}
    for rid, r in traces.items():
        c = cat_of.get(rid)
        if c in by_cat:
            by_cat[c].append(r | {"category": c, "id": rid})

    chosen, i = [], 0
    while len(chosen) < n:
        progressed = False
        for c in CATEGORIES:
            pool = by_cat[c]
            if len(pool) > i:
                chosen.append(rng.choice(pool) if i == 0 else pool[i])
                progressed = True
                if len(chosen) == n:
                    break
        if not progressed:
            break
        i += 1
    return chosen[:n]


def cat_map() -> dict:
    sys.path.insert(0, str(ROOT))
    from evaluation.questions import QUESTIONS
    return {q["id"]: q["category"] for q in QUESTIONS}


def by_category(traces: dict) -> dict:
    cat = cat_map()
    out = {c: [] for c in CATEGORIES}
    for rid, r in traces.items():
        c = cat.get(rid)
        if c in out:
            out[c].append(r | {"category": c, "id": rid})
    for c in out:
        out[c].sort(key=lambda r: r["id"])
    return out


def proportional_targets(n: int, already: dict) -> dict:
    """
    How many answers each category contributes to a sample of n, in the same
    proportions as the benchmark. A category never contributes fewer than the
    answers already rated in it, so enlarging a sample never discards work.
    """
    total = sum(BENCHMARK_SIZE.values())
    raw = {c: n * BENCHMARK_SIZE[c] / total for c in CATEGORIES}
    tgt = {c: int(raw[c]) for c in CATEGORIES}
    remainder = n - sum(tgt.values())
    for c in sorted(CATEGORIES, key=lambda c: raw[c] - int(raw[c]), reverse=True)[:remainder]:
        tgt[c] += 1
    for c in CATEGORIES:
        tgt[c] = max(tgt[c], already.get(c, 0))
    return tgt


def read_filled() -> dict:
    """Ratings already given, keyed by id, so an extension preserves them."""
    if not SHEET_CSV.exists():
        return {}
    out = {}
    with open(SHEET_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (row.get("claims_total") or "").strip() or \
               (row.get("human_answer_relevancy") or "").strip():
                out[row["id"]] = row
    return out


def write_csv(rows: list[dict], filled: dict) -> None:
    with open(SHEET_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            old = filled.get(r["id"], {})
            w.writerow({"id": r["id"], "category": r["category"],
                        "question": r["question"],
                        "claims_total": old.get("claims_total", ""),
                        "claims_supported": old.get("claims_supported", ""),
                        "human_answer_relevancy": old.get("human_answer_relevancy", ""),
                        "judge_error": old.get("judge_error", ""),
                        "notes": old.get("notes", "")})


def sheet_instructions(intro: list[str]) -> list[str]:
    return intro + [
        "",
        "**Rate each answer against the EVIDENCE SHOWN, not against outside "
        "knowledge.** If the system faithfully reports a figure that happens to be "
        "wrong in reality, that still counts as supported.",
        "",
        "## Faithfulness — count the claims, do not grade",
        "",
        "RAGAS splits an answer into atomic claims and checks each one, so its "
        "scores are ratios (0.857 = 6/7). Do the same:",
        "",
        "- `claims_total` — how many distinct factual claims the answer makes. "
        "A claim is one verifiable assertion: a figure, a date, a stated fact. "
        "Ignore hedges, restatements of the question and connective prose.",
        "- `claims_supported` — how many of those are supported by the evidence below.",
        "",
        "The script computes `claims_supported / claims_total` for you.",
        "",
        "## Answer relevancy — a coarse scale",
        "",
        "| score | meaning |",
        "|---|---|",
        "| 1.0 | answers the question asked, fully |",
        "| 0.5 | partially answers it, or answers something adjacent |",
        "| 0.0 | does not answer the question |",
        "",
        f"Fill in `{SHEET_CSV.name}` (columns `claims_total`, `claims_supported`, "
        "`human_answer_relevancy`). Leave `judge_error` empty unless you find a "
        "demonstrable mistake by the judge, in which case write `y`. Then run "
        "`python evaluation/human_validation.py --compare`.",
        "",
        "---",
        "",
    ]


def sheet_body(sample: list[dict], start: int = 1) -> list[str]:
    md = []
    for k, r in enumerate(sample, start):
        md += [
            f"## {k}. `{r['id']}` — {r['category']}",
            "",
            f"**Question.** {r['question']}",
            "",
            f"**Route taken.** {r.get('route','')} · agents: {', '.join(r.get('agents') or [])}",
            "",
            "**Answer.**",
            "",
            "> " + (r.get("answer") or "(none)").replace("\n", "\n> "),
            "",
            "**Evidence the system used.**",
            "",
        ]
        for j, c in enumerate(r.get("contexts") or [], 1):
            snippet = c if len(c) < 1200 else c[:1200] + " […]"
            md += [f"*Evidence {j}*", "", "```", snippet, "```", ""]
        md += ["---", ""]
    return md


def do_extend(n: int) -> None:
    traces = load_traces()
    if not traces:
        sys.exit(f"No traces at {TRACES}")
    filled = read_filled()
    if not filled:
        sys.exit("Nothing rated yet — use --sample instead.")
    cat = cat_map()
    already = {}
    for rid in filled:
        c = cat.get(rid)
        already[c] = already.get(c, 0) + 1
    if sum(already.values()) > n:
        sys.exit(f"{sum(already.values())} answers are already rated; --extend {n} would shrink the sample.")

    pools = by_category(traces)
    target = proportional_targets(n, already)
    rng = random.Random(SEED_EXTEND)

    chosen, added = [], []
    for c in CATEGORIES:
        rated = [r for r in pools[c] if r["id"] in filled]
        pool = [r for r in pools[c] if r["id"] not in filled]
        need = target[c] - len(rated)
        extra = rng.sample(pool, min(need, len(pool))) if need > 0 else []
        extra.sort(key=lambda r: r["id"])
        chosen += rated + extra
        added += extra

    write_csv(chosen, filled)

    intro = [
        "# Human validation sheet — additional answers",
        "",
        f"The sample now holds {len(chosen)} answers: the {len(filled)} already "
        f"rated and {len(added)} new ones, which are the only ones reproduced below. "
        f"The new answers were drawn at random from those not yet rated "
        f"(seed {SEED_EXTEND}, reproducible), with a per-category quota proportional "
        "to the benchmark itself.",
        "",
        "| category | in benchmark | in sample |",
        "|---|---|---|",
    ] + [f"| {c} | {BENCHMARK_SIZE[c]} | {target[c]} |" for c in CATEGORIES]

    SHEET_MD.write_text("\n".join(sheet_instructions(intro) +
                                  sheet_body(added, start=len(filled) + 1)),
                        encoding="utf-8")
    print(f"sample: {len(filled)} already rated + {len(added)} new = {len(chosen)}")
    for c in CATEGORIES:
        print(f"   {c:<20} {target[c]}")
    print(f"\n[sheet] {SHEET_MD}   ← contains only the {len(added)} new answers")
    print(f"[csv]   {SHEET_CSV}   ← your earlier ratings are already in it")
    print("\nJudge scores are deliberately NOT shown, so your rating stays independent.")


def do_sample(n: int) -> None:
    traces = load_traces()
    if not traces:
        sys.exit(f"No traces at {TRACES}")
    sample = pick(traces, n)

    md = [
        "# Human validation sheet",
        "",
        f"{len(sample)} answers sampled across categories (seed {SEED}, reproducible).",
        "",
        "**Rate each answer against the EVIDENCE SHOWN, not against outside "
        "knowledge.** If the system faithfully reports a figure that happens to be "
        "wrong in reality, that still counts as supported.",
        "",
        "## Faithfulness — count the claims, do not grade",
        "",
        "RAGAS splits an answer into atomic claims and checks each one, so its "
        "scores are ratios (0.857 = 6/7). Do the same:",
        "",
        "- `claims_total` — how many distinct factual claims the answer makes. "
        "A claim is one verifiable assertion: a figure, a date, a stated fact. "
        "Ignore hedges, restatements of the question and connective prose.",
        "- `claims_supported` — how many of those are supported by the evidence below.",
        "",
        "The script computes `claims_supported / claims_total` for you.",
        "",
        "## Answer relevancy — a coarse scale",
        "",
        "| score | meaning |",
        "|---|---|",
        "| 1.0 | answers the question asked, fully |",
        "| 0.5 | partially answers it, or answers something adjacent |",
        "| 0.0 | does not answer the question |",
        "",
        f"Fill in `{SHEET_CSV.name}` (columns `claims_total`, `claims_supported`, "
        "`human_answer_relevancy`), then run "
        "`python evaluation/human_validation.py --compare`.",
        "",
        "---",
        "",
    ]
    for k, r in enumerate(sample, 1):
        md += [
            f"## {k}. `{r['id']}` — {r['category']}",
            "",
            f"**Question.** {r['question']}",
            "",
            f"**Route taken.** {r.get('route','')} · agents: {', '.join(r.get('agents') or [])}",
            "",
            "**Answer.**",
            "",
            "> " + (r.get("answer") or "(none)").replace("\n", "\n> "),
            "",
            "**Evidence the system used.**",
            "",
        ]
        for j, c in enumerate(r.get("contexts") or [], 1):
            snippet = c if len(c) < 1200 else c[:1200] + " […]"
            md += [f"*Evidence {j}*", "", "```", snippet, "```", ""]
        md += ["---", ""]

    SHEET_MD.write_text("\n".join(md), encoding="utf-8")

    with open(SHEET_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "category", "question",
                    "claims_total", "claims_supported",
                    "human_answer_relevancy", "notes"])
        for r in sample:
            w.writerow([r["id"], r["category"], r["question"], "", "", "", ""])

    print(f"[sheet] {SHEET_MD}")
    print(f"[csv]   {SHEET_CSV}   ← fill in the two score columns")
    print("\nJudge scores are deliberately NOT shown, so your rating is independent.")


def do_compare() -> None:
    if not SHEET_CSV.exists():
        sys.exit(f"{SHEET_CSV} not found — run --sample first.")
    judge = load_judge_scores()

    rows = []
    with open(SHEET_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tot = (row.get("claims_total") or "").strip()
            sup = (row.get("claims_supported") or "").strip()
            rel = (row.get("human_answer_relevancy") or "").strip()
            if not tot and not rel:
                continue
            # faithfulness is derived from the counts, mirroring how RAGAS builds it
            if tot:
                try:
                    t, s = float(tot), float(sup or 0)
                    # an answer that asserts nothing has no faithfulness to measure:
                    # 0/0 is undefined, not zero, so it is left out of that comparison
                    row["human_faithfulness"] = f"{s / t:.3f}" if t else ""
                except (TypeError, ValueError):
                    row["human_faithfulness"] = ""
            rows.append(row)
    if not rows:
        sys.exit("No human scores filled in yet.")

    def judge_val(rid, kind):
        rec = judge.get(rid, {})
        for k, v in rec.items():
            if kind in k:
                return v
        return None

    comp = []
    for r in rows:
        rid = r["id"]
        for kind, col in (("faithful", "human_faithfulness"),
                          ("relevanc", "human_answer_relevancy")):
            jv = judge_val(rid, kind)
            try:
                hv = float(r[col])
            except (TypeError, ValueError):
                continue
            if jv is not None:
                comp.append((rid, r["category"], kind, hv, jv, abs(hv - jv)))

    if not comp:
        sys.exit("Could not match any human score to a judge score.")

    flagged = {r["id"] for r in rows
               if (r.get("judge_error") or "").strip().lower() in ("y", "yes", "1", "true")}

    def summarise(kind, skip=frozenset()):
        sel = [c for c in comp if c[2] == kind and c[0] not in skip]
        if not sel:
            return None
        mad = sum(c[5] for c in sel) / len(sel)
        within = sum(1 for c in sel if c[5] <= 0.25)
        return len(sel), mad, within

    lines = [
        "# Human validation of the automatic judge",
        "",
        "**Financial Multi-Agent System** · Filippo Maria Incecchi · "
        "Università degli Studi di Brescia",
        "",
        f"{len(rows)} answers rated by hand against the evidence the system used, "
        "without sight of the judge's scores.",
        "",
        "| Metric | n | Mean absolute difference | Agreement within 0.25 |",
        "|---|---|---|---|",
    ]
    for kind, label in (("faithful", "Faithfulness"), ("relevanc", "Answer relevancy")):
        st = summarise(kind)
        if st:
            n, mad, within = st
            lines.append(f"| {label} | {n} | {mad:.2f} | {within}/{n} |")
        if flagged:
            st = summarise(kind, skip=flagged)
            if st:
                n, mad, within = st
                lines.append(f"| {label}, excluding verified judge errors | {n} | "
                             f"{mad:.3f} | {within}/{n} |")

    lines += ["", "## Per-answer detail", "",
              "| id | category | metric | human | judge | |Δ| | judge error |",
              "|---|---|---|---|---|---|---|"]
    for rid, cat, kind, hv, jv, dv in comp:
        mark = "yes" if rid in flagged else ""
        lines.append(f"| {rid} | {cat} | {kind} | {hv:.2f} | {jv:.2f} | {dv:.2f} | {mark} |")
    if flagged:
        lines += ["",
                  f"{len(flagged)} answer(s) were marked as demonstrable mistakes by "
                  "the judge rather than differences of opinion; the table above "
                  "reports the agreement both with and without them."]

    lines += [
        "",
        "## Reading",
        "",
        "This is a small sample and is not intended to validate the judge in "
        "general. It checks whether, on answers a reader can verify, the "
        "automatic scores are consistent with the intended interpretation of the "
        "metric. Disagreements are as informative as agreements and are listed "
        "individually above.",
        "",
        "The two metrics are not equally comparable. Faithfulness was rated by "
        "counting supported claims over total claims, which is how RAGAS "
        "constructs the metric, so human and judge values are measured the same "
        "way. Answer relevancy is computed by RAGAS from embedding similarity, a "
        "procedure a human cannot reproduce; it was rated here on a three-point "
        "scale, so a larger divergence is expected by construction and should not "
        "be read as disagreement about the answers themselves.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"\n[report] {REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, metavar="N")
    ap.add_argument("--extend", type=int, metavar="N")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.sample:
        do_sample(args.sample)
    elif args.extend:
        do_extend(args.extend)
    elif args.compare:
        do_compare()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
