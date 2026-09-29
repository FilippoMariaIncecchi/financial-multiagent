# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Corpus audit and cleanup after the ticker-resolution ablation.

WHY
    Disabling the deterministic ticker scan forced every question through the
    LLM extractor. The extractor mistook ordinary English words for ticker
    symbols — "a", "the", "key", "open", "run", "net", "has" — and the on-demand
    ingestion faithfully downloaded and indexed the 10-K filings of the real
    companies that happen to own those symbols. The corpus was therefore
    polluted with companies that have nothing to do with the evaluation, and
    the pollution is persistent: it survives the run that caused it.

    That is a more serious failure mode than a wrong answer, and it is the
    strongest evidence in this work for the deterministic ticker guardrail.

WHAT THIS DOES
    --audit    writes a documented snapshot of the current corpus (every ticker,
               its chunk count, and whether it belongs to the benchmark, to the
               alias catalogue, or to neither) plus a short report suitable for
               the thesis. Read-only.

    --clean    removes from the vector store every ticker outside the benchmark
               set. Asks for explicit confirmation and refuses to run unless an
               audit snapshot already exists.

BENCHMARK SET
    The nine companies the 100-question benchmark actually uses. Note this is
    larger than config.SEC_COMPANIES (seven): Intel and Palantir were added to
    the benchmark and appear in fifteen of the hundred questions, so removing
    them would make those questions irreproducible.

USAGE
    python evaluation/corpus_cleanup.py --audit
    python evaluation/corpus_cleanup.py --clean
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evaluation" / "results"
SNAPSHOT = RESULTS_DIR / "corpus_snapshot_before_cleanup.json"
REPORT = RESULTS_DIR / "corpus_contamination.md"

# the nine tickers the benchmark uses (config.SEC_COMPANIES plus INTC and PLTR)
BENCHMARK = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "INTC", "PLTR"}


def corpus_state():
    """ticker -> chunk count, read straight from the collection metadata."""
    from rag.vectorstore import VectorStoreManager
    vs = VectorStoreManager()
    res = vs.collection.get(include=["metadatas"])
    counts = {}
    for meta in res.get("metadatas", []):
        if isinstance(meta, dict) and "ticker" in meta:
            counts[meta["ticker"]] = counts.get(meta["ticker"], 0) + 1
    return vs, counts


def classify(counts):
    from config import COMPANY_ALIASES, SEC_COMPANIES
    aliases = set(COMPANY_ALIASES.values())
    groups = {"benchmark": {}, "alias_catalogue": {}, "spurious": {}}
    for tk, n in sorted(counts.items()):
        if tk in BENCHMARK:
            groups["benchmark"][tk] = n
        elif tk in aliases:
            groups["alias_catalogue"][tk] = n
        else:
            groups["spurious"][tk] = n
    return groups, set(SEC_COMPANIES), aliases


def do_audit() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _, counts = corpus_state()
    groups, sec7, aliases = classify(counts)

    snap = {
        "date": date.today().isoformat(),
        "total_tickers": len(counts),
        "total_chunks": sum(counts.values()),
        "benchmark_set": sorted(BENCHMARK),
        "counts": counts,
        "groups": {k: sorted(v) for k, v in groups.items()},
    }
    SNAPSHOT.write_text(json.dumps(snap, indent=2), encoding="utf-8")

    sp = groups["spurious"]
    md = [
        "# Corpus contamination from LLM ticker extraction",
        "",
        "**Financial Multi-Agent System** · Filippo Maria Incecchi · "
        "Università degli Studi di Brescia",
        "",
        f"Snapshot taken {snap['date']}, before cleanup. "
        f"{snap['total_tickers']} distinct companies, "
        f"{snap['total_chunks']} indexed chunks.",
        "",
        "## What happened",
        "",
        "During the ablation of the deterministic ticker guardrail, every question "
        "was routed through the LLM ticker extractor. The extractor returned "
        "ordinary English words as ticker symbols, and the on-demand ingestion "
        "pipeline resolved each one against SEC EDGAR, downloading and indexing "
        "the 10-K of whichever real company owns that symbol.",
        "",
        "The failure is therefore not a wrong answer but a persistent corruption "
        "of the retrieval corpus: the spurious filings remain indexed and "
        "available to later queries until removed by hand.",
        "",
        "## Corpus composition at snapshot time",
        "",
        "| Group | Tickers | Chunks |",
        "|---|---|---|",
        f"| Benchmark companies | {len(groups['benchmark'])} | "
        f"{sum(groups['benchmark'].values())} |",
        f"| In the alias catalogue (legitimate on-demand downloads) | "
        f"{len(groups['alias_catalogue'])} | {sum(groups['alias_catalogue'].values())} |",
        f"| Spurious (not requested by any question) | {len(sp)} | {sum(sp.values())} |",
        "",
        "## Spurious entries",
        "",
        "| Ticker | Company the symbol belongs to | Chunks |",
        "|---|---|---|",
    ]
    known = {
        "A": "Agilent Technologies", "ACI": "Albertsons", "APPS": "Digital Turbine",
        "BE": "Bloom Energy", "HAS": "Hasbro", "IOT": "Samsara", "KEY": "KeyCorp",
        "LAKE": "Lakeland Industries", "MU": "Micron Technology", "NET": "Cloudflare",
        "OPEN": "Opendoor", "RUN": "Sunrun", "S": "SentinelOne", "T": "AT&T",
        "VSTS": "Vestis", "YOU": "Clear Secure",
    }
    for tk, n in sorted(sp.items()):
        md.append(f"| `{tk}` | {known.get(tk, '—')} | {n} |")
    md += [
        "",
        "Most of these symbols are common English words appearing in the questions "
        "themselves, which is what makes the failure mode systematic rather than "
        "incidental.",
        "",
        "## Action taken",
        "",
        "The spurious entries were removed and the corpus restricted to the nine "
        "benchmark companies, so that the results reported in Chapter 6 remain "
        "reproducible. This snapshot preserves the evidence.",
        "",
    ]
    REPORT.write_text("\n".join(md), encoding="utf-8")

    print(f"Corpus: {snap['total_tickers']} tickers, {snap['total_chunks']} chunks")
    print(f"  benchmark        : {len(groups['benchmark'])} -> {sorted(groups['benchmark'])}")
    print(f"  alias catalogue  : {len(groups['alias_catalogue'])} -> {sorted(groups['alias_catalogue'])}")
    print(f"  spurious         : {len(sp)} -> {sorted(sp)}")
    print(f"\n[snapshot] {SNAPSHOT}")
    print(f"[report]   {REPORT}")


def do_clean() -> None:
    if not SNAPSHOT.exists():
        sys.exit("Run --audit first: the snapshot is the evidence for the thesis.")

    vs, counts = corpus_state()
    remove = sorted(t for t in counts if t not in BENCHMARK)
    if not remove:
        print("Nothing to remove — the corpus already contains only the benchmark set.")
        return

    chunks = sum(counts[t] for t in remove)
    print(f"About to remove {len(remove)} tickers ({chunks} chunks):")
    print("  " + ", ".join(remove))
    print(f"\nKeeping the {len(BENCHMARK)} benchmark companies: {', '.join(sorted(BENCHMARK))}")
    if input("\nType REMOVE to confirm: ").strip() != "REMOVE":
        print("Aborted — nothing was deleted.")
        return

    vs.collection.delete(where={"ticker": {"$in": remove}})
    vs.refresh()

    _, after = corpus_state()
    left = sorted(after)
    print(f"\nDone. Corpus now holds {len(left)} tickers, {sum(after.values())} chunks.")
    print("  " + ", ".join(left))
    unexpected = set(left) - BENCHMARK
    missing = BENCHMARK - set(left)
    if unexpected:
        print(f"  ⚠️  still present but not in the benchmark: {sorted(unexpected)}")
    if missing:
        print(f"  ⚠️  benchmark companies MISSING from the corpus: {sorted(missing)}")
        print("      re-index them with: python main.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    if args.audit:
        do_audit()
    elif args.clean:
        do_clean()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
