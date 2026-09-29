# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Automated assertions for the routing / selection logic.

Run:
    python tests/test_routing.py          # deterministic logic suite (no LLM)
    python tests/test_routing.py --live   # + real routing accuracy vs the model

The default suite is fully deterministic: it injects a scripted fake LLM into
the orchestrator/agents and asserts that, GIVEN a model output, the surrounding
logic behaves correctly (router fallback, agent selection + fallback,
deterministic ticker resolution, defensive LLM parsing, synthesizer combination,
retriever ticker filter). It needs neither Ollama nor a populated vector store,
so it runs anywhere and is safe for CI.

The optional --live mode instantiates the real Orchestrator and measures how
often gemma3:4b routes/selects the labelled questions correctly. It needs Ollama
running and the vector store populated; it reports accuracy and does NOT affect
the exit code (it is an evaluation, not a unit test).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Make the project root importable when run as "python tests/test_routing.py".
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage      # noqa: E402
from agents.market_agent import MarketAgent          # noqa: E402
from agents.orchestrator import Orchestrator          # noqa: E402
from rag.retriever import Retriever                   # noqa: E402
from tickers import parse_llm_tickers, scan_known_tickers  # noqa: E402


# ── TEST HARNESS ──────────────────────────────────────────────────────────────

_passed = 0
_failed = 0
_failures: list[str] = []


def expect(label: str, got, expected) -> None:
    """Assert equality, record the result, and print a one-line verdict."""
    global _passed, _failed
    if got == expected:
        _passed += 1
        print(f"  ✅ {label}")
    else:
        _failed += 1
        _failures.append(f"{label}\n      expected: {expected!r}\n      got:      {got!r}")
        print(f"  ❌ {label}  (expected {expected!r}, got {got!r})")


class FakeLLM:
    """
    Stand-in for ChatOllama. Returns a scripted `.content` regardless of input,
    so the tests assert the LOGIC around the model deterministically.
    """

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.response)


def _orchestrator(llm_response: str = "") -> Orchestrator:
    """An Orchestrator with __init__ bypassed and a scripted fake LLM."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.llm = FakeLLM(llm_response)
    return orch


def _answer(text: str, ticker: str | None = None) -> SimpleNamespace:
    """Minimal stand-in for a *Answer dataclass (only `.answer`/`.ticker` used)."""
    ns = SimpleNamespace(answer=text)
    if ticker is not None:
        ns.ticker = ticker
    return ns


# ── 1. DETERMINISTIC TICKER RESOLUTION (tickers.py) ───────────────────────────

def test_scan_known_tickers():
    print("\n[1] Deterministic ticker scan")
    expect("single company → TSLA", scan_known_tickers("What is Tesla's business model?"), ["TSLA"])
    expect("multi company, ordered", scan_known_tickers("Compare Apple and Microsoft revenue"), ["AAPL", "MSFT"])
    expect("verbatim ticker AAPL", scan_known_tickers("What is AAPL's strategy?"), ["AAPL"])
    expect("Alphabet/Google dedup", scan_known_tickers("Alphabet vs Google"), ["GOOGL"])
    expect("two-word brand → BAC", scan_known_tickers("How does Bank of America make money?"), ["BAC"])
    expect("macro question → none", scan_known_tickers("What is the US inflation rate?"), [])
    expect("substring guard: 'intelligent'", scan_known_tickers("Is the model intelligent?"), [])
    expect("substring guard: 'metadata'", scan_known_tickers("Explain metadata handling"), [])


def test_parse_llm_tickers():
    print("\n[2] Defensive LLM ticker parsing")
    expect("filler 'Okay' → []", parse_llm_tickers("Okay"), [])
    expect("filler + real ticker → ticker only", parse_llm_tickers("Okay, here is AAPL"), ["AAPL"])
    expect("UNKNOWN → []", parse_llm_tickers("UNKNOWN"), [])
    expect("clean CSV", parse_llm_tickers("AAPL, MSFT"), ["AAPL", "MSFT"])
    expect("unknown-but-valid ticker passes (on-demand)", parse_llm_tickers("RBLX"), ["RBLX"])


# ── 2. ORCHESTRATOR TICKER EXTRACTION (two-stage) ─────────────────────────────

def test_extract_tickers_for_rag():
    print("\n[3] Orchestrator two-stage ticker extraction")
    # Stage 1 wins: even a buggy LLM output is ignored when the scan matches.
    expect("Microsoft Q → MSFT (LLM says 'Okay', ignored)",
           _orchestrator("Okay")._extract_tickers_for_rag("Assess Microsoft's strategy and valuation."),
           ["MSFT"])
    expect("Tesla Q → TSLA (LLM says 'AAPL', ignored; history has Apple)",
           _orchestrator("AAPL")._extract_tickers_for_rag(
               "How resilient is Tesla's business model given interest rates?",
               history="Turn 1: Apple stock price ... $297"),
           ["TSLA"])
    # Stage 2 fallback: unknown company / follow-up reference.
    expect("unknown company → LLM fallback → RBLX",
           _orchestrator("RBLX")._extract_tickers_for_rag("Tell me about Roblox"), ["RBLX"])
    expect("follow-up 'its' → LLM fallback → AMZN",
           _orchestrator("AMZN")._extract_tickers_for_rag("And what is its revenue?",
                                                          history="...Amazon segments..."),
           ["AMZN"])


# ── 3. ROUTER NODE ────────────────────────────────────────────────────────────

def test_router_node():
    print("\n[4] Router node (classification + fallback)")
    def route(model_out):
        orch = _orchestrator(model_out)
        return orch._router_node({"question": "q", "history": ""})["route"]
    expect("'multi' → multi", route("multi"), "multi")
    expect("whitespace/case 'MARKET\\n' → market", route("MARKET\n"), "market")
    expect("quoted '\"macro\"' → macro", route('"macro"'), "macro")
    expect("invalid 'banana' → fallback rag", route("banana"), "rag")
    expect("_route_condition reads state",
           _orchestrator()._route_condition({"route": "multi"}), "multi")


# ── 4. MULTI-AGENT SELECTION ──────────────────────────────────────────────────

def test_select_multi_agents():
    print("\n[5] Multi-agent selection (parsing + ≥2 fallback)")
    def select(model_out):
        return _orchestrator(model_out)._select_multi_agents("q")
    expect("'rag, macro'", select("rag, macro"), ["rag", "macro"])
    expect("'rag, market, macro'", select("rag, market, macro"), ["rag", "market", "macro"])
    expect("newlines+caps 'RAG\\nMacro'", select("RAG\nMacro"), ["rag", "macro"])
    expect("invalid 'banana' → fallback rag+market", select("banana"), ["rag", "market"])
    expect("single 'macro' (needs ≥2) → fallback rag+market", select("macro"), ["rag", "market"])
    expect("dedup 'rag, rag, market'", select("rag, rag, market"), ["rag", "market"])


# ── 5. SYNTHESIZER COMBINATION LOGIC ──────────────────────────────────────────

def test_synthesizer_node():
    print("\n[6] Synthesizer (single passthrough / multi synth / none)")
    base = {"question": "q", "history": "", "rag_answer": None,
            "market_answer": None, "macro_answer": None}

    single = _orchestrator()._synthesizer_node({**base, "rag_answer": _answer("RAG ONLY")})
    expect("single active → passthrough", single["final_answer"], "RAG ONLY")

    multi = _orchestrator("SYNTHESIZED")._synthesizer_node({
        **base, "rag_answer": _answer("r"), "market_answer": _answer("m", ticker="AAPL")})
    expect("two active → LLM synthesis", multi["final_answer"], "SYNTHESIZED")

    none = _orchestrator()._synthesizer_node({**base})
    expect("no active → graceful message",
           none["final_answer"], "I could not find any relevant information.")


# ── 6. ENSURE-INDEXED RETURN FILTER ───────────────────────────────────────────

def test_ensure_tickers_indexed_filter():
    print("\n[7] _ensure_tickers_indexed returns only indexed tickers")
    orch = _orchestrator()
    orch._indexed_tickers = {"AAPL", "MSFT"}
    # Both companies are cached → no download path, returns both for filtering.
    expect("cached tickers returned",
           orch._ensure_tickers_indexed("Compare Apple and Microsoft"), ["AAPL", "MSFT"])


# ── 7. RETRIEVER TICKER FILTER ────────────────────────────────────────────────

def test_build_ticker_filter():
    print("\n[8] Retriever ticker filter (ChromaDB where clause)")
    expect("none → no filter", Retriever._build_ticker_filter(None), None)
    expect("empty → no filter", Retriever._build_ticker_filter([]), None)
    expect("single → equality", Retriever._build_ticker_filter(["aapl"]), {"ticker": "AAPL"})
    expect("multi → $in deduped/upper",
           Retriever._build_ticker_filter(["AAPL", "msft", "AAPL"]),
           {"ticker": {"$in": ["AAPL", "MSFT"]}})


# ── 8. MARKET AGENT (deterministic-first, shared module) ──────────────────────

def test_market_agent_extract():
    print("\n[9] Market agent ticker extraction (shared resolution)")
    def extract(question, model_out="UNKNOWN", history=""):
        ma = MarketAgent.__new__(MarketAgent)
        ma.llm = FakeLLM(model_out)
        return ma._extract_ticker(question, history=history)
    expect("Apple price Q → AAPL (deterministic)", extract("What is Apple's current stock price?"), "AAPL")
    expect("Tesla Q → TSLA (agrees with RAG path)",
           extract("How resilient is Tesla's business model given interest rates?"), "TSLA")
    expect("market-wide, no company → UNKNOWN (no forced AAPL)",
           extract("How are rising interest rates affecting the stock market?"), "UNKNOWN")
    expect("filler 'Okay' fallback → UNKNOWN", extract("Tell me about that mystery firm", model_out="Okay"), "UNKNOWN")
    expect("follow-up 'its' → AMZN (LLM fallback)",
           extract("And its stock price?", model_out="AMZN", history="...Amazon..."), "AMZN")


DETERMINISTIC_TESTS = [
    test_scan_known_tickers,
    test_parse_llm_tickers,
    test_extract_tickers_for_rag,
    test_router_node,
    test_select_multi_agents,
    test_synthesizer_node,
    test_ensure_tickers_indexed_filter,
    test_build_ticker_filter,
    test_market_agent_extract,
]


# ── OPTIONAL: LIVE ROUTING ACCURACY (needs Ollama + vector store) ─────────────

ROUTING_CASES = [
    ("What is Tesla's business model?",                                              "rag"),
    ("What is Apple's current stock price and P/E ratio?",                           "market"),
    ("What is the current US inflation rate?",                                       "macro"),
    ("How does Nvidia's strategy compare with its recent share price performance?",  "multi"),
    ("How resilient is Tesla's business model given current US interest rates?",     "multi"),
    ("How are rising interest rates affecting the stock market right now?",          "multi"),
    ("Assess Microsoft's strategy, its current stock valuation, and the macro backdrop.", "multi"),
]

SELECTION_CASES = [
    ("How does Nvidia's strategy compare with its recent share price performance?",  {"rag", "market"}),
    ("How resilient is Tesla's business model given current US interest rates?",     {"rag", "macro"}),
    ("How are rising interest rates affecting the stock market right now?",          {"market", "macro"}),
    ("Assess Microsoft's strategy, its current stock valuation, and the macro backdrop.", {"rag", "market", "macro"}),
]


def run_live() -> None:
    """Measure real router/selection accuracy. Reports only — never fails CI."""
    print("\n" + "=" * 70)
    print("  LIVE ROUTING ACCURACY (gemma3:4b) — evaluation, not a unit test")
    print("=" * 70)

    # The whole evaluation is guarded: ChatOllama connects lazily, so a stopped
    # Ollama server only errors on the first .invoke() — we pre-flight one tiny
    # call so a missing model degrades to a clean message instead of a crash.
    try:
        orch = Orchestrator()
        orch.llm.invoke([HumanMessage(content="ping")])  # pre-flight reachability

        hits = 0
        print("\n  Routing:")
        for q, exp in ROUTING_CASES:
            got = orch._router_node({"question": q, "history": ""})["route"]
            hits += got == exp
            print(f"    {'✅' if got == exp else '❌'} [{got:>6} vs {exp:>6}] {q[:54]}")
        print(f"  Routing accuracy: {hits}/{len(ROUTING_CASES)}")

        hits = 0
        print("\n  Multi-agent selection:")
        for q, exp in SELECTION_CASES:
            got = set(orch._select_multi_agents(q))
            hits += got == exp
            print(f"    {'✅' if got == exp else '❌'} [{sorted(got)} vs {sorted(exp)}] {q[:40]}")
        print(f"  Selection accuracy: {hits}/{len(SELECTION_CASES)}")

    except Exception as e:  # Ollama down, model missing, vector store empty, …
        print(f"  ⚠️  Live evaluation skipped — could not reach the model.")
        print(f"     ({type(e).__name__}: {e})")
        print("     Start Ollama first, e.g.:  ollama serve   then   ollama run gemma3:4b")
        return


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("  ROUTING / SELECTION LOGIC — DETERMINISTIC ASSERTIONS")
    print("=" * 70)
    for test in DETERMINISTIC_TESTS:
        test()

    print("\n" + "=" * 70)
    print(f"  RESULT: {_passed} passed, {_failed} failed")
    print("=" * 70)
    if _failures:
        print("\nFailures:")
        for f in _failures:
            print(f"  • {f}")

    if "--live" in sys.argv:
        run_live()

    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
