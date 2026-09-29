# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Streamlit web interface for the financial multi-agent system.

Run with:  streamlit run app.py
Requires Ollama running and the vector store populated (python main.py once).
"""

import streamlit as st

from config import LLM_MODEL

st.set_page_config(
    page_title="Financial Multi-Agent Assistant",
    page_icon="📊",
    layout="wide",
)

ROUTE_LABELS = {
    "rag":    "📄 Filings (RAG)",
    "market": "💹 Market",
    "macro":  "🏛️ Macro",
    "multi":  "🧩 Multi-agent",
}

EXAMPLE_QUESTIONS = [
    "What is Apple's business model?",
    "What is Tesla's current stock price and P/E ratio?",
    "What is the current US inflation rate?",
    "Assess Microsoft's strategy, its valuation, and the macro backdrop.",
]


@st.cache_resource(show_spinner="Initializing the multi-agent system…")
def get_orchestrator():
    """Build the Orchestrator once (heavy: ChromaDB + Ollama + LangGraph)."""
    from agents.orchestrator import Orchestrator
    return Orchestrator()


def render_trace(trace: dict) -> None:
    """Show which route/agents ran and the sources/data the answer used."""
    route  = trace.get("route", "")
    agents = trace.get("agents", [])
    c1, c2 = st.columns(2)
    c1.caption(f"**Route:** {ROUTE_LABELS.get(route, route or '—')}")
    c2.caption(f"**Agents:** {', '.join(agents) if agents else '—'}")

    market  = trace.get("market_summary", "")
    macro   = trace.get("macro_summary", "")
    chunks  = trace.get("rag_contexts", []) or []
    if not (market or macro or chunks):
        return

    with st.expander("🔎 Sources & data used"):
        if market:
            st.markdown("**Market data — Yahoo Finance**")
            st.code(market, language="text")
        if macro:
            st.markdown("**Macroeconomic data — FRED**")
            st.code(macro, language="text")
        if chunks:
            st.markdown(f"**Retrieved SEC-filing passages ({len(chunks)})**")
            for i, chunk in enumerate(chunks, 1):
                snippet = chunk[:800] + ("…" if len(chunk) > 800 else "")
                st.markdown(f"*Passage {i}*")
                st.write(snippet)


# ── Orchestrator + sidebar ────────────────────────────────────────────────────

orch = get_orchestrator()

with st.sidebar:
    st.title("📊 Financial Multi-Agent Assistant")
    st.caption("Router → specialized agents (RAG · Market · Macro) → synthesizer")

    st.markdown(f"**Generation model:** `{LLM_MODEL}` (local, via Ollama)")
    tickers = sorted(getattr(orch, "_indexed_tickers", set()) or set())
    st.markdown(f"**Indexed companies:** {', '.join(tickers) if tickers else '—'}")
    st.markdown("**Data sources:** SEC 10-K filings · Yahoo Finance · FRED")

    if orch.vector_store.is_empty():
        st.warning("Vector store is empty — run `python main.py` once to ingest filings.")

    st.divider()
    st.markdown("**Example questions**")
    for ex in EXAMPLE_QUESTIONS:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        orch.memory.clear()
        st.session_state.history = []
        st.rerun()


# ── Chat ──────────────────────────────────────────────────────────────────────

st.session_state.setdefault("history", [])

# Replay the conversation so far.
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("trace"):
            render_trace(msg["trace"])

# A question comes either from the input box or from an example button.
question = st.chat_input("Ask about a company, a stock, or the macro economy…")
question = question or st.session_state.pop("pending", None)

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing to the agents…"):
            try:
                trace = orch.run_traced(question, use_memory=True)
            except Exception as e:
                trace = None
                st.error(f"The system could not answer ({type(e).__name__}: {e}). "
                         f"Is Ollama running?")
        if trace:
            st.markdown(trace["answer"] or "_No answer produced._")
            render_trace(trace)
            st.session_state.history.append(
                {"role": "assistant", "content": trace["answer"], "trace": trace}
            )
