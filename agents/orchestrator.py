# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
import re
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from agents.macro_agent import MacroAgent, MacroAnswer
from agents.market_agent import MarketAgent, MarketAnswer
from agents.rag_agent import RagAgent, RagAnswer
from config import LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE
from memory import ConversationMemory
from rag.ingestor import ingest_single_ticker
from rag.retriever import Retriever
from rag.vectorstore import VectorStoreManager
from tickers import parse_llm_tickers, scan_known_tickers


# ── ROUTING / SELECTION CUE TERMS ─────────────────────────────────────────────
# Lightweight, deterministic cue detection used by the routing guardrail and the
# multi-agent selection floor. Matched as whole words/phrases, case-insensitive.
MACRO_TERMS = (
    "interest rate", "interest rates", "rate", "rates", "inflation", "cpi",
    "gdp", "unemployment", "federal reserve", "federal funds", "fed funds",
    "fed", "monetary policy", "recession", "labor market", "jobs report",
)
MARKET_TERMS = (
    "stock", "stocks", "stock price", "share price", "shares", "share",
    "valuation", "valued", "overvalued", "undervalued", "market cap",
    "market capitalization", "p/e", "pe ratio", "price-to-earnings",
    "dividend", "stock return", "52-week", "trading volume", "equity", "equities",
)


def _text_has_any(text: str, terms) -> bool:
    """True if any term appears in `text` as a whole word/phrase."""
    t = text.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", t) for term in terms)


# ── PROMPTS ───────────────────────────────────────────────────────────────────

MULTI_TICKER_EXTRACTION_PROMPT = """Extract ALL stock ticker symbols mentioned in the user's question.
The question may be in any language (English, Italian, French, Spanish, etc.).

Rules:
- Return ONLY ticker symbols, comma-separated. No explanations, no punctuation other than commas.
- Use standard US ticker format (uppercase, 1-5 letters).
- Map company names to tickers:
    Apple → AAPL    Microsoft → MSFT    Google / Alphabet → GOOGL
    Amazon → AMZN   Meta / Facebook → META    Tesla → TSLA
    Palantir → PLTR Nvidia → NVDA    JPMorgan → JPM    Netflix → NFLX
    Salesforce → CRM    Adobe → ADBE    AMD → AMD    Intel → INTC
    Goldman Sachs → GS  Morgan Stanley → MS    Bank of America → BAC
    Qualcomm → QCOM Uber → UBER    Airbnb → ABNB    Spotify → SPOT    Shopify → SHOP
- If the ticker is written directly (e.g. AAPL, PLTR), include it as-is.
- If NO publicly traded company is mentioned, return: UNKNOWN
- If multiple companies are mentioned, return all of them comma-separated.

Examples:
  "Compare Apple and Microsoft revenue"       → AAPL, MSFT
  "How do Tesla, Amazon and Meta make money?" → TSLA, AMZN, META
  "What is Apple's strategy?"                 → AAPL
  "What is the US inflation rate?"            → UNKNOWN

Respond with ticker symbols only (comma-separated) or UNKNOWN."""



class AgentState(TypedDict):
    """
    Shared state across all nodes of the LangGraph graph.
    Each node reads from the state and writes its own output.
    """
    question:      str
    history:       str   # recent conversation turns, for follow-up resolution
    route:         str
    rag_answer:    Optional[RagAnswer]
    market_answer: Optional[MarketAnswer]
    macro_answer:  Optional[MacroAnswer]
    final_answer:  str


# ── ROUTER PROMPT ─────────────────────────────────────────────────────────────

ROUTER_PROMPT = """You are a routing assistant for a financial Q&A system.
Classify the user's question into exactly one of these categories.

CRITICAL: The question may be written in ANY language (English, Italian, French, Spanish…).
Always classify based on the TOPIC, not the language of the question.

STEP 0 — CHECK THIS FIRST, IT OVERRIDES EVERYTHING BELOW:
If the question mentions a MACRO topic (interest rate(s), inflation, CPI, GDP,
unemployment, Federal Reserve / Fed, monetary policy, recession) AND ALSO
mentions the stock market, a stock price, or a company → the answer is "multi".
Never answer "market" or "macro" alone in that situation.
  ✓ "How are rising interest rates affecting the stock market right now?" → multi
  ✓ "Do interest rates impact the stock market?"                          → multi
  ✓ "How does inflation affect Apple?"                                    → multi

Categories:

- "rag" : ANY question about a SPECIFIC COMPANY — its business model, operations,
          products, services, strategy, financials, risk factors, management,
          or anything that would be found in an annual report (10-K filing).
          DEFAULT to "rag" whenever a company name or brand is mentioned.

          English → rag:  "How does Meta generate revenue?"
                          "What is Amazon's business model?"
                          "What are Tesla's risk factors?"

          Italian → rag:  "qual è il business principale di palantir?"
                          "come guadagna apple?"
                          "qual è la strategia di microsoft?"
                          "descrivi il modello di business di nvidia"
                          "quali sono i prodotti principali di amazon?"

          RULE: If you see ANY company/brand name → classify as "rag",
                UNLESS the question is exclusively about stock price or market data.

- "market" : ONLY a SINGLE company's own stock figures — price, returns,
             market cap, P/E ratio, dividends, or trading volume — and NOTHING
             about the wider economy.
             The question must contain words like: price, stock, prezzo, azione,
             rendimento, market cap, P/E, dividendo.
             Examples: "What was Apple's stock price last year?", "TSLA 52-week high"
             NOT market: any question that also names a macro topic (interest
             rates, inflation, GDP…) — that is "multi". The phrase "the stock
             market" driven by interest rates / inflation is "multi", not market.

- "macro"  : economy-wide indicators. Use "macro" ALONE only when NO company
             and NO stock/market term is present.
             Examples: US GDP, inflation rate, Federal Reserve rate, unemployment.

- "multi"  : Questions that need MORE THAN ONE of the sources above combined.
             Any combination of two or all three is valid:
               • company info + stock price        (rag + market)
               • company info + macro economy       (rag + macro)
               • stock market + macro economy       (market + macro)
               • company + stock price + macro      (rag + market + macro)
             Examples:
               "How has Apple's revenue growth compared to US GDP growth?"          (rag + macro)
               "How resilient is Tesla's business model given current interest rates?" (rag + macro)
               "Compare Tesla's fundamentals with its current share price."         (rag + market)
               "How do rising interest rates affect the stock market right now?"    (market + macro)
               "How are rising interest rates affecting the stock market right now?" (market + macro)
               "Assess Nvidia's strategy, valuation and the macro backdrop."        (rag + market + macro)

MACRO KEYWORDS (decisive): interest rate(s), inflation, CPI, GDP, unemployment,
Federal Reserve, Fed, monetary policy, recession — and their equivalents in any
language (tassi di interesse, inflazione, disoccupazione, PIL…).
  • If a macro keyword appears TOGETHER WITH a company name OR a stock/market
    term, the question is ALWAYS "multi" (never plain "market" or plain "rag").
  • Use plain "macro" only when a macro keyword appears with NO company and NO
    stock/market term.

DECISION TREE (apply in order):
  1. Does the question contain a MACRO KEYWORD together with a company name
     and/or a stock/market term? → multi
  2. Does the question need TWO OR MORE of:
       (a) company-specific info, (b) stock price / market data,
       (c) economy-wide macro data?
     → multi
  3. A company/brand name is present and only its business, strategy or
     financials are asked → rag
  4. ONLY price/stock metrics, no macro keyword, no business question → market
  5. ONLY economy-wide macro data, no company, no stock/market term → macro

Respond with ONLY one word: rag, market, macro, or multi.
No explanations. No punctuation. No quotes."""


SYNTHESIZER_PROMPT = """You are a financial analyst. You have received answers
from specialized agents. Combine them into ONE direct, concise answer to the
user's exact question.

Strict rules:
- Answer the specific question asked — do not add background or context the user
  did not request.
- Use ONLY the facts and figures the agents provided. Every number and claim must
  come from an agent's answer; never add outside knowledge.
- Do NOT recompute or alter figures the agents gave; quote them as given.
- If an agent reports information is NOT available, keep that — do not fill the gap.
- Do NOT add investment recommendations, generic advice, disclaimers or filler
  ("investors should…", "consult a professional", "I am an AI…").
- Brief interpretation is allowed only when it directly connects figures the
  agents provided. Be concise."""


# Used inside the "multi" route to decide WHICH agents to combine. The router
# only decides that more than one source is needed; this step decides which
# specific subset (any two, or all three) actually answers the question.
MULTI_AGENT_SELECTION_PROMPT = """You are a planner for a financial Q&A system.
The user's question has already been classified as needing MORE THAN ONE
specialized agent. Decide which agents are required. Choose from:

- rag    : company-specific information from SEC 10-K filings (business model,
           strategy, products, segments, risk factors, management, financials).
- market : live stock-market data (price, returns, market cap, P/E, dividend,
           52-week range, volume) via Yahoo Finance.
- macro  : economy-wide indicators (GDP, inflation/CPI, Federal Funds interest
           rate, unemployment) via FRED.

Rules:
- Return ONLY the needed agent names, lowercase, comma-separated.
- Return AT LEAST two agents (the question already needs a combination).
- The question may be in ANY language; classify by topic, not language.
- MACRO KEYWORDS — interest rate(s), inflation, CPI, GDP, unemployment, Federal
  Reserve / Fed, monetary policy, recession (and any-language equivalents) —
  ALWAYS require "macro". Never use "market" for these: "market" is ONLY for a
  company's own stock figures (price, return, market cap, P/E, dividend, volume).
- "rag" ONLY when the question asks about a company's business, strategy, model,
  products, risks or financials. A company name ALONE does NOT require "rag":
  a pure price/valuation question ("is its P/E justified given rates?") is
  market + macro, NOT rag.

Examples:
  "How does Apple's revenue compare to its stock performance?"           → rag, market
  "How resilient is Tesla's business model given current interest rates?" → rag, macro
  "Is Tesla's strategy resilient given current interest rates?"          → rag, macro
  "Compare Nvidia's fundamentals, share price and the macro cycle."      → rag, market, macro
  "How are rising interest rates affecting the stock market right now?"  → market, macro
  "How do interest rates affect the stock market right now?"             → market, macro

Respond with comma-separated agent names only. No other text."""


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Orchestrator agent: coordinates the RAG, Market, and Macro agents
    through a LangGraph execution graph.

    It also handles on-demand downloading of SEC EDGAR filings:
    if the user asks about a ticker that is not yet indexed,
    the orchestrator downloads and indexes it before invoking the RAG agent.
    """

    def __init__(self):
        self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, num_ctx=LLM_NUM_CTX)

        # RAG components
        self.vector_store = VectorStoreManager()
        self.retriever    = Retriever(self.vector_store)
        self.rag_agent    = RagAgent(self.retriever)

        # Specialized agents
        self.market_agent = MarketAgent()
        self.macro_agent  = MacroAgent()

        # Conversational memory: shared (read-only) with every agent so that
        # follow-up questions can be resolved against recent turns.
        self.memory = ConversationMemory()

        # Cache of tickers already indexed in the vector store.
        # Loaded once at bootstrap and updated on-demand.
        # Avoids querying ChromaDB on every question.
        self._indexed_tickers: set[str] = self.vector_store.get_indexed_tickers()
        if self._indexed_tickers:
            print(f"[Orchestrator] Already indexed tickers: {sorted(self._indexed_tickers)}")

        self.graph = self._build_graph()

    # ── ON-DEMAND INGESTION ───────────────────────────────────────────────────

    def _extract_tickers_for_rag(self, question: str, history: str = "") -> list[str]:
        """
        Resolves all tickers a question is about. Two-stage strategy
        (shared with the Market agent via the `tickers` module):

          1. Deterministic scan of the question text against the known company
             map. Precise and immune to small-model errors — this alone handles
             every company in the catalogue, regardless of LLM behaviour.
          2. LLM fallback ONLY when the scan finds nothing — i.e. the company is
             outside the catalogue (so it can be downloaded on demand) or the
             question is a follow-up that relies on the conversation history
             ("and its revenue?"). The LLM output is then parsed defensively.

        Returns a list of uppercase tickers, or an empty list if none is found.
        """
        # ── Stage 1: deterministic resolution from the question text ──────────
        matched = scan_known_tickers(question)
        if matched:
            print(f"[Orchestrator] Tickers matched from question: {matched}")
            return matched

        # ── Stage 2: LLM fallback (unknown company or follow-up reference) ────
        messages = [SystemMessage(content=MULTI_TICKER_EXTRACTION_PROMPT)]
        if history:
            messages.append(SystemMessage(
                content=f"Conversation so far (use it to resolve references "
                        f"like 'it' or 'that company'):\n{history}"
            ))
        messages.append(HumanMessage(content=question))
        response = self.llm.invoke(messages)

        tickers = parse_llm_tickers(response.content)
        if tickers:
            print(f"[Orchestrator] Tickers extracted by LLM (fallback): {tickers}")
        else:
            print("[Orchestrator] No ticker identified.")
        return tickers

    def _ensure_tickers_indexed(self, question: str, history: str = "") -> list[str]:
        """
        Ensures that ALL tickers mentioned in the question are in the
        vector store. For each missing one, downloads and indexes the 10-K
        on-demand from SEC EDGAR.

        For each ticker:
          1. In-memory cache (fast path, O(1)).
          2. If not cached, reload from the DB once per call
             (handles tickers indexed in previous sessions).
          3. If still absent, start the on-demand download.
          4. If the download fails, print an explicit message
             and continue with the remaining tickers.

        Returns:
            The subset of requested tickers that are actually indexed and can
            therefore be used to filter retrieval. Tickers that could not be
            downloaded are excluded so the retriever filter stays valid.
        """
        tickers = self._extract_tickers_for_rag(question, history)

        if not tickers:
            print("[Orchestrator] No ticker identified — proceeding without download.")
            return []

        print(f"[Orchestrator] Tickers to verify: {tickers}")

        # Reload fresh tickers from the DB once for the entire call
        fresh_tickers: set[str] | None = None

        for ticker in tickers:
            # ── Fast path: already in the in-memory cache ─────────────────────
            if ticker in self._indexed_tickers:
                print(f"[Orchestrator] '{ticker}' already indexed (cache hit).")
                continue

            # ── Slow path: reload from the DB (once) ──────────────────────────
            if fresh_tickers is None:
                fresh_tickers = self.vector_store.get_indexed_tickers()
                self._indexed_tickers.update(fresh_tickers)

            if ticker in self._indexed_tickers:
                print(f"[Orchestrator] '{ticker}' found in the vector store (cache updated).")
                continue

            # ── On-demand download from SEC EDGAR ─────────────────────────────
            print(f"[Orchestrator] '{ticker}' not indexed — starting on-demand download...")
            success = ingest_single_ticker(ticker, self.vector_store)

            if success:
                self.vector_store.refresh()
                self._indexed_tickers.add(ticker)
                print(f"[Orchestrator] ✅ '{ticker}' indexed successfully.")
            else:
                print(
                    f"[Orchestrator] ⚠️  No SEC EDGAR document found for '{ticker}'. "
                    f"The ticker may not be listed in the US or may not have "
                    f"10-K filings available. The RAG agent will respond using the "
                    f"documents of the other available companies."
                )

        # Only return tickers that are actually indexed — these are the ones
        # the retriever can safely filter on.
        indexed = [t for t in tickers if t in self._indexed_tickers]
        if indexed:
            print(f"[Orchestrator] Tickers available for retrieval filter: {indexed}")
        return indexed

    # ── MULTI-AGENT SELECTION ────────────────────────────────────────────────

    def _select_multi_agents(self, question: str, history: str = "") -> list[str]:
        """
        Decides which agents the 'multi' route should combine for this question.
        Returns an ordered list drawn from {"rag", "market", "macro"} with at
        least two entries. Falls back to ["rag", "market"] if the LLM output
        cannot be parsed into a valid combination.
        """
        messages = [SystemMessage(content=MULTI_AGENT_SELECTION_PROMPT)]
        if history:
            messages.append(SystemMessage(
                content=f"Conversation so far (use it to resolve references):\n{history}"
            ))
        messages.append(HumanMessage(content=question))
        response = self.llm.invoke(messages)

        raw   = response.content.strip().lower().replace("\n", ",")
        valid = {"rag", "market", "macro"}

        ordered: list[str] = []
        for token in raw.split(","):
            name = token.strip()
            if name in valid and name not in ordered:
                ordered.append(name)

        # The 'multi' route requires a combination — fall back if the model
        # returned fewer than two valid agents.
        if len(ordered) < 2:
            print(f"[Orchestrator] ⚠️  Multi selection '{response.content.strip()}' "
                  f"invalid → defaulting to rag + market")
            ordered = ["rag", "market"]

        # Deterministic floor: never drop an agent the question clearly needs.
        ordered = self._ensure_required_agents(question, ordered)

        print(f"[Orchestrator] MULTI agents selected: {ordered}")
        return ordered

    @staticmethod
    def _ensure_required_agents(question: str, agents: list[str]) -> list[str]:
        """
        Guarantees recall: if the question carries an explicit macro cue or an
        explicit stock/valuation cue, the matching agent must be present even if
        the LLM selector dropped it. Only ADDS agents (never removes) — this is
        what fixes the market+macro cascade (dropping macro on a rate question
        and then inventing a rate, or dropping market and not valuing the stock).
        """
        agents = list(agents)
        if "macro" not in agents and _text_has_any(question, MACRO_TERMS):
            agents.append("macro")
        if "market" not in agents and _text_has_any(question, MARKET_TERMS):
            agents.append("market")
        return agents

    def _guard_macro_route(self, question: str, route: str) -> str:
        """
        Deterministic routing guardrail (downgrade only). A question mentioning a
        macro indicator but NO company and NO stock/market term is pure macro —
        never 'multi'. The company/market checks protect genuine multi questions.
        """
        if route != "multi":
            return route
        if scan_known_tickers(question):            # a specific company is named
            return route
        if _text_has_any(question, MARKET_TERMS):   # an explicit stock/market angle
            return route
        if _text_has_any(question, MACRO_TERMS):
            print("[Orchestrator] Guardrail: pure-macro question → downgrade multi → macro")
            return "macro"
        return route

    # ── NODES ──────────────────────────────────────────────────────────────────

    def _router_node(self, state: AgentState) -> AgentState:
        """Classifies the question and sets the route via a single LLM call."""
        question = state["question"]
        history  = state.get("history", "")
        print(f"\n[Orchestrator] Routing question: '{question}'")

        messages = [SystemMessage(content=ROUTER_PROMPT)]
        if history:
            messages.append(SystemMessage(
                content=f"Conversation so far (a follow-up may rely on it; "
                        f"classify the LATEST user question in context):\n{history}"
            ))
        messages.append(HumanMessage(content=question))
        response = self.llm.invoke(messages)
        route    = response.content.strip().lower().strip('"\'').strip()

        if route not in ["rag", "market", "macro", "multi"]:
            print(f"[Orchestrator] ⚠️  Unexpected route '{route}', fallback → rag")
            route = "rag"

        route = self._guard_macro_route(question, route)
        print(f"[Orchestrator] Route: '{route}'")
        return {**state, "route": route}

    def _rag_node(self, state: AgentState) -> AgentState:
        """
        Runs the RAG Agent.
        Before proceeding, it verifies that the ticker in the question is
        indexed — if not, it starts the on-demand download. The resolved
        tickers are then used to filter retrieval to the right company(ies).
        """
        history = state.get("history", "")
        tickers = self._ensure_tickers_indexed(state["question"], history)
        rag_answer = self.rag_agent.run(state["question"], tickers=tickers, history=history)
        return {**state, "rag_answer": rag_answer}

    def _market_node(self, state: AgentState) -> AgentState:
        """Runs the Market Agent (yfinance)."""
        market_answer = self.market_agent.run(state["question"], history=state.get("history", ""))
        return {**state, "market_answer": market_answer}

    def _macro_node(self, state: AgentState) -> AgentState:
        """Runs the Macro Agent (FRED API)."""
        macro_answer = self.macro_agent.run(state["question"], history=state.get("history", ""))
        return {**state, "macro_answer": macro_answer}

    def _multi_node(self, state: AgentState) -> AgentState:
        """
        Hybrid route: runs a COMBINATION of the specialized agents.

        Unlike the single-agent routes, this node first asks the LLM which
        subset of {RAG, Market, Macro} the question needs — any two of them or
        all three — then runs exactly those agents and lets the synthesizer
        merge their outputs. RAG retrieval is ticker-filtered just like in
        _rag_node, and on-demand indexing runs before it.
        """
        question = state["question"]
        history  = state.get("history", "")

        agents = self._select_multi_agents(question, history)
        print(f"[Orchestrator] MULTI mode — running agents: {agents}")

        updates: AgentState = {**state}

        if "rag" in agents:
            tickers = self._ensure_tickers_indexed(question, history)
            updates["rag_answer"] = self.rag_agent.run(
                question, tickers=tickers, history=history
            )
        if "market" in agents:
            updates["market_answer"] = self.market_agent.run(question, history=history)
        if "macro" in agents:
            updates["macro_answer"] = self.macro_agent.run(question, history=history)

        return updates

    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """
        Synthesizes the agents' outputs into a single coherent final answer.
        Handles all cases: single agent or multi-agent combination.
        """
        rag    = state.get("rag_answer")
        market = state.get("market_answer")
        macro  = state.get("macro_answer")

        # Single-agent case — pass the answer through directly
        active = [a for a in [rag, market, macro] if a and a.answer]
        if len(active) == 1:
            return {**state, "final_answer": active[0].answer}

        # Multi-agent case — synthesize with the LLM
        if len(active) > 1:
            parts = []
            if rag    and rag.answer:
                parts.append(f"[RAG Agent — SEC Filings]\n{rag.answer}")
            if market and market.answer:
                parts.append(f"[Market Agent — {market.ticker}]\n{market.answer}")
            if macro  and macro.answer:
                parts.append(f"[Macro Agent — FRED]\n{macro.answer}")

            combined = "\n\n---\n\n".join(parts)
            history  = state.get("history", "")
            messages = [SystemMessage(content=SYNTHESIZER_PROMPT)]
            if history:
                messages.append(SystemMessage(
                    content=f"Conversation so far (keep the answer consistent "
                            f"with it and resolve any follow-up references):\n{history}"
                ))
            messages.append(HumanMessage(
                content=f"Answers from specialized agents:\n\n{combined}"
                        f"\n\nOriginal question: {state['question']}"
            ))
            response = self.llm.invoke(messages)
            return {**state, "final_answer": response.content}

        # No agent responded
        return {**state, "final_answer": "I could not find any relevant information."}

    # ── CONDITIONAL ROUTING ─────────────────────────────────────────────────────

    def _route_condition(self, state: AgentState) -> str:
        return state["route"]

    # ── GRAPH CONSTRUCTION ──────────────────────────────────────────────────────

    def _build_graph(self):
        graph = StateGraph(AgentState)

        graph.add_node("router",      self._router_node)
        graph.add_node("rag",         self._rag_node)
        graph.add_node("market",      self._market_node)
        graph.add_node("macro",       self._macro_node)
        graph.add_node("multi",       self._multi_node)
        graph.add_node("synthesizer", self._synthesizer_node)

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router",
            self._route_condition,
            {
                "rag":    "rag",
                "market": "market",
                "macro":  "macro",
                "multi":  "multi",
            }
        )

        graph.add_edge("rag",         "synthesizer")
        graph.add_edge("market",      "synthesizer")
        graph.add_edge("macro",       "synthesizer")
        graph.add_edge("multi",       "synthesizer")
        graph.add_edge("synthesizer", END)

        return graph.compile()

    # ── PUBLIC INTERFACE ────────────────────────────────────────────────────────

    def run(self, question: str) -> str:
        # Inject the recent conversation history so every node can resolve
        # follow-up references against it.
        history = self.memory.as_context()

        initial_state: AgentState = {
            "question":      question,
            "history":       history,
            "route":         "",
            "rag_answer":    None,
            "market_answer": None,
            "macro_answer":  None,
            "final_answer":  ""
        }
        final_state = self.graph.invoke(initial_state)
        answer = final_state["final_answer"]

        # Persist this turn so the next question has context.
        self.memory.add_turn(question, answer, final_state.get("route", ""))
        return answer

    def run_traced(self, question: str, use_memory: bool = False) -> dict:
        """
        Evaluation entry point. Runs ONE question and returns the full trace.

        By default conversational memory is bypassed (history="") so single-turn
        evaluation questions stay independent. With use_memory=True the recent
        history is injected AND this turn is appended afterwards, so the method
        can drive a multi-turn dialogue (the memory-ablation experiment).

        Trace fields: route, agents, answer, contexts, rag_contexts,
        market_summary, macro_summary, and `tickers` (the tickers the system
        actually resolved and used — used for reference-resolution scoring).
        """
        history = self.memory.as_context() if use_memory else ""
        initial_state: AgentState = {
            "question":      question,
            "history":       history,
            "route":         "",
            "rag_answer":    None,
            "market_answer": None,
            "macro_answer":  None,
            "final_answer":  ""
        }
        state = self.graph.invoke(initial_state)

        rag    = state.get("rag_answer")
        market = state.get("market_answer")
        macro  = state.get("macro_answer")

        rag_contexts = [c.text for c in rag.sources] if (rag and rag.sources) else []
        contexts: list[str] = list(rag_contexts)
        if market and market.data_summary:
            contexts.append(market.data_summary)
        if macro and macro.data_summary:
            contexts.append(macro.data_summary)

        agents = [
            name for name, val in (("rag", rag), ("market", market), ("macro", macro))
            if val and getattr(val, "answer", "")
        ]

        answer = state.get("final_answer", "")
        if use_memory:
            self.memory.add_turn(question, answer, state.get("route", ""))

        # Tickers the system actually resolved and used (for reference resolution).
        tickers: set[str] = set()
        if market and getattr(market, "ticker", "") and market.ticker != "UNKNOWN":
            tickers.add(market.ticker)
        if rag and rag.sources:
            tickers.update(
                c.ticker for c in rag.sources
                if getattr(c, "ticker", "") and c.ticker not in ("unknown", "")
            )

        return {
            "question":       question,
            "route":          state.get("route", ""),
            "agents":         agents,
            "answer":         answer,
            "contexts":       contexts,
            "rag_contexts":   rag_contexts,
            "market_summary": market.data_summary if market else "",
            "macro_summary":  macro.data_summary if macro else "",
            "tickers":        sorted(tickers),
        }

    @property
    def vector_store_count(self) -> int:
        return self.vector_store.count()