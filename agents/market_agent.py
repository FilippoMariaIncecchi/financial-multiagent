# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
import math
from dataclasses import dataclass

import yfinance as yf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_NUM_CTX
from tickers import parse_llm_tickers, scan_known_tickers

load_dotenv()


def _finite(x) -> bool:
    """True only for a real, finite number — filters out None and NaN."""
    try:
        return x is not None and not math.isnan(float(x))
    except (TypeError, ValueError):
        return False


@dataclass
class MarketAnswer:
    """Structured output of the Market Agent."""
    answer:       str
    ticker:       str
    data_summary: str


TICKER_EXTRACTION_PROMPT = """Extract the stock ticker symbol from the user's question.
The question may be in any language (English, Italian, French, Spanish, etc.).

Rules:
- Return ONLY the ticker symbol. No explanations, no punctuation.
- Use standard US ticker format.
- Map company names to tickers (multilingual — same name works across languages):
    Apple / apple → AAPL       Microsoft / microsoft → MSFT
    Google / Alphabet → GOOGL  Amazon / amazon → AMZN
    Tesla / tesla → TSLA       Meta / Facebook → META
    Palantir / palantir → PLTR Nvidia / nvidia → NVDA
    JPMorgan / JP Morgan → JPM Netflix / netflix → NFLX
    Salesforce → CRM           Adobe → ADBE
    AMD → AMD                  Intel → INTC
- If the ticker is written directly (e.g. AAPL, PLTR), return it as-is.
- If you cannot identify a specific publicly traded company, return: UNKNOWN

Respond with ONE word only (the ticker or UNKNOWN)."""


MARKET_ANALYSIS_PROMPT = """You are a financial market analyst. Answer the
user's question directly and concisely using ONLY the market data provided.

Rules:
- Lead with the specific figure(s) the question asks for — the exact number,
  percentage and date from the data.
- Use ONLY the figures given. Do NOT add outside facts, rankings ("largest
  company in the world"), forecasts, investment recommendations or disclaimers.
- If a value shows N/A, report it as not available rather than guessing.
- Keep any interpretation brief and tied directly to the provided figures."""


class MarketAgent:
    """
    Market Agent: retrieves historical quantitative data
    via yfinance (prices, returns, financial metrics).
    No API key required — public Yahoo Finance data.
    """

    def __init__(self):
        self.llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, num_ctx=LLM_NUM_CTX)

    def _extract_ticker(self, question: str, history: str = "") -> str:
        """
        Resolves the ticker the question is about, deterministic-first
        (shared with the Orchestrator via the `tickers` module):

          1. Scan the question text against the known company map. Precise and
             immune to small-model errors — this is the single source of truth,
             so the Market agent can never disagree with the RAG path.
          2. LLM fallback ONLY when the scan finds nothing — an unknown company
             or a follow-up reference resolved via the conversation history.

        Returns an uppercase ticker, or "UNKNOWN" if none is identified.
        """
        # ── Stage 1: deterministic resolution from the question text ──────────
        matched = scan_known_tickers(question)
        if matched:
            print(f"[MarketAgent] Ticker matched from question: {matched[0]}")
            return matched[0]

        # ── Stage 2: LLM fallback (unknown company or follow-up reference) ────
        messages = [SystemMessage(content=TICKER_EXTRACTION_PROMPT)]
        if history:
            messages.append(SystemMessage(
                content=f"Conversation so far (use it to resolve references "
                        f"like 'it' or 'that company'):\n{history}"
            ))
        messages.append(HumanMessage(content=question))
        response = self.llm.invoke(messages)

        tickers = parse_llm_tickers(response.content)
        return tickers[0] if tickers else "UNKNOWN"

    def _fetch_market_data(self, ticker: str) -> dict:
        """Retrieves market data from yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info  = stock.info
            hist  = stock.history(period="1y")

            if hist.empty:
                return {"error": f"No data found for {ticker}"}

            current_price = hist["Close"].iloc[-1]
            price_1y_ago  = hist["Close"].iloc[0]
            return_1y     = (((current_price - price_1y_ago) / price_1y_ago) * 100
                             if _finite(price_1y_ago) and price_1y_ago else None)

            hist_3m   = stock.history(period="3mo")
            price_3m  = hist_3m["Close"].iloc[0] if not hist_3m.empty else None
            return_3m = (((current_price - price_3m) / price_3m) * 100
                         if _finite(price_3m) and price_3m else None)

            return {
                "ticker":         ticker,
                "company_name":   info.get("longName", ticker),
                "current_price":  round(current_price, 2),
                "currency":       info.get("currency", "USD"),
                "market_cap":     info.get("marketCap"),
                "pe_ratio":       info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "52w_high":       info.get("fiftyTwoWeekHigh"),
                "52w_low":        info.get("fiftyTwoWeekLow"),
                "return_1y_pct":  round(return_1y, 2) if _finite(return_1y) else None,
                "return_3m_pct":  round(return_3m, 2) if _finite(return_3m) else None,
                "avg_volume":     info.get("averageVolume"),
                "data_date":      str(hist.index[-1].date()),
            }

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

    def _format_data_summary(self, data: dict) -> str:
        """Formats the market data into LLM-readable text."""
        if "error" in data:
            return f"Error retrieving data: {data['error']}"

        mc  = data.get("market_cap")
        mc_str = f"${mc/1e9:.1f}B" if mc else "N/A"

        dy = data.get("dividend_yield")
        # yfinance already returns the value as a decimal (e.g. 0.005 = 0.5%)
        # sanity check: if > 0.20 (20%) it is almost certainly a provider error
        if dy and 0 < dy < 0.20:
            dy_str = f"{dy*100:.2f}%"
        else:
            dy_str = "N/A"

        pe = data.get("pe_ratio")
        pe_str = f"{pe:.1f}x" if pe else "N/A"

        r3m = data.get("return_3m_pct")
        r3m_str = f"{r3m:+.2f}%" if r3m is not None else "N/A"

        r1y = data.get("return_1y_pct")
        r1y_str = f"{r1y:+.2f}%" if r1y is not None else "N/A"

        vol = data.get("avg_volume")
        vol_str = f"{vol:,}" if vol else "N/A"

        return (
            f"Market Data — {data['company_name']} ({data['ticker']}) "
            f"as of {data['data_date']}\n"
            f"  Current Price:    {data['currency']} {data['current_price']}\n"
            f"  1-Year Return:    {r1y_str}\n"
            f"  3-Month Return:   {r3m_str}\n"
            f"  52-Week High:     {data.get('52w_high', 'N/A')}\n"
            f"  52-Week Low:      {data.get('52w_low', 'N/A')}\n"
            f"  Market Cap:       {mc_str}\n"
            f"  P/E Ratio:        {pe_str}\n"
            f"  Dividend Yield:   {dy_str}\n"
            f"  Avg Daily Volume: {vol_str} shares"
        )

    def run(self, question: str, history: str = "") -> MarketAnswer:
        """
        Full pipeline: extract ticker → fetch yfinance → generate LLM answer.

        Args:
            question: the user's natural-language question
            history:  optional conversation history, used to resolve follow-up
                      references when extracting the ticker.
        """
        print(f"\n[MarketAgent] Question received: '{question}'")

        # 1. Extract ticker (history helps resolve follow-up references)
        ticker = self._extract_ticker(question, history)
        print(f"[MarketAgent] Extracted ticker: {ticker}")

        if ticker == "UNKNOWN":
            return MarketAnswer(
                answer="I did not identify a company in the question. "
                       "Please specify the name or the ticker (e.g. Apple, AAPL).",
                ticker="UNKNOWN",
                data_summary=""
            )

        # 2. Fetch data from Yahoo Finance
        data         = self._fetch_market_data(ticker)
        data_summary = self._format_data_summary(data)
        print(f"[MarketAgent] Data retrieved:\n{data_summary}")

        # 3. Generate answer with LLM
        messages = [
            SystemMessage(content=MARKET_ANALYSIS_PROMPT),
            HumanMessage(content=f"Market data:\n{data_summary}\n\nQuestion: {question}")
        ]
        response = self.llm.invoke(messages)

        return MarketAnswer(
            answer=response.content,
            ticker=ticker,
            data_summary=data_summary
        )