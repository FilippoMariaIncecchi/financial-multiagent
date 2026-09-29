# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
import os
from dataclasses import dataclass

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_NUM_CTX, MACRO_HISTORY_MONTHS

load_dotenv()


@dataclass
class MacroAnswer:
    """Structured output of the Macro Agent."""
    answer:         str
    series_fetched: list[str]
    data_summary:   str


# Maps keyword → (FRED series ID, human-readable description)
# NOTE: for index series (e.g. CPI), the description reflects the value we
# actually report — the year-over-year inflation rate, NOT the raw index level.
FRED_SERIES = {
    "gdp":          ("GDP",      "US Gross Domestic Product (Billions USD, quarterly)"),
    "inflation":    ("CPIAUCSL", "US CPI inflation rate (year-over-year %, monthly)"),
    "cpi":          ("CPIAUCSL", "US CPI inflation rate (year-over-year %, monthly)"),
    "interest":     ("FEDFUNDS", "Federal Funds Effective Rate (%, monthly)"),
    "fed":          ("FEDFUNDS", "Federal Funds Effective Rate (%, monthly)"),
    "unemployment": ("UNRATE",   "US Unemployment Rate (%, monthly)"),
    "jobs":         ("UNRATE",   "US Unemployment Rate (%, monthly)"),
}

# Series that are published as INDEX LEVELS (base period = 100) rather than
# rates. These must be converted to a year-over-year % change before being
# shown to the LLM — otherwise the model has no way to derive inflation from a
# single index value and will hallucinate (e.g. treating "100" as last year's CPI).
INDEX_SERIES = {"CPIAUCSL"}

# Series published QUARTERLY rather than monthly. The reporting window is set in
# months (config.MACRO_HISTORY_MONTHS), so these are scaled down by ~3 to cover
# the same time span as the monthly series (e.g. 24 months → 8 quarters).
QUARTERLY_SERIES = {"GDP"}


SERIES_SELECTION_PROMPT = """You are a macroeconomic data assistant.
From the user's question, identify which economic indicators are needed.

Available keywords (respond ONLY with these, comma-separated):
- gdp          → US Gross Domestic Product
- inflation    → US Consumer Price Index / inflation rate
- interest     → Federal Funds Rate / interest rates
- unemployment → US Unemployment Rate

Examples:
  "What is the current inflation?" → inflation
  "How are interest rates affecting the economy?" → interest, gdp
  "Give me a macro overview" → gdp, inflation, interest, unemployment

Respond with ONLY comma-separated keywords. No other text."""


MACRO_ANALYSIS_PROMPT = """You are a macroeconomic analyst specializing in US markets.
Answer the user's question using ONLY the economic data provided below.

IMPORTANT: The values provided are already fully computed figures in their stated
units (e.g. inflation is already a year-over-year percentage, interest and
unemployment are already percentages). Do NOT transform, rebase, or recompute them.
Report and interpret them exactly as given.

Use ONLY the dates explicitly listed in the data. Do NOT forecast, predict,
extrapolate, or invent values for any month or period that is not present.
Do NOT state or guess the current date — the most recent period in the data is
the latest information available, not necessarily today's date.
If the data covers fewer periods than the question asks for, say so plainly
and analyze only what is given.

Each series includes a precomputed SUMMARY block (Latest, Change vs previous
period, Change vs one year earlier, Trend). When you state the latest value, a
change, or the direction of the trend, QUOTE the SUMMARY fields verbatim. Do NOT
infer which period is the latest, do NOT compute changes yourself, and do NOT
decide the trend yourself — the SUMMARY already did this for you.

Answer the question directly and concisely, quoting the SUMMARY figures.
Keep any interpretation brief and tied to the data. Do NOT add investment
recommendations, disclaimers, or generic "businesses/investors may…" commentary
that is not grounded in the provided figures."""


class MacroAgent:
    """
    Macro Agent: retrieves aggregated economic indicators from FRED
    (Federal Reserve Economic Data) via fredapi.
    Main series: GDP, CPI (inflation), FEDFUNDS, UNRATE.

    Index-level series (e.g. CPIAUCSL) are converted to year-over-year
    percentage change in code before being passed to the LLM, so the model
    only interprets finished rates and never performs the arithmetic itself.
    """

    def __init__(self):
        self.llm     = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE, num_ctx=LLM_NUM_CTX)
        api_key      = os.getenv("FRED_API_KEY", "")
        self.fred    = Fred(api_key=api_key) if api_key else None

    # ── SERIES SELECTION ───────────────────────────────────────────────────────

    def _select_series(self, question: str, history: str = "") -> list[tuple[str, str]]:
        """
        Uses the LLM to determine which FRED series are relevant.
        Returns a list of (series_id, description).
        The conversation history (when present) lets the agent resolve
        follow-up references such as "and unemployment?".
        """
        messages = [SystemMessage(content=SERIES_SELECTION_PROMPT)]
        if history:
            messages.append(SystemMessage(
                content=f"Conversation so far (use it to resolve follow-up "
                        f"references):\n{history}"
            ))
        messages.append(HumanMessage(content=question))
        response = self.llm.invoke(messages)
        keywords = [k.strip().lower() for k in response.content.split(",")]

        selected  = []
        seen_ids  = set()
        for kw in keywords:
            if kw in FRED_SERIES:
                series_id, description = FRED_SERIES[kw]
                if series_id not in seen_ids:
                    selected.append((series_id, description))
                    seen_ids.add(series_id)

        # Fallback: if the LLM did not return anything valid
        if not selected:
            print("[MacroAgent] ⚠️  No series identified, using GDP + inflation")
            selected = [
                ("GDP",      FRED_SERIES["gdp"][1]),
                ("CPIAUCSL", FRED_SERIES["inflation"][1]),
            ]

        return selected

    # ── FETCH & FORMAT ────────────────────────────────────────────────────────

    @staticmethod
    def _periods_for(series_id: str) -> int:
        """
        Number of periods to fetch so every series covers the same time window
        (config.MACRO_HISTORY_MONTHS), regardless of its publication frequency:
        monthly series report one point per month; quarterly series (GDP) report
        one per quarter, so they are scaled down by ~3.
        """
        if series_id in QUARTERLY_SERIES:
            return max(4, MACRO_HISTORY_MONTHS // 3)   # e.g. 24 months → 8 quarters
        return MACRO_HISTORY_MONTHS

    def _fetch_series(self, series_id: str, periods: int | None = None) -> pd.Series:
        """
        Retrieves the last N periods of a FRED series. When `periods` is not
        given, it is derived from MACRO_HISTORY_MONTHS and the series frequency.

        For INDEX series (e.g. CPIAUCSL), an index level cannot be interpreted
        as inflation on its own. We pull 12 extra months of history, compute the
        year-over-year % change, and return that rate instead of the raw index:

            inflation_YoY(t) = (CPI[t] / CPI[t-12] - 1) * 100

        For all other series (rates/levels), the raw values are returned as-is.
        """
        if periods is None:
            periods = self._periods_for(series_id)

        data = self.fred.get_series(series_id).dropna()

        if series_id in INDEX_SERIES:
            # Need 12 extra months so the first reported period has a valid
            # year-earlier comparison point.
            data = data.tail(periods + 12)
            yoy  = (data / data.shift(12) - 1.0) * 100.0
            return yoy.dropna().tail(periods)

        return data.tail(periods)

    def _format_series(self, series: pd.Series, description: str,
                       series_id: str = "") -> str:
        """
        Formats a time series into LLM-readable text.

        Leads with a SUMMARY block whose Latest value, period-over-period change,
        year-over-year change and trend direction are COMPUTED HERE, in code, so
        the (small) language model never has to identify the latest period or do
        arithmetic over the rows — it only quotes the finished figures. The full
        rows follow for detail; ordering is oldest → newest.
        """
        if series.empty:
            return f"{description}:\n  (no data available)"

        series  = series.sort_index()
        ppy     = 4 if series_id in QUARTERLY_SERIES else 12   # periods per year
        first_d = series.index[0]
        last_d  = series.index[-1]
        latest  = float(series.iloc[-1])
        first_v = float(series.iloc[0])

        summary = ["  SUMMARY (computed — quote these; do not recompute):",
                   f"    Latest: {last_d.strftime('%Y-%m')} = {latest:.2f}"]
        if len(series) >= 2:
            prev = float(series.iloc[-2])
            summary.append(
                f"    Change vs previous period "
                f"({series.index[-2].strftime('%Y-%m')} = {prev:.2f}): {latest - prev:+.2f}")
        if len(series) > ppy:
            yr = float(series.iloc[-(ppy + 1)])
            yd = series.index[-(ppy + 1)].strftime('%Y-%m')
            summary.append(
                f"    Change vs one year earlier ({yd} = {yr:.2f}): {latest - yr:+.2f}")
        diff  = latest - first_v
        trend = "rising" if diff > 0.05 else "falling" if diff < -0.05 else "roughly flat"
        summary.append(
            f"    Trend over {len(series)} periods "
            f"({first_d.strftime('%Y-%m')} {first_v:.2f} -> {last_d.strftime('%Y-%m')} "
            f"{latest:.2f}): {trend}")

        lines = [f"{description}:"]
        lines += summary
        lines.append(
            f"  Coverage: {len(series)} period(s), {first_d.strftime('%Y-%m')} to "
            f"{last_d.strftime('%Y-%m')} (most recent available, not necessarily the "
            f"current month)")
        for date, value in series.items():
            lines.append(f"  {date.strftime('%Y-%m')}: {value:.2f}")
        return "\n".join(lines)

    # ── MAIN PIPELINE ──────────────────────────────────────────────────────────

    def run(self, question: str, history: str = "") -> MacroAnswer:
        """
        Full pipeline: select series → fetch FRED → generate LLM answer.

        Args:
            question: the user's natural-language question
            history:  optional conversation history, used to resolve follow-up
                      references when selecting which series to fetch.
        """
        print(f"\n[MacroAgent] Question received: '{question}'")

        if not self.fred:
            return MacroAnswer(
                answer="FRED API key not configured. "
                       "Add it to the .env file as FRED_API_KEY=...",
                series_fetched=[],
                data_summary=""
            )

        # 1. Select relevant series with the LLM (history resolves follow-ups)
        selected = self._select_series(question, history)
        print(f"[MacroAgent] Selected series: {[s[0] for s in selected]}")

        # 2. Fetch data from FRED
        summaries   = []
        fetched_ids = []
        for series_id, description in selected:
            try:
                data = self._fetch_series(series_id)
                if data.empty:
                    print(f"[MacroAgent] ⚠️  {series_id}: not enough history to report")
                    continue
                summaries.append(self._format_series(data, description, series_id))
                fetched_ids.append(series_id)
                print(f"[MacroAgent] ✅ {series_id} ({len(data)} periods)")
            except Exception as e:
                print(f"[MacroAgent] ⚠️  Error {series_id}: {e}")

        if not summaries:
            return MacroAnswer(
                answer="I could not retrieve data from FRED. "
                       "Check the FRED API key in the .env file.",
                series_fetched=[],
                data_summary=""
            )

        data_summary = "\n\n".join(summaries)

        # 3. Generate answer with the LLM
        messages = [
            SystemMessage(content=MACRO_ANALYSIS_PROMPT),
            HumanMessage(
                content=f"Economic data from FRED:\n{data_summary}"
                        f"\n\nQuestion: {question}"
            )
        ]
        response = self.llm.invoke(messages)

        return MacroAnswer(
            answer=response.content,
            series_fetched=fetched_ids,
            data_summary=data_summary
        )