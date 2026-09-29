# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Shared ticker-resolution utilities.

Both the Orchestrator (RAG path, multi-ticker) and the Market agent
(single ticker) resolve the company a question is about through this single
module, so they can never contradict each other.

Strategy: resolve deterministically from the question text against the known
company map (:data:`config.COMPANY_ALIASES`) first — precise and immune to
small-model failures — and use the LLM only as a fallback, with defensive
parsing that rejects filler tokens (e.g. "Okay") a small model may emit.
"""

import re

from config import COMPANY_ALIASES, SEC_COMPANIES

# Every ticker the system knows by name — used to validate LLM-extracted tokens.
KNOWN_TICKERS: set[str] = set(COMPANY_ALIASES.values()) | set(SEC_COMPANIES)

# Tokens a small LLM tends to emit as filler/preamble that look like tickers
# (1–5 uppercase letters) but are not. Without this guard, "Okay" was being
# treated as the ticker "OKAY" and triggering a doomed SEC download.
_NON_TICKER_WORDS: set[str] = {
    "OKAY", "OK", "SURE", "HERE", "THERE", "YES", "NO", "NONE", "NULL", "NAN",
    "NA", "THE", "AND", "OR", "FOR", "ANY", "ALL", "ARE", "IS", "IT", "ITS",
    "TO", "OF", "ON", "IN", "AT", "AN", "AS", "BY", "WE", "ANSWER", "TICKER",
    "STOCK", "PRICE", "NOTE", "HELLO", "HI", "LOOK", "SEE", "GET", "US", "USA",
    "GDP", "CPI", "FED", "PE", "EPS", "CEO", "CFO", "AI", "ETF", "SEC",
}

# Matches a run of 1–5 uppercase letters bounded by non-letters (a ticker shape).
_TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{1,5}\b")


def scan_known_tickers(question: str) -> list[str]:
    """
    Deterministically resolves tickers by scanning the question text against
    the known company map (config.COMPANY_ALIASES) plus any verbatim ticker
    symbols. This is precise and immune to LLM failures — it never invents a
    ticker and never confuses one company for another.

    Returns tickers in the order they appear, de-duplicated.
    """
    q_lower = question.lower()
    found: list[str] = []

    # 1. Company names / brands (whole-word / whole-phrase match).
    for alias, ticker in COMPANY_ALIASES.items():
        pattern = rf"(?<![a-z]){re.escape(alias)}(?![a-z])"
        if re.search(pattern, q_lower) and ticker not in found:
            found.append(ticker)

    # 2. Verbatim ticker symbols (e.g. "AAPL"), accepted only if known —
    #    avoids treating ordinary uppercase words/acronyms as tickers.
    for token in _TICKER_TOKEN_RE.findall(question):
        if token in KNOWN_TICKERS and token not in found:
            found.append(token)

    return found


def parse_llm_tickers(content: str) -> list[str]:
    """
    Defensively parses an LLM's ticker output. Extracts ticker-shaped tokens,
    drops UNKNOWN and known filler words ("OKAY", "HERE", …), and prefers known
    tickers when any are present so a stray word cannot leak through alongside a
    real ticker.
    """
    raw = content.strip().upper().strip('"\'`')
    if not raw or raw.startswith("UNKNOWN"):
        return []

    candidates: list[str] = []
    for token in _TICKER_TOKEN_RE.findall(raw):
        if token == "UNKNOWN" or token in _NON_TICKER_WORDS:
            continue
        if token not in candidates:
            candidates.append(token)

    # If the model named any company we recognise, trust only those.
    known = [t for t in candidates if t in KNOWN_TICKERS]
    return known if known else candidates
