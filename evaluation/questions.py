# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Phase 3 evaluation dataset — 100 questions.

Distribution (matches the thesis design):
    20  rag                 (single company, 10-K content)
    20  market              (single company, stock figures)
    20  macro               (economy-wide indicators)
    10  multi_rag_market    (company business + its stock)
    10  multi_rag_macro     (company business + macro context)
    10  multi_market_macro  (a stock's figures + macro context, no business)
    10  multi_all           (business + stock + macro)

All questions are in English and use the 9 already-indexed tickers
(AAPL, MSFT, GOOGL, AMZN, TSLA, NVDA, META, INTC, PLTR) so the evaluation does
not trigger on-demand SEC downloads. Each item carries the EXPECTED route and
agent set, so the harness can also report routing/selection accuracy.
"""

# expected_route ∈ {rag, market, macro, multi}
# expected_agents ⊆ {rag, market, macro}

QUESTIONS: list[dict] = [

    # ── RAG (20) — company business / strategy / products / risks / segments ──
    {"id": "rag_01", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What is Apple's business model?"},
    {"id": "rag_02", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What are the main risk factors Tesla discloses in its filings?"},
    {"id": "rag_03", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "How does Meta generate most of its revenue?"},
    {"id": "rag_04", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What are Amazon's primary business segments?"},
    {"id": "rag_05", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "Describe Microsoft's cloud computing strategy."},
    {"id": "rag_06", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What products does Nvidia design and sell?"},
    {"id": "rag_07", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "How does Alphabet (Google) make money?"},
    {"id": "rag_08", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What are Intel's main business segments?"},
    {"id": "rag_09", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What is Palantir's core business?"},
    {"id": "rag_10", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What competitive risks does Apple identify in its 10-K?"},
    {"id": "rag_11", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What is Microsoft's stated mission and overall strategy?"},
    {"id": "rag_12", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "How does Amazon describe its fulfillment and logistics operations?"},
    {"id": "rag_13", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What does Tesla say about its energy generation and storage business?"},
    {"id": "rag_14", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What supply-chain risks does Nvidia highlight?"},
    {"id": "rag_15", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "How does Meta describe its Reality Labs segment?"},
    {"id": "rag_16", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What regulatory risks does Alphabet disclose?"},
    {"id": "rag_17", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "How does Palantir describe its government versus commercial customers?"},
    {"id": "rag_18", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What does Intel say about its foundry strategy?"},
    {"id": "rag_19", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "What are Apple's main product categories and services?"},
    {"id": "rag_20", "category": "rag", "expected_route": "rag", "expected_agents": ["rag"],
     "question": "How does Microsoft describe competition in its industry?"},

    # ── MARKET (20) — single company, stock figures only ─────────────────────
    {"id": "market_01", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Apple's current stock price?"},
    {"id": "market_02", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What was Tesla's 1-year stock return?"},
    {"id": "market_03", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Nvidia's market capitalization?"},
    {"id": "market_04", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Microsoft's P/E ratio?"},
    {"id": "market_05", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What are Amazon's 52-week high and low prices?"},
    {"id": "market_06", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "Does Meta pay a dividend, and what is its yield?"},
    {"id": "market_07", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Alphabet's current share price?"},
    {"id": "market_08", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Intel's average daily trading volume?"},
    {"id": "market_09", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "How has Palantir's stock performed over the last 3 months?"},
    {"id": "market_10", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Tesla's current P/E ratio?"},
    {"id": "market_11", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Apple's market cap right now?"},
    {"id": "market_12", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What was Nvidia's 1-year stock return?"},
    {"id": "market_13", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Microsoft's current stock price?"},
    {"id": "market_14", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Amazon's trailing P/E ratio?"},
    {"id": "market_15", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Meta's 52-week high?"},
    {"id": "market_16", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "How has Google's share price moved over the past year?"},
    {"id": "market_17", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Tesla's market capitalization?"},
    {"id": "market_18", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Nvidia's current share price?"},
    {"id": "market_19", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Apple's dividend yield?"},
    {"id": "market_20", "category": "market", "expected_route": "market", "expected_agents": ["market"],
     "question": "What is Palantir's current market cap?"},

    # ── MACRO (20) — economy-wide, no company, no stock term ──────────────────
    {"id": "macro_01", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "What is the current US inflation rate?"},
    {"id": "macro_02", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "What is the latest US GDP figure?"},
    {"id": "macro_03", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "What is the current Federal Funds rate?"},
    {"id": "macro_04", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "What is the US unemployment rate?"},
    {"id": "macro_05", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "How has US inflation changed over the past two years?"},
    {"id": "macro_06", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "What is the recent trend in US GDP?"},
    {"id": "macro_07", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "How have US interest rates moved recently?"},
    {"id": "macro_08", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "How has US unemployment evolved over the last two years?"},
    {"id": "macro_09", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "Give me an overview of the US macroeconomic situation."},
    {"id": "macro_10", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "What is the relationship between recent US inflation and unemployment?"},
    {"id": "macro_11", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "How has the Federal Reserve's policy rate trended recently?"},
    {"id": "macro_12", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "Is US inflation rising or falling lately?"},
    {"id": "macro_13", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "What does recent GDP growth indicate about the US economy?"},
    {"id": "macro_14", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "How tight is the US labor market based on the unemployment rate?"},
    {"id": "macro_15", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "Summarize recent trends in US inflation and interest rates."},
    {"id": "macro_16", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "What is the most recent US GDP reading and how does it compare to prior quarters?"},
    {"id": "macro_17", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "Has the Fed been raising or cutting rates recently?"},
    {"id": "macro_18", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "What is the current level and trend of US unemployment?"},
    {"id": "macro_19", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "How do current US interest rates compare to a year ago?"},
    {"id": "macro_20", "category": "macro", "expected_route": "macro", "expected_agents": ["macro"],
     "question": "Provide a macro overview covering growth, inflation, rates and jobs."},

    # ── MULTI: rag + market (10) — business + the company's stock ─────────────
    {"id": "rm_01", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "How does Apple's business strategy compare with its current stock valuation?"},
    {"id": "rm_02", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "Compare Tesla's fundamentals with its recent share price performance."},
    {"id": "rm_03", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "How does Nvidia's product strategy align with its current market cap?"},
    {"id": "rm_04", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "Does Microsoft's cloud strategy justify its current P/E ratio?"},
    {"id": "rm_05", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "How do Amazon's business segments relate to its current stock performance?"},
    {"id": "rm_06", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "Is Meta's Reality Labs investment reflected in its share price?"},
    {"id": "rm_07", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "How does Alphabet's revenue model compare with its current valuation?"},
    {"id": "rm_08", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "Does Palantir's business model support its current stock price?"},
    {"id": "rm_09", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "How does Intel's foundry strategy relate to its recent stock performance?"},
    {"id": "rm_10", "category": "multi_rag_market", "expected_route": "multi", "expected_agents": ["rag", "market"],
     "question": "How does Apple's services growth compare with its 1-year stock return?"},

    # ── MULTI: rag + macro (10) — business + macro context ───────────────────
    {"id": "rmac_01", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "How resilient is Tesla's business model given current US interest rates?"},
    {"id": "rmac_02", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "How might current US inflation affect Apple's business?"},
    {"id": "rmac_03", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "How does the current interest-rate environment impact Nvidia's strategy?"},
    {"id": "rmac_04", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "Given recent US GDP trends, how is Amazon's business positioned?"},
    {"id": "rmac_05", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "How could rising interest rates affect Microsoft's cloud investments?"},
    {"id": "rmac_06", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "How does current US unemployment relate to Meta's advertising business?"},
    {"id": "rmac_07", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "How might inflation impact Alphabet's advertising revenue model?"},
    {"id": "rmac_08", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "Given the macro backdrop, how exposed is Palantir's business?"},
    {"id": "rmac_09", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "How do current interest rates affect Tesla's capital-intensive business?"},
    {"id": "rmac_10", "category": "multi_rag_macro", "expected_route": "multi", "expected_agents": ["rag", "macro"],
     "question": "How might a slowing economy affect Intel's business?"},

    # ── MULTI: market + macro (10) — a stock's figures + macro, no business ──
    {"id": "mm_01", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "Is Nvidia's current P/E ratio justified given today's interest rates?"},
    {"id": "mm_02", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "How does Apple's stock return compare with the recent inflation trend?"},
    {"id": "mm_03", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "Given the Federal Funds rate, is Tesla's valuation stretched?"},
    {"id": "mm_04", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "Does Microsoft's current valuation make sense in this interest-rate environment?"},
    {"id": "mm_05", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "How does Amazon's share price performance line up with recent GDP growth?"},
    {"id": "mm_06", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "Is Meta's market cap reasonable given the current macro backdrop?"},
    {"id": "mm_07", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "How does Google's stock return compare with current inflation?"},
    {"id": "mm_08", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "Given rising interest rates, is Nvidia's stock price overvalued?"},
    {"id": "mm_09", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "Does Palantir's valuation look stretched given current interest rates?"},
    {"id": "mm_10", "category": "multi_market_macro", "expected_route": "multi", "expected_agents": ["market", "macro"],
     "question": "How does Intel's stock performance compare with recent unemployment trends?"},

    # ── MULTI: all three (10) — business + stock + macro ─────────────────────
    {"id": "all_01", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Assess Apple's strategy, its current stock valuation, and the macro backdrop."},
    {"id": "all_02", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Evaluate Tesla's business model, share price, and the interest-rate environment."},
    {"id": "all_03", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Give a full picture of Nvidia: strategy, valuation, and macro context."},
    {"id": "all_04", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Assess Microsoft's cloud strategy, stock valuation, and current interest rates."},
    {"id": "all_05", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Evaluate Amazon's segments, stock performance, and recent GDP trends."},
    {"id": "all_06", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Analyze Meta's business, market cap, and the macro environment."},
    {"id": "all_07", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Assess Alphabet's revenue model, valuation, and the inflation backdrop."},
    {"id": "all_08", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Evaluate Palantir's business, stock price, and macro conditions."},
    {"id": "all_09", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Give a complete view of Intel: strategy, stock performance, and the economy."},
    {"id": "all_10", "category": "multi_all", "expected_route": "multi", "expected_agents": ["rag", "market", "macro"],
     "question": "Assess Tesla's strategy, valuation, and how interest rates affect it."},
]


# Expected counts per category — asserted by the harness and the test suite.
EXPECTED_COUNTS = {
    "rag":                20,
    "market":             20,
    "macro":              20,
    "multi_rag_market":   10,
    "multi_rag_macro":    10,
    "multi_market_macro": 10,
    "multi_all":          10,
}


if __name__ == "__main__":
    from collections import Counter
    counts = Counter(q["category"] for q in QUESTIONS)
    print(f"Total questions: {len(QUESTIONS)}")
    for cat, exp in EXPECTED_COUNTS.items():
        got = counts.get(cat, 0)
        print(f"  {'✅' if got == exp else '❌'} {cat:<20} {got}/{exp}")
    ids = [q["id"] for q in QUESTIONS]
    print(f"Unique ids: {len(set(ids)) == len(ids)}  |  Total matches 100: {len(QUESTIONS) == 100}")
