# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Multi-turn dialogue set for the conversational-memory ablation.

Each dialogue opens with an EXPLICIT company question and continues with
FOLLOW-UP turns that refer to that company only through a reference ("its",
"it") — never by name. Resolving such a follow-up therefore requires the
conversational memory: with memory ON the system should recover the remembered
ticker; with memory OFF the reference cannot be resolved.

Per turn:
    expected_tickers : the ticker(s) the turn is about — for follow-ups this is
                       the referent that must be recovered from memory.
    is_followup      : True if the turn depends on the conversation history.

Metric: reference-resolution accuracy over the follow-up turns = fraction where
the expected ticker appears among the tickers the system actually resolved
(field `tickers` in the trace). Compared between memory ON and memory OFF.
"""

DIALOGUES: list[dict] = [
    {"id": "d1", "turns": [
        {"id": "d1_t1", "text": "What is Apple's business model?",
         "expected_tickers": ["AAPL"], "is_followup": False},
        {"id": "d1_t2", "text": "And what is its current stock price?",
         "expected_tickers": ["AAPL"], "is_followup": True},
        {"id": "d1_t3", "text": "How has it performed over the past year?",
         "expected_tickers": ["AAPL"], "is_followup": True},
    ]},
    {"id": "d2", "turns": [
        {"id": "d2_t1", "text": "What are Tesla's main risk factors?",
         "expected_tickers": ["TSLA"], "is_followup": False},
        {"id": "d2_t2", "text": "What is its current P/E ratio?",
         "expected_tickers": ["TSLA"], "is_followup": True},
        {"id": "d2_t3", "text": "And its market capitalization?",
         "expected_tickers": ["TSLA"], "is_followup": True},
    ]},
    {"id": "d3", "turns": [
        {"id": "d3_t1", "text": "What products does Nvidia sell?",
         "expected_tickers": ["NVDA"], "is_followup": False},
        {"id": "d3_t2", "text": "What is its market capitalization?",
         "expected_tickers": ["NVDA"], "is_followup": True},
        {"id": "d3_t3", "text": "How has its stock done this year?",
         "expected_tickers": ["NVDA"], "is_followup": True},
    ]},
    {"id": "d4", "turns": [
        {"id": "d4_t1", "text": "What are Amazon's business segments?",
         "expected_tickers": ["AMZN"], "is_followup": False},
        {"id": "d4_t2", "text": "What is its current share price?",
         "expected_tickers": ["AMZN"], "is_followup": True},
    ]},
    {"id": "d5", "turns": [
        {"id": "d5_t1", "text": "How does Meta make money?",
         "expected_tickers": ["META"], "is_followup": False},
        {"id": "d5_t2", "text": "What is its P/E ratio?",
         "expected_tickers": ["META"], "is_followup": True},
        {"id": "d5_t3", "text": "And its 52-week high?",
         "expected_tickers": ["META"], "is_followup": True},
    ]},
    {"id": "d6", "turns": [
        {"id": "d6_t1", "text": "How does Alphabet make money?",
         "expected_tickers": ["GOOGL"], "is_followup": False},
        {"id": "d6_t2", "text": "What is its current share price?",
         "expected_tickers": ["GOOGL"], "is_followup": True},
        {"id": "d6_t3", "text": "How has it done over the last year?",
         "expected_tickers": ["GOOGL"], "is_followup": True},
    ]},
    {"id": "d7", "turns": [
        {"id": "d7_t1", "text": "What is Microsoft's cloud strategy?",
         "expected_tickers": ["MSFT"], "is_followup": False},
        {"id": "d7_t2", "text": "What is its current valuation?",
         "expected_tickers": ["MSFT"], "is_followup": True},
    ]},
    {"id": "d8", "turns": [
        {"id": "d8_t1", "text": "What is Tesla's energy generation and storage business?",
         "expected_tickers": ["TSLA"], "is_followup": False},
        {"id": "d8_t2", "text": "What's its current stock price?",
         "expected_tickers": ["TSLA"], "is_followup": True},
        {"id": "d8_t3", "text": "Is it overvalued given current interest rates?",
         "expected_tickers": ["TSLA"], "is_followup": True},
    ]},
]


def counts() -> dict:
    turns = [t for d in DIALOGUES for t in d["turns"]]
    return {
        "dialogues": len(DIALOGUES),
        "turns": len(turns),
        "followup_turns": sum(t["is_followup"] for t in turns),
    }


if __name__ == "__main__":
    c = counts()
    print(f"Dialogues: {c['dialogues']}  |  turns: {c['turns']}  |  "
          f"follow-up turns: {c['followup_turns']}")
    ids = [t["id"] for d in DIALOGUES for t in d["turns"]]
    print(f"Unique turn ids: {len(set(ids)) == len(ids)}")
