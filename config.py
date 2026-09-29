# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
from pathlib import Path

# ── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data" / "sec_filings"
FILINGS_DIR     = BASE_DIR / "data" / "sec-edgar-filings"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL       = "gemma3:4b"
LLM_TEMPERATURE = 0.0   # deterministic — essential for reproducible RAGAS evaluation
LLM_NUM_CTX     = 8192

# ── EMBEDDINGS ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "nomic-embed-text"

# ── CHROMADB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "sec_filings"

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────
SEC_COMPANIES   = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
SEC_FILING_TYPE = "10-K"
SEC_NUM_FILINGS = 1

# ── COMPANY → TICKER MAP ──────────────────────────────────────────────────────
# Deterministic alias → ticker lookup used to resolve the company a question is
# about WITHOUT relying on the LLM. Scanning the question text against this map
# is precise and immune to small-model failures (e.g. emitting filler like
# "Okay" or mis-identifying the company). It mirrors the mapping embedded in the
# LLM extraction prompts; the LLM is used only as a fallback for companies not
# listed here (on-demand download) or for follow-up references ("its", "that").
# Keys are lowercase; multi-word aliases are matched as whole phrases.
COMPANY_ALIASES = {
    "apple":            "AAPL",
    "microsoft":        "MSFT",
    "google":           "GOOGL",
    "alphabet":         "GOOGL",
    "amazon":           "AMZN",
    "meta":             "META",
    "facebook":         "META",
    "tesla":            "TSLA",
    "palantir":         "PLTR",
    "nvidia":           "NVDA",
    "jpmorgan":         "JPM",
    "jp morgan":        "JPM",
    "j.p. morgan":      "JPM",
    "netflix":          "NFLX",
    "salesforce":       "CRM",
    "adobe":            "ADBE",
    "amd":              "AMD",
    "intel":            "INTC",
    "goldman sachs":    "GS",
    "morgan stanley":   "MS",
    "bank of america":  "BAC",
    "qualcomm":         "QCOM",
    "uber":             "UBER",
    "airbnb":           "ABNB",
    "spotify":          "SPOT",
    "shopify":          "SHOP",
}

# ── RAG / CHUNKING ────────────────────────────────────────────────────────────
CHUNK_SIZE         = 1000
CHUNK_OVERLAP      = 200
TOP_K              = 5
MAX_CHARS_PER_FILE = 800_000   # a full 10-K narrative (~220-540k chars) fits with margin

# ── MACRO / FRED ──────────────────────────────────────────────────────────────
# How much macroeconomic history the Macro agent reports, expressed in MONTHS
# so the window is independent of each series' publication frequency. Monthly
# series (inflation, Fed rate, unemployment) report this many points; quarterly
# series (GDP) are scaled down to cover the same span (e.g. 24 months → 8 quarters).
MACRO_HISTORY_MONTHS = 24
