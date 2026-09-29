# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
import os
import re
from pathlib import Path

import pdfplumber
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sec_edgar_downloader import Downloader

from config import (CHUNK_OVERLAP, CHUNK_SIZE, FILINGS_DIR,
                    MAX_CHARS_PER_FILE, SEC_COMPANIES,
                    SEC_FILING_TYPE, SEC_NUM_FILINGS)
from rag.vectorstore import VectorStoreManager

load_dotenv()


# ── 1. DOWNLOAD ───────────────────────────────────────────────────────────────

def _make_downloader() -> Downloader:
    """Builds the SEC EDGAR Downloader using the email from .env."""
    email = os.getenv("SEC_EMAIL")
    if not email:
        raise RuntimeError(
            "SEC_EMAIL is not set. SEC EDGAR requires a contact address in the "
            "User-Agent header of every request; copy .env.example to .env and "
            "fill it in before running the ingestion."
        )
    return Downloader("ThesisProject", email, str(FILINGS_DIR.parent))


def download_filings() -> None:
    """Downloads the 10-K filings from SEC EDGAR for the companies in config."""
    dl = _make_downloader()
    for ticker in SEC_COMPANIES:
        print(f"  [Ingestor] Downloading {SEC_FILING_TYPE} for {ticker}...")
        try:
            dl.get(SEC_FILING_TYPE, ticker, limit=SEC_NUM_FILINGS)
            print(f"  [Ingestor] ✅ {ticker} downloaded")
        except Exception as e:
            print(f"  [Ingestor] ⚠️  Error {ticker}: {e}")


# ── 2. TEXT EXTRACTION ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text from a PDF with pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"  [Ingestor] ⚠️  PDF error {pdf_path.name}: {e}")
    return text


def extract_text_from_html(html_path: Path) -> str:
    """Extracts text from an HTML filing with BeautifulSoup."""
    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"  [Ingestor] ⚠️  HTML error {html_path.name}: {e}")
        return ""


def _extract_primary_10k_html(raw: str) -> str | None:
    """
    A SEC full-submission file bundles many <DOCUMENT> blocks: the 10-K report
    itself PLUS every exhibit and the inline XBRL financial data. Only the block
    whose <TYPE> begins with "10-K" is the report we want; the rest (EX-21
    subsidiary lists, EX-31/32 certifications, EX-101 XBRL …) is 95-98% of the
    bytes and pure noise for retrieval. Returns the primary document's HTML, or
    None if it cannot be located.
    """
    for block in re.split(r"<DOCUMENT>", raw):
        m = re.search(r"<TYPE>\s*([^\s<]+)", block)
        if m and m.group(1).upper().startswith("10-K"):
            body = re.search(r"<TEXT>(.*?)</TEXT>", block, re.S)
            return body.group(1) if body else block
    return None


def extract_text_from_txt(txt_path: Path) -> str:
    """
    Extracts the 10-K narrative from a SEC EDGAR full-submission.txt file.

    The file is SGML wrapping many embedded HTML documents. We first isolate the
    PRIMARY 10-K document — dropping all exhibits and the inline XBRL data, which
    together are 95-98% of the file and only pollute retrieval — then extract its
    text with BeautifulSoup. Falls back to the whole file if the primary document
    cannot be found.
    """
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        primary = _extract_primary_10k_html(content)
        if primary is not None:
            content = primary
        else:
            print(f"  [Ingestor] ⚠️  {txt_path.name}: primary 10-K block not found, "
                  f"using full submission.")

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"  [Ingestor] ⚠️  TXT error {txt_path.name}: {e}")
        return ""


# ── 3. FILE SEARCH ──────────────────────────────────────────────────────────────

def _find_ticker_documents(ticker: str) -> list[tuple[Path, str]]:
    """
    Searches for a single ticker's files in the download directory.
    Returns a list of (file_path, ticker). Ignores files < 50KB (indexes).
    """
    ticker_dir = FILINGS_DIR / ticker.upper()
    if not ticker_dir.exists():
        return []

    filing_type_dir = ticker_dir / SEC_FILING_TYPE
    if not filing_type_dir.exists():
        return []

    documents = []
    for accession_dir in filing_type_dir.iterdir():
        if not accession_dir.is_dir():
            continue
        for file in accession_dir.rglob("*"):
            if file.suffix.lower() in [".pdf", ".htm", ".html", ".txt"]:
                if file.stat().st_size > 50_000:   # > 50KB
                    documents.append((file, ticker.upper()))
    return documents


def find_filing_documents() -> list[tuple[Path, str]]:
    """
    Walks the entire download directory and collects the documents
    of all the tickers present. Returns a list of (file_path, ticker).
    """
    if not FILINGS_DIR.exists():
        print(f"  [Ingestor] ⚠️  Directory not found: {FILINGS_DIR}")
        return []

    all_docs = []
    for ticker_dir in FILINGS_DIR.iterdir():
        if ticker_dir.is_dir():
            all_docs.extend(_find_ticker_documents(ticker_dir.name))
    return all_docs


def find_content_start(text: str) -> int:
    """
    Locates the start of the actual 10-K content, skipping
    the SGML headers and the initial SEC metadata.
    """
    markers = [
        "PART I\nItem 1",
        "PART I\r\nItem 1",
        "Part I\nItem 1",
        "PART I - Item 1",
        "PART I.",
        "Item 1.",
        "Item 1 ",
        "ITEM 1.",
        "ITEM 1 ",
    ]
    text_upper = text.upper()
    for marker in markers:
        idx = text_upper.find(marker.upper())
        if idx != -1:
            return max(0, idx - 200)
    return 0


# ── 4. CHUNKING ───────────────────────────────────────────────────────────────

def _is_boilerplate_chunk(text: str) -> bool:
    """
    Detects low-value 10-K boilerplate — SOX certifications, signature blocks,
    exhibit indices and subsidiary jurisdiction tables — so it is excluded from
    the vector store and stops crowding out real disclosure at retrieval time.
    Deliberately conservative: fires only on strong, unambiguous signals.
    (Takes effect on the NEXT ingestion — re-index to apply.)
    """
    t = text.lower()
    if t.count("/s/") >= 2:                                    # signature block
        return True
    if "i have reviewed this annual report" in t:              # SOX 302 cert
        return True
    if "certify that" in t and "/s/" in t:
        return True
    if "dividend reinvestment plan" in t and "election to participate" in t:
        return True
    if re.search(r"\bex-3\d", t) and "certification" in t:    # exhibit 31/32 cert
        return True
    # Subsidiary jurisdiction table: many uncommon corporate suffixes together
    # (foreign entity types rarely appear this densely in real disclosure prose).
    suffixes = sum(t.count(x) for x in (" ltd.", " llc", " gmbh", " s.a",
                                        " b.v", " kabushiki", " sdn", " co., ltd"))
    if suffixes >= 4:
        return True
    return False


def chunk_text(text: str, metadata: dict) -> tuple[list[str], list[dict]]:
    """
    Splits the text into overlapping chunks.
    The separators are ordered from most to least significant:
    paragraph → line → sentence → word → character.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    raw_chunks = splitter.split_text(text)
    # Drop boilerplate (signatures, certifications, subsidiary tables) so the
    # vector store holds substantive disclosure only — improves retrieval precision.
    chunks    = [c for c in raw_chunks if not _is_boilerplate_chunk(c)]
    metadatas = [metadata.copy() for _ in chunks]
    return chunks, metadatas


# ── 5. DOCUMENT PROCESSING (shared logic) ──────────────────────────────────────

def _ingest_documents_list(
    documents: list[tuple[Path, str]],
    vector_store: VectorStoreManager
) -> int:
    """
    Extracts text, chunks, and indexes a list of (file_path, ticker).
    Returns the total number of chunks added to the vector store.

    Private function shared between ingest_all() and ingest_single_ticker()
    to avoid duplicating the processing logic.
    """
    total_chunks = 0

    for file_path, ticker in documents:
        print(f"\n  [Ingestor] Processing: {ticker} — {file_path.name}")

        # Text extraction based on the file type
        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif file_path.suffix.lower() == ".txt":
            text = extract_text_from_txt(file_path)
        else:
            text = extract_text_from_html(file_path)

        if not text.strip():
            print("    [Ingestor] ⚠️  Empty text, skip")
            continue

        # Skip the initial SGML headers and truncate to the configured limit
        start = find_content_start(text)
        text  = text[start : start + MAX_CHARS_PER_FILE]
        print(f"    [Ingestor] Extracted {len(text):,} characters")

        metadata = {
            "ticker":       ticker,
            "filing_type":  SEC_FILING_TYPE,
            "source_file":  file_path.name,
        }
        chunks, metadatas = chunk_text(text, metadata)
        print(f"    [Ingestor] Split into {len(chunks)} chunks")

        vector_store.add_documents(chunks, metadatas)
        total_chunks += len(chunks)

    return total_chunks


# ── 6. FULL PIPELINE (batch) ───────────────────────────────────────────────────

def ingest_all(vector_store: VectorStoreManager) -> None:
    """
    Full batch pipeline:
      1. Download from SEC EDGAR of all the tickers in config
      2. Search for the downloaded files
      3. Text extraction → chunking → indexing
    """
    print("\n[Ingestor] Starting batch ingestion pipeline...")

    download_filings()

    documents = find_filing_documents()
    print(f"\n[Ingestor] {len(documents)} documents found to process")

    if not documents:
        print("[Ingestor] ⚠️  No documents found.")
        return

    total_chunks = _ingest_documents_list(documents, vector_store)
    print(f"\n[Ingestor] ✅ Ingestion completed. "
          f"Total chunks indexed: {total_chunks}")


# ── 7. ON-DEMAND INGESTION (single ticker) ─────────────────────────────────────

def ingest_single_ticker(ticker: str, vector_store: VectorStoreManager) -> bool:
    """
    Downloads and indexes the most recent 10-K of a single ticker on-demand.

    Strategy:
      1. Check whether the files are already on disk (previous or half-failed
         download). If so, skip the download and use the existing files.
      2. If there are no files on disk, download from SEC EDGAR.
      3. Extract, chunk, and index into the vector store.

    Returns:
        True  if at least one chunk was indexed successfully.
        False if the download failed or the ticker is not valid on SEC EDGAR.
    """
    ticker = ticker.upper()
    print(f"\n[Ingestor] On-demand ingestion for: {ticker}")

    # ── Step 1: check for files already on disk ───────────────────────────────
    documents = _find_ticker_documents(ticker)

    if documents:
        print(f"  [Ingestor] Files already on disk ({len(documents)} found). "
              f"Skipping the download.")
    else:
        # ── Step 2: download from SEC EDGAR ───────────────────────────────────
        print(f"  [Ingestor] No local file found. Downloading from SEC EDGAR...")
        dl = _make_downloader()
        try:
            dl.get(SEC_FILING_TYPE, ticker, limit=SEC_NUM_FILINGS)
            print(f"  [Ingestor] ✅ Download completed for {ticker}")
        except Exception as e:
            print(f"  [Ingestor] ⚠️  Download failed for {ticker}: {e}")
            return False

        # Search for the files again after the download
        documents = _find_ticker_documents(ticker)
        if not documents:
            print(f"  [Ingestor] ⚠️  No document found for {ticker} "
                  f"(ticker not valid on SEC EDGAR?)")
            return False

    # ── Step 3: extraction + chunking + indexing ──────────────────────────────
    total_chunks = _ingest_documents_list(documents, vector_store)

    if total_chunks > 0:
        print(f"\n[Ingestor] ✅ On-demand completed: "
              f"{total_chunks} chunks indexed for {ticker}")
        return True
    else:
        print(f"\n[Ingestor] ⚠️  No chunk indexed for {ticker}")
        return False