# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
import traceback
from dataclasses import dataclass

from config import TOP_K
from rag.vectorstore import VectorStoreManager


@dataclass
class RetrievedChunk:
    """
    Represents a single chunk retrieved from the vector store.
    Using a dataclass instead of a raw dictionary makes the
    code more readable and type-safe — useful for the thesis.
    """
    text:     str
    ticker:   str
    source:   str
    distance: float


class Retriever:
    """
    Retrieves the most relevant chunks from the vector store
    given a natural-language query text.

    Retrieval can optionally be restricted to one or more tickers. When the
    orchestrator has resolved which company(ies) the question is about, it
    passes that list to :meth:`retrieve`, which then applies a ChromaDB
    metadata filter. This keeps the context focused on the right company and
    prevents semantically similar passages from other filings from leaking in.
    """

    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    @staticmethod
    def _build_ticker_filter(tickers: list[str] | None) -> dict | None:
        """
        Builds a ChromaDB ``where`` clause that restricts the search to the
        given tickers' metadata. Returns ``None`` (no filter) when no ticker
        is provided, so the retriever falls back to an unconstrained semantic
        search over the whole collection.
        """
        if not tickers:
            return None

        unique = sorted({t.strip().upper() for t in tickers if t and t.strip()})
        if not unique:
            return None
        if len(unique) == 1:
            return {"ticker": unique[0]}
        return {"ticker": {"$in": unique}}

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        tickers: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if self.vector_store.is_empty():
            print("[Retriever] ⚠️  Vector store empty — run the ingestion first.")
            return []

        where = self._build_ticker_filter(tickers)
        if where is not None:
            print(f"[Retriever] Filtering retrieval by ticker(s): {tickers}")

        results = self.vector_store.query(query, top_k, where=where)
        docs_list = results.get("documents", [[]])[0]
        meta_list = results.get("metadatas", [[]])[0]
        dist_list = results.get("distances", [[]])[0]

        # If the ticker filter returned nothing (e.g. the company was requested
        # but could not be indexed), fall back to an unfiltered search so the
        # agent can still respond instead of returning an empty context.
        if where is not None and not docs_list:
            print("[Retriever] ⚠️  No chunk matched the ticker filter — "
                  "falling back to unfiltered search.")
            results   = self.vector_store.query(query, top_k, where=None)
            docs_list = results.get("documents", [[]])[0]
            meta_list = results.get("metadatas", [[]])[0]
            dist_list = results.get("distances", [[]])[0]

        chunks = []
        for i, doc in enumerate(docs_list):
            try:
                meta     = meta_list[i] if i < len(meta_list) else {}
                distance = dist_list[i] if i < len(dist_list) else 0.0
                chunks.append(RetrievedChunk(
                    text=doc,
                    ticker=meta.get("ticker", "unknown") if isinstance(meta, dict) else "unknown",
                    source=meta.get("source_file", "unknown") if isinstance(meta, dict) else "unknown",
                    distance=round(float(distance), 4) if distance is not None else 0.0
                ))
            except Exception as e:
                print(f"[Retriever] ⚠️  Chunk error {i}: {e}")
                traceback.print_exc()
        return chunks

    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Formats the chunks into a text block ready for the LLM.
        Includes the metadata so the model can cite its sources.
        """
        if not chunks:
            return "No context available."

        sections = []
        for i, chunk in enumerate(chunks, 1):
            sections.append(
                f"[Source {i} — {chunk.ticker} | {chunk.source}]\n{chunk.text}"
            )
        return "\n\n---\n\n".join(sections)
