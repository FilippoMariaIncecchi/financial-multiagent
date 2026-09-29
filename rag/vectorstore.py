# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
import chromadb
from chromadb import Collection
from config import VECTORSTORE_DIR, CHROMA_COLLECTION, EMBEDDING_MODEL


class VectorStoreManager:
    """
    Manages the persistent ChromaDB vector store.

    Embeddings use the Ollama ``nomic-embed-text`` model (768-dim) — a stronger
    retriever than ChromaDB's default all-MiniLM-L6-v2 (384-dim). If the Ollama
    embedding function is unavailable, it falls back to the ChromaDB default.
    NOTE: changing the embedding changes the vector dimension, so the store must
    be re-indexed (delete ``vectorstore/`` and re-ingest) after switching.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
        self.collection: Collection = self._get_or_create_collection()

    def _ollama_embedding_function(self):
        """
        ChromaDB embedding function backed by Ollama ``nomic-embed-text``. Tries
        a couple of constructor signatures for cross-version robustness; returns
        None (→ ChromaDB default all-MiniLM) if the function is unavailable.
        """
        try:
            from chromadb.utils import embedding_functions as ef
        except Exception:
            return None
        oef = getattr(ef, "OllamaEmbeddingFunction", None)
        if oef is None:
            return None
        for kwargs in (
            {"url": "http://localhost:11434/api/embeddings", "model_name": EMBEDDING_MODEL},
            {"model_name": EMBEDDING_MODEL},
            {"url": "http://localhost:11434", "model_name": EMBEDDING_MODEL},
        ):
            try:
                return oef(**kwargs)
            except Exception:
                continue
        return None

    def _get_or_create_collection(self) -> Collection:
        embedding_fn = self._ollama_embedding_function()
        kwargs = {"name": CHROMA_COLLECTION, "metadata": {"hnsw:space": "cosine"}}
        if embedding_fn is not None:
            kwargs["embedding_function"] = embedding_fn
        else:
            print("  [VectorStore] ⚠️  Ollama embedding unavailable — "
                  "using ChromaDB default (all-MiniLM).")
        return self.client.get_or_create_collection(**kwargs)

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def refresh(self) -> None:
        self.collection = self._get_or_create_collection()
        print(f"  [VectorStore] Collection reloaded. "
              f"Available chunks: {self.collection.count()}")

    def get_indexed_tickers(self) -> set[str]:
        """
        Returns the set of tickers already present in the collection.

        Uses include=["metadatas"] without retrieving the chunk text —
        a lightweight operation even with thousands of indexed documents.
        Called once at bootstrap (result cached in the Orchestrator).
        """
        if self.is_empty():
            return set()
        try:
            results = self.collection.get(include=["metadatas"])
            return {
                meta["ticker"]
                for meta in results.get("metadatas", [])
                if isinstance(meta, dict) and "ticker" in meta
            }
        except Exception as e:
            print(f"  [VectorStore] ⚠️  get_indexed_tickers error: {e}")
            return set()

    def add_documents(self, chunks: list[str], metadatas: list[dict]) -> None:
        if not chunks:
            print("  [VectorStore] No chunk to add.")
            return

        print(f"  [VectorStore] Indexing {len(chunks)} chunks...")
        start_idx = self.collection.count()
        ids = [f"chunk_{start_idx + i}" for i in range(len(chunks))]

        # ChromaDB computes the embeddings internally via the collection's
        # embedding function (Ollama nomic-embed-text, or all-MiniLM fallback).
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )
        print(f"  [VectorStore] ✅ {len(chunks)} chunks added. "
              f"Total in DB: {self.collection.count()}")

    def query(self, query_text: str, top_k: int, where: dict | None = None) -> dict:
        # ChromaDB embeds the query with the same model used for indexing.
        # The optional `where` clause filters on metadata (e.g. ticker) so the
        # search can be restricted to one or more specific companies.
        kwargs = {
            "query_texts": [query_text],
            "n_results":   top_k,
            "include":     ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        return results

    def count(self) -> int:
        return self.collection.count()