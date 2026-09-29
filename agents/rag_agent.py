# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config import LLM_MODEL, LLM_TEMPERATURE, LLM_NUM_CTX
from rag.retriever import Retriever, RetrievedChunk


@dataclass
class RagAnswer:
    """
    Structured output of the RAG Agent.
    Separating the answer from its sources is essential for
    RAGAS evaluation (faithfulness, context recall).
    """
    answer:  str
    sources: list[RetrievedChunk]
    context: str   # raw text passed to the LLM — needed for RAGAS


SYSTEM_PROMPT = """You are a financial analyst assistant specialized in SEC filings.
You answer questions about companies based EXCLUSIVELY on the context provided below.
Rules you must always follow:
1. Base your answer ONLY on the provided context. Never use external knowledge.
2. If the context does not contain enough information, say so explicitly.
3. Always mention which company (ticker) and source document your answer refers to.
4. Be precise and concise. Avoid unnecessary introductions.
5. If numbers or financial figures are present in the context, include them in your answer.
"""

# Conversation history is provided ONLY so the agent can interpret follow-up
# references (e.g. "it", "that company", "the same period"). It must never be
# treated as a source of facts — the answer must still come strictly from the
# retrieved SEC context, preserving faithfulness for the RAGAS evaluation.
HISTORY_PROMPT = """Conversation so far (use ONLY to resolve references such as
"it" or "that company"; NEVER use it as a source of facts — answer strictly
from the SEC context provided in the next message):

{history}"""


class RagAgent:
    """
    RAG Agent: retrieves context from the indexed documents
    and generates answers grounded in the sources.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            num_ctx=LLM_NUM_CTX
        )

    def run(
        self,
        question: str,
        tickers: list[str] | None = None,
        history: str = "",
    ) -> RagAnswer:
        """
        Runs the full retrieve → augment → generate cycle.

        Args:
            question: the user's natural-language question
            tickers:  optional list of tickers to restrict retrieval to
                      (resolved upstream by the orchestrator). When given,
                      only those companies' chunks are searched.
            history:  optional conversation history, used solely to resolve
                      follow-up references — never as a source of facts.

        Returns:
            RagAnswer with the answer, sources, and raw context
        """
        print(f"\n[RagAgent] Question received: '{question}'")

        # ── RETRIEVE ─────────────────────────────────────────
        chunks = self.retriever.retrieve(question, tickers=tickers)
        if not chunks:
            return RagAnswer(
                answer="I did not find any relevant documents in the vector store. "
                       "Make sure you have run the ingestion.",
                sources=[],
                context=""
            )

        print(f"[RagAgent] Retrieved {len(chunks)} chunks "
              f"(distances: {[c.distance for c in chunks]})")

        # ── AUGMENT ──────────────────────────────────────────
        context = self.retriever.format_context(chunks)
        user_message = f"""Context extracted from SEC filings:
{context}
---
Question: {question}
Answer based strictly on the context above:"""

        # ── GENERATE ─────────────────────────────────────────
        print("[RagAgent] Generating answer with LLM...")
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        if history:
            messages.append(SystemMessage(content=HISTORY_PROMPT.format(history=history)))
        messages.append(HumanMessage(content=user_message))
        response = self.llm.invoke(messages)

        return RagAnswer(
            answer=response.content,
            sources=chunks,
            context=context
        )

    def print_answer(self, rag_answer: RagAnswer) -> None:
        """Prints the formatted answer with its sources — useful for debugging."""
        print("\n" + "="*60)
        print("RAG AGENT ANSWER")
        print("="*60)
        print(rag_answer.answer)
        print("\n--- SOURCES USED ---")
        seen = set()
        for chunk in rag_answer.sources:
            key = (chunk.ticker, chunk.source)
            if key not in seen:
                print(f"  • {chunk.ticker} | {chunk.source} "
                      f"(distance: {chunk.distance})")
                seen.add(key)
        print("="*60)
