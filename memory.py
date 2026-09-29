# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
"""
Conversational memory for the multi-agent system.

A single :class:`ConversationMemory` instance is held by the Orchestrator and
shared (read-only) with every agent. It stores the most recent turns of the
dialogue so that follow-up questions such as "and its stock price?",
"what about last year?" or "compare it with Microsoft" can be resolved:
the router, the ticker/series extractors and the synthesizer all receive the
recent history as additional context.

Only the last ``max_turns`` turns are kept and each stored answer is truncated
to ``max_answer_chars`` characters, so the prompt overhead stays bounded even
in long sessions.
"""

from dataclasses import dataclass


@dataclass
class ConversationTurn:
    """A single exchange in the conversation."""
    question: str
    answer:   str
    route:    str = ""   # which route handled the turn (rag/market/macro/multi)


class ConversationMemory:
    """
    Lightweight, in-memory conversation history.

    The memory is deliberately simple (a bounded list of turns) because the
    goal for the thesis is to demonstrate *contextual continuity* across turns,
    not to build a persistent database. It is rendered to a plain-text block
    via :meth:`as_context` and injected into the relevant prompts.
    """

    def __init__(self, max_turns: int = 5, max_answer_chars: int = 600):
        self.max_turns        = max_turns
        self.max_answer_chars = max_answer_chars
        self.turns: list[ConversationTurn] = []

    # ── WRITE ────────────────────────────────────────────────────────────────

    def add_turn(self, question: str, answer: str, route: str = "") -> None:
        """Records a completed turn and trims history to the last N turns."""
        self.turns.append(
            ConversationTurn(question.strip(), (answer or "").strip(), route)
        )
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def clear(self) -> None:
        """Forgets the whole conversation (used by the 'clear' command)."""
        self.turns.clear()
        print("[Memory] Conversation history cleared.")

    # ── READ ─────────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        return not self.turns

    def as_context(self) -> str:
        """
        Renders the recent turns as a plain-text block for prompt injection.
        Returns an empty string when there is no history (first turn), so
        callers can simply check ``if history:`` before adding it to a prompt.
        """
        if not self.turns:
            return ""

        blocks = []
        for i, turn in enumerate(self.turns, 1):
            answer = turn.answer
            if len(answer) > self.max_answer_chars:
                answer = answer[: self.max_answer_chars].rstrip() + " […]"
            blocks.append(
                f"Turn {i}:\n  User: {turn.question}\n  Assistant: {answer}"
            )
        return "\n\n".join(blocks)

    def __len__(self) -> int:
        return len(self.turns)
