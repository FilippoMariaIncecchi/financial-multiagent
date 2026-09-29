# =============================================================================
# Financial Multi-Agent System
# Author:    Filippo Maria Incecchi
# Professor: Angela Locoro
# Course:    Analytics and Data Science for Economics and Management
# Università degli Studi di Brescia
# =============================================================================
from agents.orchestrator import Orchestrator
from rag.ingestor import ingest_all


def main():
    print("=" * 60)
    print("  Financial Multi-Agent System — Thesis Project")
    print("  Filippo Maria Incecchi — UniBS Analytics and Data Science for Economics and Management")
    print("=" * 60)
    print("\n[Main] Initializing system...")

    orchestrator = Orchestrator()

    if orchestrator.vector_store.is_empty():
        print("\n[Main] Vector store empty — starting document ingestion...")
        ingest_all(orchestrator.vector_store)
        orchestrator.vector_store.refresh()
    else:
        count = orchestrator.vector_store_count
        print(f"\n[Main] Vector store already populated ({count} chunks). "
              f"Ingestion skipped.")

    print("\n[Main] System ready.")
    print("[Main] Commands: 'exit' to quit · 'clear' to reset the conversation memory.\n")
    print("-" * 60)

    while True:
        try:
            question = input("\n📊 Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[Main] Exiting.")
            break

        if not question:
            continue

        if question.lower() in ["exit", "quit", "q"]:
            print("[Main] Exiting.")
            break

        # Reset the conversational memory without restarting the program.
        if question.lower() in ["clear", "reset"]:
            orchestrator.memory.clear()
            continue

        answer = orchestrator.run(question)

        print("\n" + "=" * 60)
        print("💬 ANSWER")
        print("=" * 60)
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()
