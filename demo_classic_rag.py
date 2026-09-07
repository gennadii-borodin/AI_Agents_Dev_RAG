"""
Classical RAG demo.

The application code decides when to add, update, or delete memories.
The LLM only gets retrieved context and answers the question.

Run: python demo_classic_rag.py
"""

from common import memory, build_context, ask_llm, separator
from data import DEMO1_SYSTEM_PROMPT as SYSTEM_PROMPT


def ask(question: str):
    separator(f"USER: {question}")
    context = build_context(question)
    ask_llm(user_message=question, memory_context=context, system_prompt=SYSTEM_PROMPT)


def pause():
    input("\nPress ENTER to continue...\n")


def main():
    separator("Classical RAG Demo")

    print(
        """
Classical vector RAG:

- memory writes are explicit in code (add / update / delete)
- retrieval runs on each question
- the LLM does not manage the database
"""
    )

    memory.clear()
    pause()

    separator("Step 1 — store memories")
    memory.add("The user's favorite IDE is PyCharm.", category="preference")
    memory.add("The user primarily writes Python.", category="skill")
    memory.add("The user is vegetarian.", category="preference")
    memory.show_database()
    pause()

    separator("Step 2 — question")
    ask("Which IDE do I use?")
    pause()

    separator("Step 3 — another question")
    ask("Recommend lunch for me.")
    pause()

    separator("Step 4 — update memory")
    print("User: I switched from PyCharm to VS Code.")
    print("In classical RAG the code calls memory.update(), not the model.\n")

    memory.update(
        old_substring="PyCharm",
        new_text="The user's favorite IDE is VS Code.",
        category="preference",
    )
    memory.show_database()
    pause()

    separator("Step 5 — ask again")
    ask("Which IDE do I use?")
    pause()

    separator("Step 6 — delete memory")
    print("User: Forget everything about IDEs.")
    print("Again, deletion is done in application code.\n")

    memory.delete_by_text("IDE")
    memory.show_database()
    pause()

    separator("Step 7 — ask again")
    ask("Which IDE do I use?")
    pause()

    separator("Demo complete")


if __name__ == "__main__":
    main()
