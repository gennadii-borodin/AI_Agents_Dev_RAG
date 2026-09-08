"""
Agentic RAG demo.

Memory operations are exposed as tools; the model decides when to call them.

Run: python demo_agentic_rag.py
"""

import json
import os

from openai import OpenAI

from common import memory, separator
from data import DEMO2_SYSTEM_PROMPT as SYSTEM_PROMPT

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL") or "gpt-4.1-mini"


def tool_store(text: str, category: str = "general"):
    separator("[TOOL] store_memory")
    memory_id = memory.add(text, category)
    return {"status": "stored", "id": memory_id, "text": text}


def tool_delete(substring: str):
    separator("[TOOL] delete_memory")
    memory.delete_by_text(substring)
    return {"status": "deleted", "match": substring}


def tool_update(old: str, new: str):
    separator("[TOOL] update_memory")
    memory.update(old, new)
    return {"status": "updated", "old": old, "new": new}


def tool_search(query: str):
    separator("[TOOL] search_memory")
    result = memory.search(query, verbose=True)
    return {"query": query, "documents": result["documents"][0]}


TOOLS = {
    "store": tool_store,
    "delete": tool_delete,
    "update": tool_update,
    "search": tool_search,
}

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "store",
            "description": "Store a memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete",
            "description": "Delete memories containing substring",
            "parameters": {
                "type": "object",
                "properties": {"substring": {"type": "string"}},
                "required": ["substring"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update",
            "description": "Replace a stored fact: delete memories matching old, store new",
            "parameters": {
                "type": "object",
                "properties": {
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search memory",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def execute_tool_call(tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    separator(f"[AGENT TOOL CALL] {name}")
    return TOOLS[name](**args)


def run_agent(user_message: str, max_steps: int = 6):
    separator(f"USER: {user_message}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    answer = None

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMA,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            answer = msg.content
            break

        for tool_call in msg.tool_calls:
            result = execute_tool_call(tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    separator("AGENT RESPONSE")
    print(answer)
    return answer


def main():
    separator("Agentic RAG Demo")

    print(
        """
Same Chroma store as the classical demo, but:

- store / update / delete / search are tools
- the model chooses when to call them
- no LangChain or other agent framework — just a tool loop
"""
    )

    memory.clear()
    input("\nPress ENTER to start...\n")

    run_agent("My favorite IDE is PyCharm")
    input("\n--- ENTER ---\n")

    run_agent("I mostly write Python")
    input("\n--- ENTER ---\n")

    run_agent("What IDE do I use?")
    input("\n--- ENTER ---\n")

    run_agent("I switched to VS Code")
    input("\n--- ENTER ---\n")

    run_agent("What IDE do I use now?")
    input("\n--- ENTER ---\n")

    run_agent("Forget everything about IDEs")
    input("\n--- ENTER ---\n")

    run_agent("What do you know about me?")

    separator("DONE")


if __name__ == "__main__":
    main()
