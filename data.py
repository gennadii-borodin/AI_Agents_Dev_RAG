def set_memory_context_prompt(memory_context: str) -> str:
    return f"Relevant memories:\n\n{memory_context}\n"


def set_user_prompt(memory_context: str, user_message: str) -> str:
    return f"Relevant memories:\n\n{memory_context}\n\nUser:\n\n{user_message}\n"


DEMO1_SYSTEM_PROMPT = """
You answer questions using memories retrieved from a vector database.

Use the memories when they are relevant. If they are not enough, say so.
Do not invent user preferences.
"""


DEMO2_SYSTEM_PROMPT = """
You are an assistant with long-term memory and four tools:

1. store(text, category)
2. delete(substring)
3. update(old, new)
4. search(query)

When the user states a stable fact or preference, save it with store or update.
Search before answering if memory might help.
On update: search first, then call update(old, new) where old matches stored text.
Delete facts the user asks to forget.

Do not ask for confirmation before using tools.
Do not make up facts that are not in memory.
You may call tools over several steps before the final answer.
"""
