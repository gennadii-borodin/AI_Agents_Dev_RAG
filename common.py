"""Shared code for both RAG demos: Chroma, embeddings, memory CRUD, LLM calls."""

from __future__ import annotations

import json
import os
import uuid
from typing import List

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data import set_memory_context_prompt, set_user_prompt

load_dotenv()
console = Console()
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="agent_memory")


def separator(title: str):
    console.rule(f"[bold cyan]{title}")


def embed(text: str) -> List[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


class MemoryStore:

    def add(self, text: str, category: str = "general"):
        memory_id = str(uuid.uuid4())

        collection.add(
            ids=[memory_id],
            documents=[text],
            embeddings=[embed(text)],
            metadatas=[{"category": category}],
        )

        console.print(f"[green]stored:[/green] {text}")
        return memory_id

    def search(self, query: str, k: int = 3, verbose: bool = True):
        if self.count() == 0:
            return {"documents": [[]], "ids": [[]], "distances": [[]]}

        result = collection.query(query_embeddings=[embed(query)], n_results=k)

        docs = result["documents"][0]
        ids = result["ids"][0]
        distances = result["distances"][0]

        if verbose:
            separator("Retrieved memories")

            table = Table()
            table.add_column("Score")
            table.add_column("Memory")

            for d, text in zip(distances, docs):
                score = 1.0 - d
                table.add_row(f"{score:.3f}", text)

            console.print(table)

        return result

    def count(self):
        return len(collection.get()["ids"])

    def delete(self, memory_id: str):
        collection.delete(ids=[memory_id])
        console.print(f"[red]deleted[/red] {memory_id}")

    def delete_by_text(self, substring: str):
        data = collection.get()
        deleted = 0
        for idx, text in zip(data["ids"], data["documents"]):
            if substring.lower() in text.lower():
                collection.delete(ids=[idx])
                deleted += 1

        console.print(f"[red]deleted {deleted} memories[/red]")

    def update(self, old_substring: str, new_text: str, category: str = "general"):
        self.delete_by_text(old_substring)
        self.add(new_text, category)

    def show_database(self):
        separator("Vector database")

        data = collection.get()

        table = Table()
        table.add_column("ID")
        table.add_column("Category")
        table.add_column("Memory")

        for idx, meta, doc in zip(data["ids"], data["metadatas"], data["documents"]):
            table.add_row(idx[:8], meta["category"], doc)

        console.print(table)

    def clear(self):
        data = collection.get()

        if len(data["ids"]):
            collection.delete(ids=data["ids"])

        console.print("[yellow]database cleared[/yellow]")


memory = MemoryStore()


def build_context(query: str, k: int = 3):
    result = memory.search(query, k=k, verbose=True)
    docs = result["documents"][0]
    return "\n".join(f"- {doc}" for doc in docs)


def ask_llm(user_message: str, memory_context: str = "", system_prompt: str = ""):
    separator("Prompt sent to GPT")

    prompt = set_memory_context_prompt(memory_context)
    console.print(Panel(prompt))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append(
        {
            "role": "user",
            "content": set_user_prompt(memory_context, user_message),
        }
    )

    response = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    answer = response.choices[0].message.content

    separator("Assistant")
    console.print(answer)
    return answer


def dump_json(obj):
    console.print(Panel(json.dumps(obj, indent=2, ensure_ascii=False)))
