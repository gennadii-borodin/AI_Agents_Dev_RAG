"""Demo runner: naive vs hybrid retrieval compared side by side with LLM answer."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hybrid_rag import graph
from hybrid_rag.documents import DOCUMENTS, QUERIES
from hybrid_rag.entities import Entity
from hybrid_rag.graph import RequirementGraph
from hybrid_rag.hybrid import HybridResult, hybrid_search, naive_search
from hybrid_rag.kind import Kind
from hybrid_rag.vector import SearchResult, VectorStore

load_dotenv()

console = Console()
CHAT_MODEL = "gpt-4.1-mini"


def short_kind(kind: Kind) -> str:
    return kind.short


def format_entities(entities: list[Entity]) -> str:
    lines = []
    for entity in entities:
        lines.append(
            f"- [{short_kind(entity.kind)}] {entity.id}: {entity.title} — {entity.text}"
        )
    return "\n".join(lines)


def ask_llm(client: OpenAI, question: str, context: str) -> str:
    prompt = (
        "Answer using only the retrieved context. "
        "If the context is not enough, say what is missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content or ""


def render_comparison(
    question: str, naive: list[SearchResult], hybrid: HybridResult
) -> None:
    table = Table(title=f"Query: {question}")
    table.add_column("Naive (vector only)")
    table.add_column("Hybrid (vector + graph)")

    naive_rows = [f"[{r.doc_type.short}] {r.id}: {r.text[:60]}" for r in naive]
    hybrid_rows = [f"[{short_kind(e.kind)}] {e.id}: {e.title}" for e in hybrid.entities]

    height = max(len(naive_rows), len(hybrid_rows))
    for i in range(height):
        left = naive_rows[i] if i < len(naive_rows) else ""
        right = hybrid_rows[i] if i < len(hybrid_rows) else ""
        table.add_row(left, right)

    console.print(table)


def main() -> None:
    console.rule("[bold cyan]Hybrid RAG Demo")

    console.print(
        Panel(
            "Vector search + Neo4j graph traversal.\n"
            "Vector search finds semantically relevant requirements/tests,\n"
            "graph traversal expands results through IMPLEMENTS and COVERS links."
        )
    )

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    with graph.connect() as conn:
        conn.session.execute_write(graph.erase_graph)
        conn.session.execute_write(graph.build_graph, DOCUMENTS)
        console.print("[green]knowledge graph ready[/green]")

        store = VectorStore()
        store.clear()
        store.add_all(DOCUMENTS)
        console.print(f"[green]vector store ready: {store.count()} documents[/green]")

        traversal = RequirementGraph(conn.session)

        for item in QUERIES:
            question = item["question"]
            console.rule(f"[bold]Q: {question}")

            naive = naive_search(store, question)
            hybrid = hybrid_search(store, traversal, question)

            render_comparison(question, naive, hybrid)

            context = format_entities(hybrid.entities)
            console.print(Panel("Hybrid context sent to LLM:\n" + context))
            answer = ask_llm(client, question, context)
            console.print(Panel(answer, title="Assistant"))


if __name__ == "__main__":
    main()
