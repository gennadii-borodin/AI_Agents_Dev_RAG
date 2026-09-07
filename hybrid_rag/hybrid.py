"""Hybrid retrieval combining vector similarity with graph traversal."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.graph import (
    Queryable,
    get_children_of_br,
    get_related_requirements,
    get_related_tests,
)
from hybrid_rag.vector import SearchResult, VectorStore


@dataclass
class Entity:
    id: str
    title: str
    text: str
    kind: str


@dataclass
class HybridResult:
    entities: list[Entity]


_REQ_LABELS = {"FunctionalRequirement", "NonFunctionalRequirement"}
_BR_LABEL = "BusinessRequirement"


def _fetch_nodes(session: Queryable, ids: set[str]) -> list[Entity]:
    if not ids:
        return []
    rows = session.run(
        "MATCH (n) WHERE n.id IN $ids "
        "RETURN n.id AS id, n.title AS title, n.text AS text, "
        "labels(n)[0] AS label ORDER BY n.id",
        ids=list(ids),
    )
    return [
        Entity(id=r["id"], title=r["title"], text=r["text"], kind=r["label"])
        for r in rows
    ]


def _expand_from_requirements(
    session: Queryable, req_ids: set[str], test_ids: set[str]
) -> None:
    """Expand requirement seeds: tests covering them, plus BR children and tests."""
    seeds = _fetch_nodes(session, req_ids)
    br_ids = {e.id for e in seeds if e.kind == _BR_LABEL}
    fr_nfr_ids = {e.id for e in seeds if e.kind in _REQ_LABELS}

    # Tests that cover the FR/NFR seeds (reverse COVERS)
    test_ids.update(r["id"] for r in get_related_tests(session, list(fr_nfr_ids)))

    # Parent business requirements of the FR/NFR seeds (reverse IMPLEMENTS)
    parents = session.run(
        "MATCH (n)-[:IMPLEMENTS]->(br) WHERE n.id IN $ids RETURN br.id AS id",
        ids=list(fr_nfr_ids),
    )
    br_ids.update(r["id"] for r in parents)

    # Children implementing the BRs (forward IMPLEMENTS, from either seeds or parents)
    if br_ids:
        children = get_children_of_br(session, list(br_ids))
        child_ids = [r["id"] for r in children]
        req_ids.update(child_ids)
        test_ids.update(r["id"] for r in get_related_tests(session, child_ids))


def _expand_from_tests(
    session: Queryable, test_ids: set[str], req_ids: set[str]
) -> None:
    """Expand test seeds: their parent FR/NFR and grandparent BR."""
    for row in get_related_requirements(session, list(test_ids)):
        req_ids.add(row["id"])
        if row.get("parent_id"):
            req_ids.add(row["parent_id"])


def naive_search(
    vector_store: VectorStore, query: str, k: int = 5
) -> list[SearchResult]:
    return vector_store.search(query, k=k)


def hybrid_search(
    vector_store: VectorStore, session: Queryable, query: str, k: int = 5
) -> HybridResult:
    results = vector_store.search(query, k=k)

    req_ids: set[str] = set()
    test_ids: set[str] = set()

    for r in results:
        if r.doc_type == "test":
            test_ids.add(r.id)
        else:
            req_ids.add(r.id)

    _expand_from_requirements(session, req_ids, test_ids)
    _expand_from_tests(session, test_ids, req_ids)

    return HybridResult(entities=_fetch_nodes(session, req_ids | test_ids))
