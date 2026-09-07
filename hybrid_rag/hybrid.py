"""Hybrid retrieval combining vector similarity with graph traversal."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.entities import Entity
from hybrid_rag.graph import GraphTraversal
from hybrid_rag.kind import Kind
from hybrid_rag.vector import SearchResult, VectorStore


@dataclass
class HybridResult:
    entities: list[Entity]


_REQ_KINDS = frozenset({Kind.FUNCTIONAL_REQUIREMENT, Kind.NON_FUNCTIONAL_REQUIREMENT})


def _expand_from_requirements(
    graph: GraphTraversal, req_ids: set[str], test_ids: set[str]
) -> None:
    """Expand requirement seeds: tests covering them, plus BR children and tests."""
    seeds = graph.get_nodes(list(req_ids))
    br_ids = {e.id for e in seeds if e.kind is Kind.BUSINESS_REQUIREMENT}
    fr_nfr_ids = {e.id for e in seeds if e.kind in _REQ_KINDS}

    # Tests that cover the FR/NFR seeds (reverse COVERS)
    test_ids.update(e.id for e in graph.get_related_tests(list(fr_nfr_ids)))

    # Parent business requirements of the FR/NFR seeds (reverse IMPLEMENTS)
    br_ids.update(
        e.id for e in graph.get_parent_business_requirements(list(fr_nfr_ids))
    )

    # Children implementing the BRs (forward IMPLEMENTS, from either seeds or parents)
    if br_ids:
        child_ids = [e.id for e in graph.get_children_of_br(list(br_ids))]
        req_ids.update(child_ids)
        test_ids.update(e.id for e in graph.get_related_tests(child_ids))


def _expand_from_tests(
    graph: GraphTraversal, test_ids: set[str], req_ids: set[str]
) -> None:
    """Expand test seeds: their parent FR/NFR and grandparent BR."""
    for covered in graph.get_related_requirements(list(test_ids)):
        req_ids.add(covered.requirement.id)
        if covered.parent:
            req_ids.add(covered.parent.id)


def naive_search(
    vector_store: VectorStore, query: str, k: int = 5
) -> list[SearchResult]:
    return vector_store.search(query, k=k)


def hybrid_search(
    vector_store: VectorStore, graph: GraphTraversal, query: str, k: int = 5
) -> HybridResult:
    results = vector_store.search(query, k=k)

    req_ids: set[str] = set()
    test_ids: set[str] = set()

    for r in results:
        if r.doc_type is Kind.TEST_SCENARIO:
            test_ids.add(r.id)
        else:
            req_ids.add(r.id)

    _expand_from_requirements(graph, req_ids, test_ids)
    _expand_from_tests(graph, test_ids, req_ids)

    return HybridResult(entities=graph.get_nodes(list(req_ids | test_ids)))
