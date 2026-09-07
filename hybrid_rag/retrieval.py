"""Retrieval strategies behind a single seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hybrid_rag.entities import Entity
from hybrid_rag.graph import GraphTraversal
from hybrid_rag.kind import Kind
from hybrid_rag.vector import VectorStore


@dataclass
class HybridResult:
    entities: list[Entity]


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 5) -> HybridResult: ...


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


@dataclass
class NaiveRetriever:
    """Vector ranking, hydrated into domain entities without neighbour expansion."""

    vector_store: VectorStore
    graph: GraphTraversal

    def retrieve(self, query: str, k: int = 5) -> HybridResult:
        results = self.vector_store.search(query, k=k)
        if not results:
            return HybridResult(entities=[])
        rank = {result.id: i for i, result in enumerate(results)}
        entities = self.graph.get_nodes([r.id for r in results])
        entities.sort(key=lambda entity: rank[entity.id])
        return HybridResult(entities=entities)


@dataclass
class HybridRetriever:
    """Vector ranking plus graph expansion over IMPLEMENTS and COVERS."""

    vector_store: VectorStore
    graph: GraphTraversal

    def retrieve(self, query: str, k: int = 5) -> HybridResult:
        results = self.vector_store.search(query, k=k)

        req_ids: set[str] = set()
        test_ids: set[str] = set()

        for result in results:
            if result.doc_type is Kind.TEST_SCENARIO:
                test_ids.add(result.id)
            else:
                req_ids.add(result.id)

        _expand_from_requirements(self.graph, req_ids, test_ids)
        _expand_from_tests(self.graph, test_ids, req_ids)

        return HybridResult(entities=self.graph.get_nodes(list(req_ids | test_ids)))
