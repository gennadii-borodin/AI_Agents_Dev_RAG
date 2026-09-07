"""Tests for retrieval strategies behind the Retriever seam."""

from unittest.mock import MagicMock

from fakes import FakeGraph

from hybrid_rag.entities import Entity
from hybrid_rag.kind import Kind
from hybrid_rag.retrieval import HybridRetriever, NaiveRetriever
from hybrid_rag.vector import SearchResult


def make_entity(doc_id: str, kind: Kind) -> Entity:
    return Entity(id=doc_id, title="t", text="x", kind=kind)


def test_naive_retriever_restores_vector_order_through_graph_lookup():
    store = MagicMock()
    store.search.return_value = [
        SearchResult(
            id="BR-003", text="a", doc_type=Kind.BUSINESS_REQUIREMENT, score=0.9
        ),
        SearchResult(id="TS-011", text="b", doc_type=Kind.TEST_SCENARIO, score=0.6),
        SearchResult(
            id="FR-005", text="c", doc_type=Kind.FUNCTIONAL_REQUIREMENT, score=0.4
        ),
    ]
    graph = FakeGraph(
        nodes={
            "BR-003": make_entity("BR-003", Kind.BUSINESS_REQUIREMENT),
            "TS-011": make_entity("TS-011", Kind.TEST_SCENARIO),
            "FR-005": make_entity("FR-005", Kind.FUNCTIONAL_REQUIREMENT),
        }
    )

    result = NaiveRetriever(store, graph).retrieve("перевод средств")

    assert [e.id for e in result.entities] == ["BR-003", "TS-011", "FR-005"]


def test_naive_retriever_does_not_expand_neighbors():
    store = MagicMock()
    store.search.return_value = [
        SearchResult(
            id="BR-003", text="a", doc_type=Kind.BUSINESS_REQUIREMENT, score=0.9
        ),
    ]
    graph = FakeGraph(
        nodes={"BR-003": make_entity("BR-003", Kind.BUSINESS_REQUIREMENT)},
        children_by_br={"BR-003": ["FR-005"]},
        tests_by_requirement={"FR-005": ["TS-011"]},
    )

    result = NaiveRetriever(store, graph).retrieve("перевод средств")

    assert [e.id for e in result.entities] == ["BR-003"]


def test_naive_retriever_empty_search_returns_empty():
    store = MagicMock()
    store.search.return_value = []

    result = NaiveRetriever(store, FakeGraph(nodes={})).retrieve("ничего")

    assert result.entities == []


def test_hybrid_retriever_expands_requirement_seeds():
    store = MagicMock()
    store.search.return_value = [
        SearchResult(
            id="NFR-004",
            text="text",
            doc_type=Kind.NON_FUNCTIONAL_REQUIREMENT,
            score=0.8,
        ),
        SearchResult(
            id="BR-006", text="text", doc_type=Kind.BUSINESS_REQUIREMENT, score=0.8
        ),
    ]

    graph = FakeGraph(
        nodes={
            "NFR-004": make_entity("NFR-004", Kind.NON_FUNCTIONAL_REQUIREMENT),
            "BR-006": make_entity("BR-006", Kind.BUSINESS_REQUIREMENT),
            "FR-013": make_entity("FR-013", Kind.FUNCTIONAL_REQUIREMENT),
            "TS-031": make_entity("TS-031", Kind.TEST_SCENARIO),
            "TS-032": make_entity("TS-032", Kind.TEST_SCENARIO),
        },
        tests_by_requirement={
            "NFR-004": ["TS-031", "TS-032"],
            "FR-013": [],
        },
        children_by_br={"BR-006": ["FR-013"]},
        parent_by_requirement={"NFR-004": "BR-006"},
    )

    result = HybridRetriever(store, graph).retrieve("секреты и шифрование")

    entity_ids = {e.id for e in result.entities}
    assert {"NFR-004", "BR-006", "FR-013", "TS-031", "TS-032"} <= entity_ids


def test_hybrid_retriever_expands_business_requirement_seed():
    store = MagicMock()
    store.search.return_value = [
        SearchResult(
            id="BR-003", text="text", doc_type=Kind.BUSINESS_REQUIREMENT, score=0.8
        )
    ]

    graph = FakeGraph(
        nodes={
            "BR-003": make_entity("BR-003", Kind.BUSINESS_REQUIREMENT),
            "FR-005": make_entity("FR-005", Kind.FUNCTIONAL_REQUIREMENT),
            "TS-011": make_entity("TS-011", Kind.TEST_SCENARIO),
        },
        tests_by_requirement={"FR-005": ["TS-011"]},
        children_by_br={"BR-003": ["FR-005"]},
    )

    result = HybridRetriever(store, graph).retrieve("перевод средств")

    entity_ids = {e.id for e in result.entities}
    assert {"BR-003", "FR-005", "TS-011"} <= entity_ids


def test_hybrid_retriever_expands_test_seed_to_parents():
    store = MagicMock()
    store.search.return_value = [
        SearchResult(id="TS-037", text="text", doc_type=Kind.TEST_SCENARIO, score=0.8)
    ]

    graph = FakeGraph(
        nodes={
            "TS-037": make_entity("TS-037", Kind.TEST_SCENARIO),
            "NFR-008": make_entity("NFR-008", Kind.NON_FUNCTIONAL_REQUIREMENT),
            "BR-006": make_entity("BR-006", Kind.BUSINESS_REQUIREMENT),
        },
        tests_by_requirement={"NFR-008": ["TS-037"]},
        parent_by_requirement={"NFR-008": "BR-006"},
    )

    result = HybridRetriever(store, graph).retrieve("какие тесты про сессию")

    entity_ids = {e.id for e in result.entities}
    assert {"TS-037", "NFR-008", "BR-006"} <= entity_ids


def test_hybrid_retriever_returns_entities_with_kinds():
    store = MagicMock()
    store.search.return_value = [
        SearchResult(
            id="BR-001", text="text", doc_type=Kind.BUSINESS_REQUIREMENT, score=0.8
        )
    ]

    graph = FakeGraph(
        nodes={
            "BR-001": make_entity("BR-001", Kind.BUSINESS_REQUIREMENT),
            "FR-001": make_entity("FR-001", Kind.FUNCTIONAL_REQUIREMENT),
        },
        children_by_br={"BR-001": ["FR-001"]},
    )

    result = HybridRetriever(store, graph).retrieve("вход")

    kinds = {e.kind for e in result.entities}
    assert Kind.BUSINESS_REQUIREMENT in kinds
    assert Kind.FUNCTIONAL_REQUIREMENT in kinds
