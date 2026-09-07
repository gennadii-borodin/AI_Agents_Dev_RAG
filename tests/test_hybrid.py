"""Tests for hybrid retrieval, mocking the vector store and graph traversal."""

from unittest.mock import MagicMock

from fakes import FakeGraph

from hybrid_rag.entities import Entity
from hybrid_rag.hybrid import hybrid_search, naive_search
from hybrid_rag.kind import Kind
from hybrid_rag.vector import SearchResult


def make_entity(doc_id: str, kind: Kind) -> Entity:
    return Entity(id=doc_id, title="t", text="x", kind=kind)


def make_search_result(doc_id: str, kind: Kind) -> SearchResult:
    return SearchResult(id=doc_id, text="text", doc_type=kind, score=0.8)


def test_naive_search_returns_vector_results_only():
    store = MagicMock()
    store.search.return_value = [
        make_search_result("BR-006", Kind.BUSINESS_REQUIREMENT)
    ]

    results = naive_search(store, "security")

    store.search.assert_called_once_with("security", k=5)
    assert [r.id for r in results] == ["BR-006"]


def test_hybrid_search_expands_requirement_seeds():
    store = MagicMock()
    store.search.return_value = [
        make_search_result("NFR-004", Kind.NON_FUNCTIONAL_REQUIREMENT),
        make_search_result("BR-006", Kind.BUSINESS_REQUIREMENT),
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

    result = hybrid_search(store, graph, "секреты и шифрование")

    entity_ids = {e.id for e in result.entities}
    assert {"NFR-004", "BR-006", "FR-013", "TS-031", "TS-032"} <= entity_ids


def test_hybrid_search_expands_business_requirement_seed():
    store = MagicMock()
    store.search.return_value = [
        make_search_result("BR-003", Kind.BUSINESS_REQUIREMENT)
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

    result = hybrid_search(store, graph, "перевод средств")

    entity_ids = {e.id for e in result.entities}
    assert {"BR-003", "FR-005", "TS-011"} <= entity_ids


def test_hybrid_search_expands_test_seed_to_parents():
    store = MagicMock()
    store.search.return_value = [make_search_result("TS-037", Kind.TEST_SCENARIO)]

    graph = FakeGraph(
        nodes={
            "TS-037": make_entity("TS-037", Kind.TEST_SCENARIO),
            "NFR-008": make_entity("NFR-008", Kind.NON_FUNCTIONAL_REQUIREMENT),
            "BR-006": make_entity("BR-006", Kind.BUSINESS_REQUIREMENT),
        },
        tests_by_requirement={"NFR-008": ["TS-037"]},
        parent_by_requirement={"NFR-008": "BR-006"},
    )

    result = hybrid_search(store, graph, "какие тесты про сессию")

    entity_ids = {e.id for e in result.entities}
    assert {"TS-037", "NFR-008", "BR-006"} <= entity_ids


def test_hybrid_search_returns_entities_with_kinds():
    store = MagicMock()
    store.search.return_value = [
        make_search_result("BR-001", Kind.BUSINESS_REQUIREMENT)
    ]

    graph = FakeGraph(
        nodes={
            "BR-001": make_entity("BR-001", Kind.BUSINESS_REQUIREMENT),
            "FR-001": make_entity("FR-001", Kind.FUNCTIONAL_REQUIREMENT),
        },
        children_by_br={"BR-001": ["FR-001"]},
    )

    result = hybrid_search(store, graph, "вход")

    kinds = {e.kind for e in result.entities}
    assert Kind.BUSINESS_REQUIREMENT in kinds
    assert Kind.FUNCTIONAL_REQUIREMENT in kinds
