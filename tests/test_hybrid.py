"""Tests for hybrid retrieval, mocking vector store and Neo4j session."""

from unittest.mock import MagicMock

from fakes import FakeSession

from hybrid_rag.hybrid import hybrid_search, naive_search
from hybrid_rag.vector import SearchResult


def make_search_result(doc_id: str, doc_type: str) -> SearchResult:
    return SearchResult(id=doc_id, text="text", doc_type=doc_type, score=0.8)


def test_naive_search_returns_vector_results_only():
    store = MagicMock()
    store.search.return_value = [make_search_result("BR-006", "business")]

    results = naive_search(store, "security")

    store.search.assert_called_once_with("security", k=5)
    assert [r.id for r in results] == ["BR-006"]


def test_hybrid_search_expands_requirement_seeds():
    store = MagicMock()
    store.search.return_value = [
        make_search_result("NFR-004", "nonfunctional"),
        make_search_result("BR-006", "business"),
    ]

    session = FakeSession(
        {
            "labels(n)[0] AS label": [
                {
                    "id": "NFR-004",
                    "title": "n",
                    "text": "i",
                    "label": "NonFunctionalRequirement",
                },
                {
                    "id": "BR-006",
                    "title": "b",
                    "text": "j",
                    "label": "BusinessRequirement",
                },
                {
                    "id": "FR-013",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                },
                {"id": "TS-031", "title": "t", "text": "x", "label": "TestScenario"},
                {"id": "TS-032", "title": "t", "text": "x", "label": "TestScenario"},
            ],
            "COVERS]->(req": [
                {"id": "TS-031", "title": "t", "text": "x"},
                {"id": "TS-032", "title": "t", "text": "x"},
            ],
            "RETURN br.id AS id": [{"id": "BR-006"}],
            "labels(child)[0] AS label": [
                {
                    "id": "FR-013",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                }
            ],
        }
    )

    result = hybrid_search(store, session, "секреты и шифрование")

    entity_ids = {e.id for e in result.entities}
    assert {"NFR-004", "BR-006", "FR-013", "TS-031", "TS-032"} <= entity_ids


def test_hybrid_search_expands_business_requirement_seed():
    store = MagicMock()
    store.search.return_value = [make_search_result("BR-003", "business")]

    session = FakeSession(
        {
            "labels(n)[0] AS label": [
                {
                    "id": "BR-003",
                    "title": "b",
                    "text": "j",
                    "label": "BusinessRequirement",
                },
                {
                    "id": "FR-005",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                },
                {"id": "TS-011", "title": "t", "text": "x", "label": "TestScenario"},
            ],
            "COVERS]->(req": [{"id": "TS-011", "title": "t", "text": "x"}],
            "labels(child)[0] AS label": [
                {
                    "id": "FR-005",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                }
            ],
        }
    )

    result = hybrid_search(store, session, "перевод средств")

    entity_ids = {e.id for e in result.entities}
    assert {"BR-003", "FR-005", "TS-011"} <= entity_ids


def test_hybrid_search_expands_test_seed_to_parents():
    store = MagicMock()
    store.search.return_value = [make_search_result("TS-037", "test")]

    session = FakeSession(
        {
            "COVERS]->(fr": [
                {
                    "id": "NFR-008",
                    "title": "n",
                    "text": "i",
                    "parent_id": "BR-006",
                    "parent_title": "b",
                }
            ],
            "labels(n)[0] AS label": [
                {"id": "TS-037", "title": "t", "text": "x", "label": "TestScenario"},
                {
                    "id": "NFR-008",
                    "title": "n",
                    "text": "i",
                    "label": "NonFunctionalRequirement",
                },
                {
                    "id": "BR-006",
                    "title": "b",
                    "text": "j",
                    "label": "BusinessRequirement",
                },
            ],
        }
    )

    result = hybrid_search(store, session, "какие тесты про сессию")

    entity_ids = {e.id for e in result.entities}
    assert {"TS-037", "NFR-008", "BR-006"} <= entity_ids


def test_hybrid_search_returns_entities_with_kinds():
    store = MagicMock()
    store.search.return_value = [make_search_result("BR-001", "business")]

    session = FakeSession(
        {
            "labels(n)[0] AS label": [
                {
                    "id": "BR-001",
                    "title": "b",
                    "text": "j",
                    "label": "BusinessRequirement",
                },
                {
                    "id": "FR-001",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                },
            ],
            "labels(child)[0] AS label": [
                {
                    "id": "FR-001",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                }
            ],
        }
    )

    result = hybrid_search(store, session, "вход")

    kinds = {e.kind for e in result.entities}
    assert "BusinessRequirement" in kinds
    assert "FunctionalRequirement" in kinds
