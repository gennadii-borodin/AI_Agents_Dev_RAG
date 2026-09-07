"""Tests for the Neo4j graph module with stubbed tx/session objects."""

from fakes import FakeSession, RecordingTx

from hybrid_rag import graph
from hybrid_rag.documents import DOCUMENTS


def test_build_graph_creates_one_node_per_document():
    tx = RecordingTx()
    graph.build_graph(tx, DOCUMENTS)

    create_queries = [q for q in tx.queries if "CREATE" in q]
    assert len(create_queries) == len(DOCUMENTS)


def test_build_graph_creates_implements_relationships():
    tx = RecordingTx()
    graph.build_graph(tx, DOCUMENTS)

    implements = [q for q in tx.queries if "IMPLEMENTS" in q]
    expected = sum(1 for d in DOCUMENTS if "implements" in d)
    assert len(implements) == expected


def test_build_graph_creates_covers_relationships():
    tx = RecordingTx()
    graph.build_graph(tx, DOCUMENTS)

    covers = [q for q in tx.queries if "COVERS" in q]
    expected = sum(1 for d in DOCUMENTS if "covers" in d)
    assert len(covers) == expected


def test_get_related_tests_returns_test_rows():
    session = FakeSession(
        {":COVERS]->(req": [{"id": "TS-001", "title": "t", "text": "x"}]}
    )
    rows = graph.get_related_tests(session, ["FR-001"])
    assert rows == [{"id": "TS-001", "title": "t", "text": "x"}]


def test_get_related_requirements_returns_parent_and_br():
    session = FakeSession(
        {
            ":COVERS]->(fr": [
                {
                    "id": "FR-001",
                    "title": "f",
                    "text": "y",
                    "parent_id": "BR-001",
                    "parent_title": "b",
                }
            ]
        }
    )
    rows = graph.get_related_requirements(session, ["TS-001"])
    assert rows[0]["id"] == "FR-001"
    assert rows[0]["parent_id"] == "BR-001"


def test_get_children_of_br_returns_children():
    session = FakeSession(
        {
            ":IMPLEMENTS]->(br": [
                {
                    "id": "FR-001",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                }
            ]
        }
    )
    rows = graph.get_children_of_br(session, ["BR-001"])
    assert rows[0]["id"] == "FR-001"
    assert rows[0]["label"] == "FunctionalRequirement"
