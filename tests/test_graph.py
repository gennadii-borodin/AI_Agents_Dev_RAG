"""Tests for the Neo4j graph module with stubbed tx/session objects."""

from fakes import FakeSession, RecordingTx

from hybrid_rag import graph
from hybrid_rag.documents import DOCUMENTS
from hybrid_rag.entities import Entity
from hybrid_rag.graph import RequirementGraph
from hybrid_rag.kind import Kind


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


def test_get_nodes_returns_entities():
    session = FakeSession(
        {
            "labels(n)[0] AS label": [
                {"id": "TS-031", "title": "t", "text": "x", "label": "TestScenario"},
                {
                    "id": "FR-013",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                },
            ]
        }
    )

    nodes = RequirementGraph(session).get_nodes(["TS-031", "FR-013"])

    assert nodes == [
        Entity(id="TS-031", title="t", text="x", kind=Kind.TEST_SCENARIO),
        Entity(id="FR-013", title="f", text="y", kind=Kind.FUNCTIONAL_REQUIREMENT),
    ]


def test_get_nodes_empty_ids_returns_empty():
    session = FakeSession({})
    assert RequirementGraph(session).get_nodes([]) == []


def test_get_related_tests_returns_entities():
    session = FakeSession(
        {
            "COVERS]->(req": [
                {"id": "TS-001", "title": "t", "text": "x"},
                {"id": "TS-002", "title": "t", "text": "x"},
            ]
        }
    )

    tests = RequirementGraph(session).get_related_tests(["FR-001"])

    assert tests == [
        Entity(id="TS-001", title="t", text="x", kind=Kind.TEST_SCENARIO),
        Entity(id="TS-002", title="t", text="x", kind=Kind.TEST_SCENARIO),
    ]


def test_get_children_of_br_returns_entities():
    session = FakeSession(
        {
            "labels(child)[0] AS label": [
                {
                    "id": "FR-001",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                }
            ]
        }
    )

    children = RequirementGraph(session).get_children_of_br(["BR-001"])

    assert children == [
        Entity(id="FR-001", title="f", text="y", kind=Kind.FUNCTIONAL_REQUIREMENT)
    ]


def test_get_parent_business_requirements_returns_entities():
    session = FakeSession(
        {
            "IMPLEMENTS]->(br": [
                {"id": "BR-001", "title": "b", "text": "z"},
            ]
        }
    )

    parents = RequirementGraph(session).get_parent_business_requirements(["FR-001"])

    assert parents == [
        Entity(id="BR-001", title="b", text="z", kind=Kind.BUSINESS_REQUIREMENT)
    ]


def test_get_related_requirements_returns_covered_pairs():
    session = FakeSession(
        {
            "COVERS]->(fr": [
                {
                    "id": "FR-001",
                    "title": "f",
                    "text": "y",
                    "label": "FunctionalRequirement",
                    "parent_id": "BR-001",
                    "parent_title": "b",
                    "parent_text": "z",
                },
                {
                    "id": "NFR-001",
                    "title": "n",
                    "text": "w",
                    "label": "NonFunctionalRequirement",
                    "parent_id": None,
                    "parent_title": None,
                    "parent_text": None,
                },
            ]
        }
    )

    covered = RequirementGraph(session).get_related_requirements(["TS-001", "TS-002"])

    assert len(covered) == 2
    assert covered[0].requirement == Entity(
        id="FR-001", title="f", text="y", kind=Kind.FUNCTIONAL_REQUIREMENT
    )
    assert covered[0].parent == Entity(
        id="BR-001", title="b", text="z", kind=Kind.BUSINESS_REQUIREMENT
    )
    assert covered[1].requirement == Entity(
        id="NFR-001", title="n", text="w", kind=Kind.NON_FUNCTIONAL_REQUIREMENT
    )
    assert covered[1].parent is None
