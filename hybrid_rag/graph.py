"""Neo4j graph module for requirement relationships."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from neo4j import Driver, GraphDatabase, Session

from hybrid_rag.documents import Document
from hybrid_rag.entities import CoveredRequirement, Entity
from hybrid_rag.kind import Kind


class Queryable(Protocol):
    def run(self, query: str, **parameters: Any) -> Any: ...


@dataclass
class GraphConnection:
    driver: Driver
    session: Session

    def __enter__(self) -> GraphConnection:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        close(self)


def connect() -> GraphConnection:
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]

    driver = GraphDatabase.driver(uri, auth=(username, password))
    session = driver.session()
    return GraphConnection(driver=driver, session=session)


def close(conn: GraphConnection) -> None:
    conn.session.close()
    conn.driver.close()


def erase_graph(tx: Queryable) -> None:
    tx.run("MATCH (n) DETACH DELETE n")


def build_graph(tx: Queryable, documents: list[Document]) -> None:
    for doc in documents:
        label = Kind.from_doc_type(doc["type"]).label
        tx.run(
            f"CREATE (n:{label} {{id: $id, title: $title, text: $text}})",
            id=doc["id"],
            title=doc["title"],
            text=doc["text"],
        )

    for doc in documents:
        if "implements" in doc:
            tx.run(
                """
                MATCH (src {id: $src_id})
                MATCH (dst:BusinessRequirement {id: $dst_id})
                MERGE (src)-[:IMPLEMENTS]->(dst)
                """,
                src_id=doc["id"],
                dst_id=doc["implements"],
            )
        if "covers" in doc:
            tx.run(
                """
                MATCH (src:TestScenario {id: $src_id})
                MATCH (dst {id: $dst_id})
                WHERE dst:FunctionalRequirement OR dst:NonFunctionalRequirement
                MERGE (src)-[:COVERS]->(dst)
                """,
                src_id=doc["id"],
                dst_id=doc["covers"],
            )


class GraphTraversal(Protocol):
    """Traversal over the requirement knowledge graph, returning domain entities."""

    def get_nodes(self, ids: list[str]) -> list[Entity]: ...

    def get_related_tests(self, requirement_ids: list[str]) -> list[Entity]: ...

    def get_children_of_br(self, br_ids: list[str]) -> list[Entity]: ...

    def get_parent_business_requirements(
        self, requirement_ids: list[str]
    ) -> list[Entity]: ...

    def get_related_requirements(
        self, test_ids: list[str]
    ) -> list[CoveredRequirement]: ...


class RequirementGraph(GraphTraversal):
    """Neo4j-backed traversal adapter returning domain entities."""

    def __init__(self, session: Queryable) -> None:
        self._session = session

    def get_nodes(self, ids: list[str]) -> list[Entity]:
        if not ids:
            return []
        rows = self._session.run(
            "MATCH (n) WHERE n.id IN $ids "
            "RETURN n.id AS id, n.title AS title, n.text AS text, "
            "labels(n)[0] AS label ORDER BY n.id",
            ids=ids,
        )
        return [
            Entity(
                id=r["id"],
                title=r["title"],
                text=r["text"],
                kind=Kind.from_label(r["label"]),
            )
            for r in rows
        ]

    def get_related_tests(self, requirement_ids: list[str]) -> list[Entity]:
        if not requirement_ids:
            return []
        rows = self._session.run(
            """
            MATCH (ts:TestScenario)-[:COVERS]->(req)
            WHERE req.id IN $ids
            RETURN ts.id AS id, ts.title AS title, ts.text AS text
            ORDER BY ts.id
            """,
            ids=requirement_ids,
        )
        return [
            Entity(
                id=r["id"],
                title=r["title"],
                text=r["text"],
                kind=Kind.TEST_SCENARIO,
            )
            for r in rows
        ]

    def get_children_of_br(self, br_ids: list[str]) -> list[Entity]:
        if not br_ids:
            return []
        rows = self._session.run(
            """
            MATCH (child)-[:IMPLEMENTS]->(br:BusinessRequirement)
            WHERE br.id IN $ids
            RETURN child.id AS id, child.title AS title, child.text AS text,
                   labels(child)[0] AS label
            ORDER BY child.id
            """,
            ids=br_ids,
        )
        return [
            Entity(
                id=r["id"],
                title=r["title"],
                text=r["text"],
                kind=Kind.from_label(r["label"]),
            )
            for r in rows
        ]

    def get_parent_business_requirements(
        self, requirement_ids: list[str]
    ) -> list[Entity]:
        if not requirement_ids:
            return []
        rows = self._session.run(
            """
            MATCH (req)-[:IMPLEMENTS]->(br:BusinessRequirement)
            WHERE req.id IN $ids
            RETURN DISTINCT br.id AS id, br.title AS title, br.text AS text
            ORDER BY br.id
            """,
            ids=requirement_ids,
        )
        return [
            Entity(
                id=r["id"],
                title=r["title"],
                text=r["text"],
                kind=Kind.BUSINESS_REQUIREMENT,
            )
            for r in rows
        ]

    def get_related_requirements(self, test_ids: list[str]) -> list[CoveredRequirement]:
        if not test_ids:
            return []
        rows = self._session.run(
            """
            MATCH (ts:TestScenario)-[:COVERS]->(fr)
            WHERE ts.id IN $ids
            OPTIONAL MATCH (fr)-[:IMPLEMENTS]->(br:BusinessRequirement)
            RETURN DISTINCT
                fr.id AS id, fr.title AS title, fr.text AS text,
                labels(fr)[0] AS label,
                br.id AS parent_id, br.title AS parent_title, br.text AS parent_text
            ORDER BY fr.id
            """,
            ids=test_ids,
        )
        return [
            CoveredRequirement(
                requirement=Entity(
                    id=r["id"],
                    title=r["title"],
                    text=r["text"],
                    kind=Kind.from_label(r["label"]),
                ),
                parent=(
                    Entity(
                        id=r["parent_id"],
                        title=r["parent_title"],
                        text=r["parent_text"],
                        kind=Kind.BUSINESS_REQUIREMENT,
                    )
                    if r["parent_id"]
                    else None
                ),
            )
            for r in rows
        ]
