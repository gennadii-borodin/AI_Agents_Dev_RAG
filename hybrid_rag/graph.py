"""Neo4j graph module for requirement relationships."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from neo4j import Driver, GraphDatabase, Session

from hybrid_rag.documents import Document


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


_TYPE_TO_LABEL = {
    "business": "BusinessRequirement",
    "functional": "FunctionalRequirement",
    "nonfunctional": "NonFunctionalRequirement",
    "test": "TestScenario",
}


def build_graph(tx: Queryable, documents: list[Document]) -> None:
    for doc in documents:
        label = _TYPE_TO_LABEL[doc["type"]]
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


def get_related_tests(
    session: Queryable, requirement_ids: list[str]
) -> list[dict[str, str]]:
    result = session.run(
        """
        MATCH (ts:TestScenario)-[:COVERS]->(req)
        WHERE req.id IN $ids
        RETURN ts.id AS id, ts.title AS title, ts.text AS text
        ORDER BY ts.id
        """,
        ids=requirement_ids,
    )
    return [dict(record) for record in result]


def get_related_requirements(
    session: Queryable, test_ids: list[str]
) -> list[dict[str, str]]:
    result = session.run(
        """
        MATCH (ts:TestScenario)-[:COVERS]->(fr)
        WHERE ts.id IN $ids
        OPTIONAL MATCH (fr)-[:IMPLEMENTS]->(br:BusinessRequirement)
        RETURN DISTINCT
            fr.id AS id, fr.title AS title, fr.text AS text,
            br.id AS parent_id, br.title AS parent_title
        ORDER BY fr.id
        """,
        ids=test_ids,
    )
    return [dict(record) for record in result]


def get_children_of_br(session: Queryable, br_ids: list[str]) -> list[dict[str, str]]:
    result = session.run(
        """
        MATCH (child)-[:IMPLEMENTS]->(br:BusinessRequirement)
        WHERE br.id IN $ids
        RETURN child.id AS id, child.title AS title, child.text AS text,
               labels(child)[0] AS label
        ORDER BY child.id
        """,
        ids=br_ids,
    )
    return [dict(record) for record in result]
