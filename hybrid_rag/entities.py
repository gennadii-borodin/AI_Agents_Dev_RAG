"""Domain node types shared by the vector store and the knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass

from hybrid_rag.kind import Kind


@dataclass(frozen=True)
class Entity:
    id: str
    title: str
    text: str
    kind: Kind


@dataclass(frozen=True)
class CoveredRequirement:
    requirement: Entity
    parent: Entity | None
