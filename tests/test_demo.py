"""Tests for pure formatting helpers in the demo module."""

from hybrid_rag.demo import format_entities, short_kind
from hybrid_rag.entities import Entity
from hybrid_rag.kind import Kind


def test_short_kind_maps_kinds():
    assert short_kind(Kind.BUSINESS_REQUIREMENT) == "BR"
    assert short_kind(Kind.FUNCTIONAL_REQUIREMENT) == "FR"
    assert short_kind(Kind.NON_FUNCTIONAL_REQUIREMENT) == "NFR"
    assert short_kind(Kind.TEST_SCENARIO) == "TS"


def test_format_entities_renders_lines():
    entities = [
        Entity(
            id="BR-001",
            title="Авторизация",
            text="вход",
            kind=Kind.BUSINESS_REQUIREMENT,
        ),
        Entity(id="TS-031", title="Шифрование", text="TLS", kind=Kind.TEST_SCENARIO),
    ]

    rendered = format_entities(entities)

    assert "- [BR] BR-001: Авторизация — вход" in rendered
    assert "- [TS] TS-031: Шифрование — TLS" in rendered


def test_format_entities_empty():
    assert format_entities([]) == ""
