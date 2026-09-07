"""Tests for pure formatting helpers in the demo module."""

from hybrid_rag.demo import format_entities, short_kind
from hybrid_rag.entities import Entity


def test_short_kind_maps_labels():
    assert short_kind("BusinessRequirement") == "BR"
    assert short_kind("FunctionalRequirement") == "FR"
    assert short_kind("NonFunctionalRequirement") == "NFR"
    assert short_kind("TestScenario") == "TS"
    assert short_kind("Unknown") == "Unknown"


def test_format_entities_renders_lines():
    entities = [
        Entity(
            id="BR-001", title="Авторизация", text="вход", kind="BusinessRequirement"
        ),
        Entity(id="TS-031", title="Шифрование", text="TLS", kind="TestScenario"),
    ]

    rendered = format_entities(entities)

    assert "- [BR] BR-001: Авторизация — вход" in rendered
    assert "- [TS] TS-031: Шифрование — TLS" in rendered


def test_format_entities_empty():
    assert format_entities([]) == ""
