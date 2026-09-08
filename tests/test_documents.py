"""Test the document corpus integrity."""

from hybrid_rag.documents import DOCUMENTS, QUERIES


def count_by_type(type_name: str) -> int:
    return sum(1 for d in DOCUMENTS if d["type"] == type_name)


def test_counts_match_sample_corpus():
    assert count_by_type("business") == 6
    assert count_by_type("functional") == 14
    assert count_by_type("nonfunctional") == 10
    assert count_by_type("test") == 39


def test_all_implements_references_point_to_existing_brs():
    business_ids = {d["id"] for d in DOCUMENTS if d["type"] == "business"}
    for d in DOCUMENTS:
        if "implements" in d:
            assert d["implements"] in business_ids


def test_all_covers_references_point_to_existing_fr_or_nfr():
    requirement_ids = {
        d["id"] for d in DOCUMENTS if d["type"] in ("functional", "nonfunctional")
    }
    for d in DOCUMENTS:
        if "covers" in d:
            assert d["covers"] in requirement_ids


def test_every_functional_requirement_has_implements():
    for d in DOCUMENTS:
        if d["type"] in ("functional", "nonfunctional"):
            assert "implements" in d


def test_every_test_scenario_has_covers():
    for d in DOCUMENTS:
        if d["type"] == "test":
            assert "covers" in d


def test_queries_are_defined():
    assert QUERIES
    for q in QUERIES:
        assert "question" in q
        assert q["question"]


def test_ids_are_unique():
    ids = [d["id"] for d in DOCUMENTS]
    assert len(ids) == len(set(ids))
