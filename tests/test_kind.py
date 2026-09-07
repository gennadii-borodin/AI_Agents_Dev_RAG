"""Tests for the Kind vocabulary: labels, doc types, short codes, parsing."""

import pytest

from hybrid_rag.kind import Kind


def test_kind_members_carry_label_doc_type_and_short():
    assert Kind.BUSINESS_REQUIREMENT.label == "BusinessRequirement"
    assert Kind.BUSINESS_REQUIREMENT.doc_type == "business"
    assert Kind.BUSINESS_REQUIREMENT.short == "BR"
    assert Kind.FUNCTIONAL_REQUIREMENT.short == "FR"
    assert Kind.NON_FUNCTIONAL_REQUIREMENT.short == "NFR"
    assert Kind.TEST_SCENARIO.short == "TS"
    assert Kind.TEST_SCENARIO.doc_type == "test"


def test_from_doc_type_round_trips():
    for kind in Kind:
        assert Kind.from_doc_type(kind.doc_type) is kind


def test_from_label_round_trips():
    for kind in Kind:
        assert Kind.from_label(kind.label) is kind


def test_from_unknown_doc_type_raises():
    with pytest.raises(ValueError):
        Kind.from_doc_type("unknown")


def test_from_unknown_label_raises():
    with pytest.raises(ValueError):
        Kind.from_label("UnknownNode")
