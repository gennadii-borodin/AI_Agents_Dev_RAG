"""Single vocabulary for the node kinds in the requirements knowledge base."""

from __future__ import annotations

from enum import Enum


class Kind(Enum):
    """A requirement/test node kind with its graph label, doc type, and short code."""

    BUSINESS_REQUIREMENT = ("BusinessRequirement", "business", "BR")
    FUNCTIONAL_REQUIREMENT = ("FunctionalRequirement", "functional", "FR")
    NON_FUNCTIONAL_REQUIREMENT = ("NonFunctionalRequirement", "nonfunctional", "NFR")
    TEST_SCENARIO = ("TestScenario", "test", "TS")

    def __init__(self, label: str, doc_type: str, short: str) -> None:
        self.label = label
        self.doc_type = doc_type
        self.short = short

    @classmethod
    def from_doc_type(cls, doc_type: str) -> Kind:
        try:
            return _DOC_TYPE_TO_KIND[doc_type]
        except KeyError:
            raise ValueError(f"unknown doc_type {doc_type!r}") from None

    @classmethod
    def from_label(cls, label: str) -> Kind:
        try:
            return _LABEL_TO_KIND[label]
        except KeyError:
            raise ValueError(f"unknown graph label {label!r}") from None


_DOC_TYPE_TO_KIND = {kind.doc_type: kind for kind in Kind}
_LABEL_TO_KIND = {kind.label: kind for kind in Kind}
