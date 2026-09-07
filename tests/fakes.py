"""Shared fakes for graph, hybrid, and retrieval tests."""

from hybrid_rag.entities import CoveredRequirement, Entity


class RecordingTx:
    def __init__(self):
        self.queries: list[str] = []

    def run(self, query: str, **params):
        self.queries.append(query)


class FakeResult:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __iter__(self):
        return iter([dict(row) for row in self.rows])


class FakeSession:
    def __init__(self, queries: dict[str, list[dict]]):
        self.queries = queries

    def run(self, query: str, **params):
        if params.get("ids") == []:
            return FakeResult([])
        for needle, rows in self.queries.items():
            if needle in query:
                return FakeResult(rows)
        return FakeResult([])


class FakeGraph:
    """In-memory adapter for the GraphTraversal seam, used by tests."""

    def __init__(
        self,
        nodes: dict[str, Entity],
        tests_by_requirement: dict[str, list[str]] | None = None,
        children_by_br: dict[str, list[str]] | None = None,
        parent_by_requirement: dict[str, str] | None = None,
    ) -> None:
        self._nodes = nodes
        self._tests = tests_by_requirement or {}
        self._children = children_by_br or {}
        self._parent = parent_by_requirement or {}
        self._covered_by_test = {
            test_id: req_id
            for req_id, test_ids in self._tests.items()
            for test_id in test_ids
        }

    def get_nodes(self, ids: list[str]) -> list[Entity]:
        return [self._nodes[doc_id] for doc_id in ids if doc_id in self._nodes]

    def get_related_tests(self, requirement_ids: list[str]) -> list[Entity]:
        return [
            entity
            for req_id in requirement_ids
            for test_id in self._tests.get(req_id, [])
            for entity in [self._nodes.get(test_id)]
            if entity is not None
        ]

    def get_children_of_br(self, br_ids: list[str]) -> list[Entity]:
        return [
            entity
            for br_id in br_ids
            for child_id in self._children.get(br_id, [])
            for entity in [self._nodes.get(child_id)]
            if entity is not None
        ]

    def get_parent_business_requirements(
        self, requirement_ids: list[str]
    ) -> list[Entity]:
        return [
            self._nodes[parent_id]
            for req_id in requirement_ids
            for parent_id in [self._parent.get(req_id)]
            if parent_id in self._nodes
        ]

    def get_related_requirements(self, test_ids: list[str]) -> list[CoveredRequirement]:
        covered: list[CoveredRequirement] = []
        for test_id in test_ids:
            req_id = self._covered_by_test.get(test_id)
            req = None
            if req_id:
                req = self._nodes.get(req_id)
            if req is None:
                continue
            parent = None
            parent_id = self._parent.get(req.id)
            if parent_id:
                parent = self._nodes[parent_id]
            covered.append(CoveredRequirement(requirement=req, parent=parent))
        return covered
