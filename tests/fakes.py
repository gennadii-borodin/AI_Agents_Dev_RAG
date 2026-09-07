"""Shared fakes for graph and hybrid tests."""


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
