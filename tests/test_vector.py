"""Tests for the vector store, with external clients mocked."""

import os
from unittest.mock import MagicMock, patch

import pytest

from hybrid_rag.documents import Document
from hybrid_rag.vector import VectorStore


def make_collection_mock():
    collection = MagicMock()
    collection.get.return_value = {"ids": []}
    return collection


def fake_embedding_response(model: str, input):
    return MagicMock(data=[MagicMock(embedding=[0.1] * 8) for _ in input])


@pytest.fixture
def store():
    with (
        patch("hybrid_rag.vector.chromadb.PersistentClient") as client_cls,
        patch("hybrid_rag.vector.OpenAI") as openai_cls,
        patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
    ):
        collection = make_collection_mock()
        client = MagicMock()
        client.get_or_create_collection.return_value = collection
        client_cls.return_value = client

        openai_client = MagicMock()
        openai_client.embeddings.create.side_effect = fake_embedding_response
        openai_cls.return_value = openai_client

        store = VectorStore()
        yield store, collection


def test_add_stores_document(store):
    vs, collection = store
    vs.add("BR-001", "text", "business")

    collection.add.assert_called_once()
    args = collection.add.call_args[1]
    assert args["ids"] == ["BR-001"]
    assert args["documents"] == ["text"]
    assert args["metadatas"] == [{"type": "business"}]


def test_add_all_batches_into_single_call(store):
    vs, collection = store
    docs: list[Document] = [
        {"id": "BR-001", "type": "business", "title": "t", "text": "a"},
        {
            "id": "FR-001",
            "type": "functional",
            "title": "t",
            "text": "b",
            "implements": "BR-001",
        },
    ]
    vs.add_all(docs)

    collection.add.assert_called_once()
    args = collection.add.call_args[1]
    assert args["ids"] == ["BR-001", "FR-001"]
    assert len(args["embeddings"]) == 2
    assert args["metadatas"] == [{"type": "business"}, {"type": "functional"}]


def test_search_returns_ranked_results(store):
    vs, collection = store
    collection.get.return_value = {"ids": ["BR-001", "BR-002"]}
    collection.query.return_value = {
        "ids": [["BR-001", "BR-002"]],
        "documents": [["text a", "text b"]],
        "distances": [[0.1, 0.3]],
        "metadatas": [[{"type": "business"}, {"type": "business"}]],
    }

    results = vs.search("balance")

    assert len(results) == 2
    assert results[0].id == "BR-001"
    assert results[0].score == 0.9
    assert results[1].id == "BR-002"
    assert results[1].score == 0.7
    assert results[0].doc_type == "business"


def test_search_empty_store_returns_empty(store):
    vs, _ = store
    vs._collection.get.return_value = {"ids": []}
    assert vs.search("anything") == []


def test_count_reflects_collection(store):
    vs, collection = store
    collection.get.return_value = {"ids": ["a", "b"]}
    assert vs.count() == 2


def test_clear_deletes_all(store):
    vs, collection = store
    collection.get.return_value = {"ids": ["a", "b"]}
    vs.clear()
    collection.delete.assert_called_once_with(ids=["a", "b"])
