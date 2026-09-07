"""Self-contained ChromaDB vector store for requirement/test embeddings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import cast

import chromadb
import numpy as np
from chromadb.api.types import Metadata
from openai import OpenAI

from hybrid_rag.documents import Document

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PATH = "./hybrid_rag/chroma_db"
COLLECTION_NAME = "requirements"


@dataclass
class SearchResult:
    id: str
    text: str
    doc_type: str
    score: float


def _metadata(doc_type: str) -> Metadata:
    return {"type": doc_type}


def _read_type(metadata: Metadata) -> str:
    value = metadata["type"]
    if isinstance(value, str):
        return value
    raise ValueError(f"missing string 'type' metadata, got {value!r}")


@dataclass
class VectorStore:
    _collection: chromadb.Collection = field(init=False, repr=False)
    _openai: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collection = db.get_or_create_collection(name=COLLECTION_NAME)
        self._openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _embed(self, text: str) -> np.ndarray:
        response = self._openai.embeddings.create(model=EMBEDDING_MODEL, input=text)
        return np.asarray(response.data[0].embedding, dtype=np.float32)

    def add(self, doc_id: str, text: str, doc_type: str) -> None:
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[self._embed(text)],
            metadatas=[_metadata(doc_type)],
        )

    def add_all(self, documents: list[Document]) -> None:
        if not documents:
            return
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [_metadata(doc["type"]) for doc in documents]

        response = self._openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        embeddings = np.asarray(
            [item.embedding for item in response.data], dtype=np.float32
        )

        self._collection.add(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_embeddings=[self._embed(query)], n_results=k
        )

        ids = result["ids"] or []
        docs = result["documents"] or []
        distances = result["distances"] or []
        metadata_rows = result["metadatas"] or []
        if not ids:
            return []

        return [
            SearchResult(
                id=doc_id,
                text=doc,
                doc_type=_read_type(cast(Metadata, meta)),
                score=1.0 - dist,
            )
            for doc_id, doc, dist, meta in zip(
                ids[0], docs[0], distances[0], metadata_rows[0], strict=True
            )
        ]

    def count(self) -> int:
        return len(self._collection.get()["ids"])

    def clear(self) -> None:
        data = self._collection.get()
        if data["ids"]:
            self._collection.delete(ids=data["ids"])
