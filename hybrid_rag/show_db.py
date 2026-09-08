"""Print the contents of the ChromaDB requirements collection."""

from chromadb import PersistentClient

CHROMA_PATH = "./hybrid_rag/chroma_db"
COLLECTION_NAME = "requirements"


def main() -> None:
    client = PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    result = collection.get(include=["embeddings", "documents", "metadatas"])

    print("ID записей:", result["ids"])
    print("Метаданные:", result["metadatas"])
    print("Документы:", result["documents"])
    print("Embeddings:", result["embeddings"])


if __name__ == "__main__":
    main()
