import chromadb
def main():

    client = chromadb.PersistentClient(path="./chroma_db")

    collection = client.get_collection(name="requirements")

    result = collection.get(include=["embeddings", "documents", "metadatas"])

    # 4. Выведите результат
    print("Документы:", result['documents'])
    print("Метаданные:", result['metadatas'])
    print("ID записей:", result['ids'])
    print("Embeddings:", result['embeddings'])

if __name__ == "__main__":
    main()