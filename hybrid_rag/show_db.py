import chromadb
def main():
    # 1. Укажите точный путь к папке, где лежит ваша БД (например, './chroma_db')
    client = chromadb.PersistentClient(path="./chroma_db")

    # 2. Получите нужную коллекцию по её имени
    collection = client.get_collection(name="requirements")

    # 3. Извлеките все данные (по умолчанию эмбеддинги не загружаются для экономии памяти)
    result = collection.get(include=["embeddings", "documents", "metadatas"])

    # 4. Выведите результат
    print("Документы:", result['documents'])
    print("Метаданные:", result['metadatas'])
    print("ID записей:", result['ids'])
    print("Embeddings:", result['embeddings'])

if __name__ == "__main__":
    main()