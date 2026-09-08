# Гибридный RAG по требованиям банковского приложения

Домашнее задание по курсу AI-агентов. Проект демонстрирует **гибридное извлечение
знаний (Hybrid RAG)**: семантический поиск по векторной базе (ChromaDB + OpenAI
compatible embeddings) дополняется обходом графа знаний (Neo4j) по связям
`IMPLEMENTS` и `COVERS`.

Корпус — **69 требований** банковского мобильного приложения на русском языке:
6 бизнес-требований (BR), 14 функциональных (FR), 10 нефункциональных (NFR) и
39 тестовых сценариев (TS), связанных в граф.

## 1. Что делает проект

Для каждого вопроса система извлекает релевантные требования двумя способами и
сравнивает их:

- **Naive** — только векторный поиск: топ-k результатов по семантической близости,
  без обхода графа;
- **Hybrid** — векторные «семена» расширяются по графу: требования дополняются
  покрывающими их тестами и родительскими бизнес-требованиями, тесты — покрываемыми
  требованиями и их родителями.

Гибридный контекст отправляется в chat-модель, которая формулирует ответ.
Наглядность результата: гибридная колонка таблицы сравнения заведомо шире за счёт
отношений графа.

**Стек:** Python 3.11+, OpenAI-compatible API (OpenAI / Ollama / vLLM / шлюзы),
ChromaDB, Neo4j 5 (Docker), rich, pytest / ruff / mypy.

---

## 2. Быстрый старт (для проверки)

Порядок действий — от получения кода до запуска демо. Предполагается Windows +
PowerShell; без `uv` работает и классический `pip` (см. шаг 1).

```powershell
# 1. Зависимости (uv — рекомендовано; или pip-fallback под блоком)
uv sync

# 2. Файл окружения: копируем шаблон и заполняем ключи
Copy-Item hybrid_rag/.env.example .env
#  -> откройте .env и впишите OPENAI_API_KEY (или OPENAI_BASE_URL для своего провайдера)

# 3. Neo4j в Docker
docker compose -f hybrid_rag/docker-compose.yml up -d

# 4. Быстрая самопроверка (не требует ни ключа, ни Neo4j)
pytest

# 5. Запуск демо
uv run -m hybrid_rag.demo
```

**Fallback без `uv`:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install pytest ruff mypy
```

**Ожидаемый результат** после шага 5:

```
knowledge graph ready
vector store ready: 69 documents
--------------------------------- Q: какой-то вопрос ---------------------------------
Query: какой-то вопрос
┌──────────────────────────────┬──────────────────────────────┐
│ Naive (vector only)          │ Hybrid (vector + graph)      │
│ [BR] BR-005: Push-уведомления │ [BR] BR-005: Push-уведомления │ ...
└──────────────────────────────┴──────────────────────────────┘
┌──────────────────────────────── Assistant ────────────────────┐
│ сформулированный LLM ответ ...                                │
└────────────────────────────────────────────────────────────────┘
```

> Демо идемпотентно: при каждом запуске стирает и пересобирает граф Neo4j и
> векторный индекс.

---

## 3. Настройка окружения

### 3.1. Файл `.env`

Копируется из `hybrid_rag/.env.example` в корень репозитория:

```ini
OPENAI_API_KEY=your-key-here

# URL OpenAI-совместимого API.
# Пустое значение = официальный endpoint OpenAI.
# Для кастомного провайдера укажите его base URL и имена моделей, например:
#   OPENAI_BASE_URL=http://localhost:11434/v1
#   OPENAI_CHAT_MODEL=qwen2.5:7b
#   OPENAI_EMBEDDING_MODEL=nomic-embed-text
OPENAI_BASE_URL=

# Имена моделей. Пустое значение = стандартные:
#   chat-модель:  gpt-4.1-mini
#   embedding:    text-embedding-3-small
OPENAI_CHAT_MODEL=
OPENAI_EMBEDDING_MODEL=

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=hybrid_rag_password
```

`.env` занесён в `.gitignore` — секреты не попадут в репозиторий. Демо загружает
`.env` из рабочего каталога и выше по дереву, поэтому файл корректно работает из
корня репозитория.

### 3.2. Neo4j

```powershell
docker compose -f hybrid_rag/docker-compose.yml up -d
```

- Bolt: `7687`, Neo4j Browser: `http://localhost:7474`
- Логин/пароль по умолчанию — `neo4j` / `hybrid_rag_password`
- Данные в Docker-томе `neo4j_data` (переживают перезапуск);
  сброс базы: `docker compose -f hybrid_rag/docker-compose.yml down -v`

Остановка: `docker compose -f hybrid_rag/docker-compose.yml down`

---

## 4. Структура репозитория

```
.
├── hybrid_rag/                 # основной пакет
│   ├── documents.py            # корпус требований (BR/FR/NFR/TS) и вопросы QUERIES
│   ├── entities.py             # доменные объекты: Entity, CoveredRequirement
│   ├── kind.py                 # виды узлов: enum Kind
│   ├── graph.py                # Neo4j: подключение, сборка графа, RequirementGraph
│   ├── retrieval.py            # извлечение: Retriever, NaiveRetriever, HybridRetriever
│   ├── vector.py               # VectorStore на ChromaDB + embeddings
│   ├── demo.py                 # демо: naive vs hybrid + ответ LLM
│   ├── show_db.py              # просмотр содержимого векторной коллекции (отладка)
│   ├── docker-compose.yml      # Neo4j 5
│   └── .env.example            # шаблон переменных окружения
│
├── data/                       # исходники корпуса требований в Markdown
├── tests/                      # pytest: документы, graph, kind, retrieval, vector, demo
├── pyproject.toml              # зависимости (uv/pip), конфиг ruff/mypy/pytest
└── uv.lock
```

---

## 5. Устройство кода `hybrid_rag`

**`Kind`** (`kind.py`) — enum видов узлов: `BUSINESS_REQUIREMENT`,
`FUNCTIONAL_REQUIREMENT`, `NON_FUNCTIONAL_REQUIREMENT`, `TEST_SCENARIO`. У каждого
члена — `label` (метка в Neo4j), `doc_type` (тип документа), `short` (короткое имя
для вывода). Строгие парсеры `from_doc_type` / `from_label` бросают `ValueError` на
неизвестное значение — новый тип узла добавляется в одном месте, опечатки выявляются
сразу.

**`Entity`** / **`CoveredRequirement`** (`entities.py`) — неизменяемые доменные
объекты, которыми оперируют все запросы к графу и поисковые стратегии.

**`RequirementGraph`** (`graph.py`) — обход графа Neo4j; весь Cypher инкапсулирован
внутри. Реализует протокол `GraphTraversal` (`get_nodes`, `get_related_tests`,
`get_children_of_br`, `get_parent_business_requirements`, `get_related_requirements`),
что позволяет тестировать логику на in-memory адаптере вместо живой БД.

**`VectorStore`** (`vector.py`) — ChromaDB-коллекция `requirements` (путь
`hybrid_rag/chroma_db`): `add`, `add_all`, `search(query, k)` → ранжированный
`list[SearchResult]`, `count`, `clear`.

**`Retriever`** (`retrieval.py`) — единая точка расширения
`retrieve(query, k) -> HybridResult`. Две реализации, обе получают `VectorStore` и
`GraphTraversal` через конструктор:

- `NaiveRetriever` — только векторный ранг, результаты «гидратируются» в `Entity`
  через граф без расширения соседей;
- `HybridRetriever` — векторные семена + расширение по `IMPLEMENTS`/`COVERS`.

Демо и тесты работают через один и тот же интерфейс.

---

## 6. Проверка проекта

```powershell
# 37 тестов, без внешних сервисов (клиенты замоканы)
pytest

# статическая типизация
mypy hybrid_rag tests

# линтер и формат
ruff check hybrid_rag tests
ruff format --check hybrid_rag tests
```

---

## 7. Что проверяется

| Аспект | Где в коде |
| --- | --- |
| Гибридное извлечение (вектор + граф) | `hybrid_rag/retrieval.py`, `hybrid_rag/graph.py` |
| Единый интерфейс двух стратегий | `Retriever` / `HybridResult` (`retrieval.py`) |
| Типизированный домен требований | `entities.py`, `kind.py` |
| Сборка и обход графа Neo4j | `graph.build_graph`, `RequirementGraph` |
| Векторный индекс и поиск | `vector.VectorStore` |
| Сравнение naive vs hybrid и ответ LLM | `hybrid_rag/demo.py` |
| Запуск демо | `python -m hybrid_rag.demo` |
| Тесты без внешних сервисов | `tests/` (mock/fake адаптеры) |

---

## 8. Устранение неполадок

- **`KeyError: 'OPENAI_API_KEY'`** — `.env` не найден в ожидаемом месте. Держите
  файл в корне репозитория и запускайте демо из корня.
- **Демо не подключается к Neo4j** — поднимите контейнер (`docker compose -f
  hybrid_rag/docker-compose.yml up -d`) и сверьте `NEO4J_*` в `.env`.
- **`Model '...' not found`** — провайдер не знает запрошенную модель. Проверьте
  `OPENAI_CHAT_MODEL` / `OPENAI_EMBEDDING_MODEL`: имена должны совпадать с теми, что
  отдаёт ваш провайдер (у шлюзов часто формат `provider/model`).
- **`test_queries_are_defined` падает** — тест проверяет инварианты списка
  `QUERIES`; если вы меняете список вопросов в `documents.py`, синхронизируйте
  ожидания в `tests/test_documents.py`.
- **Устаревший векторный индекс** — эмбеддинги хранятся локально. После смены
  модели или перевода корпуса удалите `hybrid_rag/chroma_db/` и перезапустите демо
  (оно пересоберёт индекс).