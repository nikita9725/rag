# RAG Service

Учебный RAG-сервис, который загружает локальные документы, разбивает их на чанки,
строит локальные embeddings и синхронизирует векторный индекс в Weaviate.
По пользовательскому вопросу возвращает контекст через semantic или hybrid retrieval.

## Pipeline

Индексация оформлена как последовательность явных шагов:

1. `load` — чтение и очистка UTF-8 `.txt`-файлов;
2. `chunk` — разбиение с настраиваемыми размером и overlap;
3. `embed` — построение embeddings локальной моделью;
4. `sync` — полная синхронизация коллекции Weaviate;
5. `verify` — проверка количества объектов и контрольный vector search.

Embedding provider и repository задаются интерфейсами, поэтому в unit-тестах они
заменяются стабами.

## Установка и локальная модель

Требуются `uv`, Python 3.14 и Docker Compose.

```shell
uv sync
cp .env.example .env
```

Модель `intfloat/multilingual-e5-small` сохраняется в
`models/multilingual-e5-small`. Это полноценный локальный snapshot, а не внешний
runtime-сервис. Каталог исключён из Git из-за размера модели. После первой загрузки
индексация работает без повторного обращения к Hugging Face.

Отдельная команда загрузки не требуется: первый `uv run rag-kb` автоматически
скачает модель, а следующие запуски сразу используют локальный каталог.

Документы кодируются с префиксом `passage:`, поисковые запросы — с `query:`.
Нормализованные векторы имеют размерность 384.

## Запуск Weaviate

```shell
docker compose up -d
docker compose ps
```

Compose поднимает:

- Weaviate REST API: `http://localhost:8080`;
- Weaviate gRPC: `localhost:50051`;
- локальный Weaviate UI: `http://localhost:7777`.

Данные БД сохраняются в Docker volume `weaviate_data`.

## Индексация

```shell
uv run rag-kb
```

Другую папку и параметры chunking можно передать явно:

```shell
uv run rag-kb path/to/documents --chunk-size 300 --chunk-overlap 60
```

Коллекция `KnowledgeChunk` содержит свойства `document_id`, `source_name`,
`chunk_id`, `text` и supplied vector. Стабильный UUID строится из позиции чанка,
поэтому повторный запуск обновляет объекты без дублей. Чанки, которых больше нет в
текущей базе знаний, удаляются только после успешной записи актуального набора.

Настройки находятся в `.env`:

```dotenv
EMBEDDING_MODEL_ID=intfloat/multilingual-e5-small
EMBEDDING_MODEL_REVISION=614241f622f53c4eeff9890bdc4f31cfecc418b3
EMBEDDING_MODEL_PATH=models/multilingual-e5-small
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32

WEAVIATE_URL=http://localhost:8080
WEAVIATE_GRPC_PORT=50051
WEAVIATE_COLLECTION=KnowledgeChunk
WEAVIATE_INTEGRATION_COLLECTION=KnowledgeChunkIntegration
WEAVIATE_E2E_COLLECTION=KnowledgeChunkE2E
```

## Поиск контекста — день 4

После запуска Weaviate и индексации можно передать вопрос отдельной команде:

```shell
uv run rag-query "Зачем соседние чанки частично перекрываются?" --top-k 3
uv run rag-query "Зачем соседние чанки частично перекрываются?" --top-k 3 --mode hybrid --alpha 0.5
```

По умолчанию используются semantic search и top-k 3. Выдача содержит полный текст
каждого чанка, его позицию, `source_name`, `chunk_id` и метрику. Например:

```text
1. source_name=02_chunking.txt chunk_id=0 distance=0.136178
Разбиение текста на чанки
...
```

В semantic режиме выводится cosine distance: меньше — ближе к вопросу. Hybrid
объединяет vector search и BM25 по свойству `text` через relative score fusion;
выводится score: больше — выше в выдаче. Метрики разных режимов не сравниваются
напрямую и не являются вероятностью правильного ответа. `--alpha` доступен только
для hybrid: 0 означает только BM25, 1 — только vector search, по умолчанию 0.5.

Поиск использует ту же локальную модель с префиксом `query:`. Он не изменяет индекс
и не создаёт отсутствующую коллекцию: сначала выполните `uv run rag-kb`. Пустой
вопрос, неположительный top-k и alpha вне `[0, 1]` отклоняются до загрузки модели.
Пустая выдача сопровождается сообщением, ошибки подключения завершают CLI с
ненулевым кодом. Порог релевантности пока не применяется, поэтому даже для вопроса
вне базы могут вернуться тематически слабые чанки.

Python-интерфейс для следующих этапов:

```python
from rag_service.retrieval import RetrievalService

results = RetrievalService(provider, repository).retrieve(
    "Как top-k влияет на контекст?", top_k=3, mode="hybrid", alpha=0.5
)
```

Каждый результат — `ChunkSearchResult`: `content`, `metadata`, `uuid`,
`distance` и `score`. Для semantic заполнен `distance`, для hybrid — `score`;
другая метрика равна `None`. Порядок Weaviate сохраняется.

Пять проверочных вопросов с ожидаемыми фактами находятся в
[`tests/data/retrieval_questions.json`](tests/data/retrieval_questions.json).
[Отчёт проверки](docs/day04-retrieval.md) содержит фактические top-3 обоих режимов,
полные тексты чанков и оценку релевантности. E2E-тест повторяет эти десять поисков
в отдельной тестовой коллекции.

## Тесты

Unit-тесты не требуют модели, сети или Docker:

```shell
uv run pytest tests/unit
```

Integration и e2e используют настоящий локальный snapshot модели и реальный
Weaviate. Если локальной модели ещё нет, первый запуск скачает её автоматически.

```shell
docker compose up -d
uv run pytest -m integration tests/integration
uv run pytest -m e2e tests/e2e
```

Статические проверки:

```shell
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Локальная отладка

Полный запуск с чистого checkout:

```shell
uv sync
test -f .env || cp .env.example .env
docker compose up -d
docker compose ps
uv run rag-kb
```

Первый запуск `rag-kb` без загруженной модели автоматически скачает snapshot в
`EMBEDDING_MODEL_PATH`. Загрузка модели не требует отдельного CLI-аргумента.

Логи контейнеров и проверки API:

```shell
docker compose logs -f weaviate
curl http://localhost:8080/v1/.well-known/ready
curl http://localhost:8080/v1/schema
```

Запуск одного теста и отладчика Python:

```shell
uv run pytest tests/unit/test_pipeline.py -vv
uv run python -m pdb -m rag_service
```

Обычная остановка сохраняет данные в Docker volume:

```shell
docker compose stop
```

Полное удаление контейнеров вместе с локальными данными Weaviate — деструктивная
операция:

```shell
docker compose down -v
```

## Проверка через веб-интерфейс

1. Запустите `docker compose up -d` и выполните `uv run rag-kb`.
2. Откройте `http://localhost:7777`.
3. Выберите коллекцию `KnowledgeChunk`.
4. Проверьте UUID, `document_id`, `source_name`, `chunk_id`, `text` и количество
   объектов.
5. Повторно выполните `uv run rag-kb`: количество объектов не должно увеличиться.

Готовность самой БД можно проверить в браузере по адресу
`http://localhost:8080/v1/.well-known/ready`.

## Структура

```text
knowledge_base/               исходные TXT-документы
models/                       локально сохранённая embedding-модель
src/rag_service/
  embeddings.py              model manager и локальный embedding provider
  pipeline.py                явные шаги индексации
  retrieval.py               сервис semantic/hybrid retrieval
  query_cli.py               CLI поиска контекста
  repository.py              Weaviate repository
  interfaces.py              стратегии provider/repository
  loader.py, chunker.py       подготовка документов
tests/unit/                   тесты со стабами
tests/integration/            модель + настоящий Weaviate
tests/e2e/                    полный production pipeline
```
