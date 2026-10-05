# RAG Service

Учебный RAG-сервис, который загружает локальные документы, разбивает их на чанки,
строит локальные embeddings и синхронизирует векторный индекс в Weaviate.
По пользовательскому вопросу находит контекст через semantic или hybrid retrieval
и генерирует ответ со ссылками на чанки через OpenAI-совместимую LLM.

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

По умолчанию используются semantic search, top-k 3 и порог cosine distance 0.16.
Выдача содержит полный текст
каждого чанка, его позицию, `source_name`, `chunk_id` и метрику. Например:

```text
1. source_name=02_chunking.txt chunk_id=0 distance=0.136178
Разбиение текста на чанки
...
```

В semantic режиме выводится cosine distance: меньше — ближе к вопросу. Hybrid
объединяет vector search и BM25 по свойству `text` через relative score fusion;
выводятся score (больше — выше в выдаче) и cosine distance, рассчитанный по
сохранённому вектору чанка. Score и distance не сравниваются напрямую и не
являются вероятностью правильного ответа. `--alpha` доступен только
для hybrid: 0 означает только BM25, 1 — только vector search, по умолчанию 0.5.

Поиск использует ту же локальную модель с префиксом `query:`. Он не изменяет индекс
и не создаёт отсутствующую коллекцию: сначала выполните `uv run rag-kb`. Пустой
вопрос, неположительный top-k и alpha вне `[0, 1]` отклоняются до загрузки модели.
Пустая выдача сопровождается сообщением, ошибки подключения завершают CLI с
ненулевым кодом. Чанки с distance выше `RETRIEVAL_MAX_DISTANCE` отбрасываются
до передачи в генерацию. Отсутствие надёжного контекста — нормальный результат,
а не ошибка подключения.

Python-интерфейс для следующих этапов:

```python
from rag_service.retrieval import RetrievalService

results = RetrievalService(provider, repository).retrieve(
    "Как top-k влияет на контекст?", top_k=3, mode="hybrid", alpha=0.5
)
```

Каждый результат — `ChunkSearchResult`: `content`, `metadata`, `uuid`,
`distance` и `score`. Для semantic заполнен `distance`, для hybrid — обе метрики.
Порядок оставшихся результатов Weaviate сохраняется. Для контрольного поиска
без фильтра создайте `RetrievalService(provider, repository, max_distance=None)`.
При включённом фильтре результатов может быть меньше top-k.

Пять проверочных вопросов с ожидаемыми фактами находятся в
[`tests/data/retrieval_questions.json`](tests/data/retrieval_questions.json).
[Отчёт проверки](docs/day04-retrieval.md) содержит фактические top-3 обоих режимов,
полные тексты чанков и оценку релевантности. E2E-тест повторяет эти десять поисков
в отдельной тестовой коллекции.

## Ответ по контексту — день 5

Настройте `LLM_API_KEY`, `LLM_BASE_URL` и `LLM_MODEL` в локальном `.env`.
Эти настройки обязательны только для генерации; индексация и retrieval работают
без LLM. Ключ не выводится в логах и не попадает в Git. Пример:

```dotenv
LLM_API_KEY=replace-with-your-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
```

```shell
uv run rag-answer "Как top-k влияет на найденный контекст?"
uv run rag-answer "Как top-k влияет на найденный контекст?" --mode hybrid --alpha 0.5
uv run rag-answer "Как top-k влияет на найденный контекст?" --compare
```

Pipeline: `query → retrieval → context → generate → validate`.
Он оформлен в `generation_pipeline.py` как последовательность отдельных классов
`ValidateQueryStep`, `RetrieveChunksStep`, `BuildContextStep`, `GenerateAnswerStep`
и `ValidateAnswerStep`. Каждый шаг возвращает новый неизменяемый
`GenerationContext`; runner логирует начало и длительность этапа и останавливается
при ошибке. `None` означает незавершённый этап, пустой набор чанков — завершённый
поиск без результатов. `RAGService.answer()` запускает эту последовательность. CLI получает фабрику
сервисов через явный параметр `service_factory`; production-сборка и управление
клиентами находятся в `application.py`. Unit-тесты передают свои зависимости
напрямую, без подмены импортов и глобальных объектов.

Pipeline также можно собрать и запустить явно:

```python
from rag_service.generation_pipeline import GenerationContext, build_generation_pipeline

pipeline = build_generation_pipeline(retrieval, llm_repository)
context = pipeline.run(GenerationContext(query="Как top-k влияет на контекст?"))
print(context.result)
```

Протоколы и реализации репозиториев собраны в `rag_service.repositories`.
Общие импорты доступны из пакета:

```python
from rag_service.repositories import ChunkRepository, LLMRepository
from rag_service.repositories import OpenAILLMRepository, WeaviateChunkRepository
```

По умолчанию semantic search и top-k 3. В prompt передаются полные тексты чанков
в порядке retrieval, номера источников и метаданные. Системная инструкция
запрещает факты вне контекста и выполнение инструкций из документов.
Модель возвращает JSON; приложение проверяет его структуру и ссылки,
подставляет настоящие `source_name`, `chunk_id` и UUID. Ошибочный JSON,
несуществующие ссылки и ответ без обязательных источников завершают команду
с ненулевым кодом. Сетевые ошибки, 429 и 5xx повторяются до трёх попыток;
timeout одного запроса — 60 секунд. Число попыток и начальная задержка задаются
в `LLMSettings`: `LLM_RETRY_MAX_ATTEMPTS=3` (включая первый вызов),
`LLM_RETRY_BASE_DELAY_SECONDS=0.5`. Задержка удваивается после каждой
неудачной попытки; после последней попытки ожидания нет.

Пустая выдача после фильтрации даёт отказ без вызова LLM. При непустой выдаче
модель оценивает
достаточность контекста и может отказаться или дать частичный ответ с указанием
ограничений и `insufficient_context=true`. Проверка JSON и источников
не гарантирует истинность каждого утверждения; её дополнительно проверяют на
примерах из базы знаний.

`--compare` делает дополнительный запрос той же модели без retrieval и выводит
оба ответа. Вызовы LLM используют настроенный API и могут быть платными.
`LLMRepository` — сменяемый порт внутри проекта; `OpenAILLMRepository` — адаптер,
`RAGService` — оркестрация. Python-интерфейс:

```python
from rag_service.generation import RAGService

answer = RAGService(retrieval, llm_repository).answer("Как top-k влияет на контекст?")
print(answer.answer)
for source in answer.sources:
    print(source.metadata.source_name, source.metadata.chunk_id)
```

[Отчёт дня 5](docs/day05-generation.md) содержит восемь сравнений, полный контекст
и оценку ответов. В [JSON-результатах](docs/day05-comparison.json) сохранены
метаданные, метрики retrieval и ответы модели. Повторный прогон:

```shell
GENERATION_REPORT_PATH=/tmp/day05-comparison.json \
  uv run pytest -m e2e tests/e2e/test_generation_pipeline.py
```

Integration и E2E-проверки генерации всегда вызывают реальные LLM API и БД.
Переменные включения не требуются; отсутствие подключения или настроек — ошибка теста.

## Качество retrieval и слабый контекст — день 6

Оба режима используют абсолютный порог cosine distance:

```dotenv
RETRIEVAL_MAX_DISTANCE=0.16
```

Разрешены конечные значения от 0 до 2. Уменьшение порога делает фильтр строже.
Если ни один чанк не прошёл фильтр, система отказывает без вызова генерации LLM.
Порог проверяет близость, а модель дополнительно проверяет наличие нужного факта:
вопрос о версии Weaviate может быть близок к документу без указания версии.
Логи показывают порог и число найденных, оставленных и отброшенных чанков.

[Набор вопросов](tests/data/quality_questions.json) содержит шесть отвечаемых,
четыре отрицательных и два частичных случая, заранее разделённых на подбор и
проверку. [Отчёт](docs/day06-quality.md) и [результаты](docs/day06-quality.json)
сравнивают 18 конфигураций и пороги 0.10–0.40 с выдачей без фильтра.
Рекомендация для этой маленькой базы: chunk size/overlap `800/160`, semantic,
top-k 5, distance ≤ 0.16. Размер и overlap измеряются в символах.
Рабочий индекс и прежние defaults chunking/top-k автоматически не меняются.
Чтобы применить рекомендацию, явно переиндексируйте базу:

```shell
uv run rag-kb --chunk-size 800 --chunk-overlap 160
uv run rag-answer "Как top-k влияет на найденный контекст?" --top-k 5
```

Воспроизведение в новой временной коллекции, удаляемой после эксперимента:

```shell
uv run rag-evaluate --report /tmp/day06-retrieval.json
uv run rag-evaluate --report /tmp/day06-quality.json --generate
DAY06_REPORT_PATH=/tmp/day06-quality.json \
  uv run pytest -m e2e tests/e2e/test_quality_pipeline.py
```

Первый запуск проверяет только retrieval. `--generate` и E2E дополнительно
вызывают настроенный LLM API на выбранной конфигурации; эти вызовы могут быть
платными. Выбор использует только половину вопросов для подбора. JSON сохраняет
исходные чанки и их метрики, оценки всех порогов, параметры моделей, хеши
документов и, при генерации, ответы и количество вызовов LLM.
Изменение базы или embedding-модели требует повторной калибровки.

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
  generation.py              сервис запуска генерации
  generation_content.py      модели ответа и подготовка prompt
  application.py             сборка зависимостей и закрытие клиентов
  generation_pipeline.py     явные шаги генерации ответа
  answer_cli.py              CLI ответа и сравнения
  repositories/
    interfaces.py            порты ChunkRepository и LLMRepository
    weaviate.py              Weaviate repository и подключение
    llm.py                   OpenAI-совместимый LLM repository
    __init__.py              публичные экспорты
  interfaces.py              интерфейс embedding provider
  loader.py, chunker.py       подготовка документов
tests/unit/                   тесты со стабами
tests/integration/            модель + настоящий Weaviate
tests/e2e/                    полный production pipeline
```
