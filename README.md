# RAG mini-product на Weaviate

Учебный Python-сервис отвечает по локальной базе знаний и показывает источники.
Чанки и их векторы действительно хранятся в Weaviate; LLM получает только найденный
контекст. Если сведений недостаточно, сервис возвращает отказ или частичный ответ.

```text
UTF-8 TXT → очистка → чанки → локальные embeddings → Weaviate
вопрос → embedding → поиск в Weaviate → фильтр distance → LLM → ответ + sources
```

## Быстрый запуск

Требуются `uv`, Python 3.14, Docker с Compose и доступ к OpenAI-совместимому LLM API.
Все команды выполняются из корня проекта.

```shell
uv sync --locked
test -f .env || cp .env.example .env
```

Заполните настройки генерации в `.env`, не публикуя ключ:

```dotenv
LLM_API_KEY=your-real-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-flash
```

Используйте URL и имя модели своего провайдера. Индексация и поиск работают без
LLM; генерация и соответствующие тесты обращаются к API и могут быть платными.

```shell
docker compose up -d --wait
docker compose ps
uv run rag-kb
uv run rag-query "Как top-k влияет на найденный контекст?"
uv run rag-answer "Как top-k влияет на найденный контекст?" --show-context
```

Compose поднимает Weaviate REST API на `http://localhost:8080`, gRPC на
`localhost:50051` и UI на `http://localhost:7777`. Порты доступны только локально.
Данные сохраняются в volume `weaviate_data`; `docker compose stop` сохраняет индекс.
UI вспомогательный: поиск и ответы используют Python-клиент Weaviate напрямую.

Первый запуск скачает snapshot `intfloat/multilingual-e5-small` в
`models/multilingual-e5-small`; для этого нужен интернет. Дальше embeddings
строятся локально на CPU без повторной загрузки. Префиксы: `passage:` для документов,
`query:` для вопросов; нормализованные векторы имеют размерность 384.

Ожидаемый результат на поставляемой базе: **7 документов, 7 чанков**. Повторите
`uv run rag-kb`: число объектов не должно вырасти, новых вставок должно быть ноль.

## Подготовка базы знаний и индексация

В `knowledge_base/` лежат семь коротких документов о RAG. Для своей базы добавьте
непустые `.txt` в UTF-8 непосредственно в выбранную папку: вложенные каталоги и
другие форматы не читаются. Loader нормализует Unicode, пробелы и переносы строк;
пустые и нечитаемые файлы пропускает с предупреждением. Если доступных документов
нет, команда завершится с ошибкой.

```shell
uv run rag-kb path/to/documents
uv run rag-kb path/to/documents --chunk-size 300 --chunk-overlap 60
```

По умолчанию размер чанка **800**, overlap **160** символов. Это параметры,
выбранные [экспериментом дня 6](docs/day06-quality.md) для маленькой учебной базы.
Все текущие документы короче 800 символов: каждый становится одним чанком,
перекрытие фактически не используется. Для длинных документов chunker разбивает
текст с overlap; эти значения не являются универсальным оптимумом.

Индексация выполняет `load → chunk → embed → sync → verify`.
Коллекция `KnowledgeChunk` хранит `document_id`, `source_name`, `chunk_id`, `text`
и supplied vector. Стабильный UUID определяется документом и позицией чанка.
Повторная индексация обновляет записи без дублей и удаляет устаревшие чанки после
успешной записи актуального набора. **Команда синхронизирует всю коллекцию с
выбранной папкой**, а не добавляет независимую базу к существующей.

При изменении chunking выполните индексацию заново. Для отдельной базы задайте
другое `WEAVIATE_COLLECTION` в окружении или `.env`.

## Вопросы и источники

```shell
uv run rag-query "Зачем соседние чанки частично перекрываются?"
uv run rag-answer "Зачем соседние чанки частично перекрываются?" --show-context
uv run rag-answer "Как top-k влияет на найденный контекст?" --top-k 3
uv run rag-answer "Как top-k влияет на найденный контекст?" --mode hybrid --alpha 0.5
uv run rag-answer "Как top-k влияет на найденный контекст?" --compare
```

Defaults для CLI и Python API: **semantic**, **top-k 5**, cosine distance **≤ 0.16**.
Фильтр может оставить меньше пяти чанков. Hybrid объединяет vector search и BM25
по тексту; `--alpha` разрешён только для hybrid: 0 — BM25, 1 — vector search.
`--compare` дополнительно вызывает ту же LLM без retrieval.

`rag-query` показывает найденные тексты и метрики. `rag-answer` показывает ответ,
использованные источники и признак недостаточного контекста. `--show-context`
добавляет полный набор фрагментов, переданных LLM, включая неиспользованные.

Как читать результат:

- `[1]`, `[2]` — позиции в **отфильтрованном контексте конкретного вопроса**;
  номера совпадают в ответе, разделе источников и `--show-context`.
- `source_name` — имя исходного файла; его можно открыть в папке базы знаний.
- `chunk_id` — позиция чанка в документе, начиная с **нуля**; это не номер `[N]`.
- `document_id` — идентификатор документа; `uuid` — идентификатор объекта Weaviate.
- `distance` — cosine distance: меньше означает ближе. Hybrid также показывает
  `score`: больше означает выше в выдаче. Эти метрики не являются вероятностью
  правильного ответа и не сравниваются напрямую.
- `sources` — фрагменты, на которые сослался ответ; `context` — все переданные LLM
  фрагменты. Поэтому набор источников может быть меньше контекста.
- `insufficient_context=true` — отказ или частичный ответ с указанием недостающих
  сведений. В CLI этому соответствует сообщение «Контекст недостаточен…».

При пустом контексте RAG возвращает отказ **без вызова LLM**. При непустом контексте
модель проверяет, есть ли нужные факты, и тоже может отказаться. Например, вопрос
о версии Weaviate близок к учебному документу, но версия в нём не указана:
сервис не должен брать её из Docker Compose или памяти модели.

Prompt запрещает внешние факты и выполнение инструкций из документов. Приложение
проверяет JSON и номера источников, восстанавливает реальные метаданные. Это
проверка структуры и ссылок, а не доказательство истинности каждого утверждения.

## Восемь demo-вопросов

Первые пять проверяют полный ответ, шестой — частичный, последние два — отказ:

1. Зачем соседние чанки частично перекрываются?
2. Что обычно нужно сделать с векторами индекса при замене embedding-модели?
3. Какие данные сохраняются вместе с чанком в Weaviate?
4. Как top-k влияет на найденный контекст?
5. Как оценить качество retrieval отдельно от ответа?
6. Зачем нужно перекрытие чанков и какой точный процент перекрытия оптимален для этой базы?
7. Какая точная версия Weaviate установлена в этом проекте?
8. Как приготовить борщ и сколько минут варить свёклу?

Передайте любой вопрос в `uv run rag-answer "вопрос" --show-context`.
[Фактические ответы и полный контекст](docs/day07-demo.md),
[JSON-отчёт](docs/day07-demo.json) и
[машиночитаемый набор вопросов](tests/data/demo_questions.json) позволяют проверить
демонстрацию. Тексты новых ответов LLM могут отличаться.

Воспроизведение демо в отдельной тестовой коллекции, удаляемой после теста:

```shell
DAY07_REPORT_PATH=/tmp/day07-demo.json \
  uv run pytest -m e2e tests/e2e/test_demo_pipeline.py -q
```

Тест использует production CLI, defaults, реальную embedding-модель, Weaviate и LLM.
Он проверяет повторную индексацию, источники, полные/частичные ответы и оба вида
отказа; JSON сохраняет контекст, ответы, stdout CLI и число вызовов LLM.

## Настройки

Полный образец — [`.env.example`](.env.example). Переменные окружения имеют
приоритет над корневым `.env`; относительный путь модели разрешается от корня проекта.

| Настройка | Значение по умолчанию / назначение |
| --- | --- |
| `EMBEDDING_MODEL_ID` | `intfloat/multilingual-e5-small` |
| `EMBEDDING_MODEL_REVISION` | Закреплённый snapshot из `.env.example` |
| `EMBEDDING_MODEL_PATH` | `models/multilingual-e5-small` |
| `EMBEDDING_DEVICE`, `EMBEDDING_BATCH_SIZE` | `cpu`, `32` |
| `WEAVIATE_URL`, `WEAVIATE_GRPC_PORT` | `http://localhost:8080`, `50051` |
| `WEAVIATE_COLLECTION` | `KnowledgeChunk` |
| `WEAVIATE_INTEGRATION_COLLECTION` | `KnowledgeChunkIntegration` |
| `WEAVIATE_E2E_COLLECTION` | `KnowledgeChunkE2E` |
| `RETRIEVAL_MAX_DISTANCE` | `0.16`, допустимы конечные значения 0–2 |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Обязательны для генерации |
| `LLM_RETRY_MAX_ATTEMPTS` | `3`, включая первую попытку |
| `LLM_RETRY_BASE_DELAY_SECONDS` | `0.5`, задержка удваивается |

Timeout запроса LLM — 60 секунд. Сетевые ошибки, 429 и 5xx повторяются;
некорректный JSON, неизвестные ссылки и полный ответ без источников дают ошибку CLI.
Пустой/некорректный вопрос тоже даёт ненулевой exit code; честный отказ является
нормальным результатом, а не ошибкой подключения.

При смене корпуса или embedding-модели переиндексируйте документы и повторно
проверьте порог relevance. Матрица качества:

```shell
uv run rag-evaluate --report /tmp/retrieval-quality.json
uv run rag-evaluate --report /tmp/answer-quality.json --generate
```

## Проверки и диагностика

Unit-тесты не требуют Docker, модели или API:

```shell
uv run pytest tests/unit -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Реальные integration/E2E-тесты требуют запущенный Weaviate и локальную модель;
проверки генерации дополнительно требуют настройки LLM API. Отсутствие подключения
или ключа — ошибка, а не автоматический skip. Тестовые коллекции очищаются;
не задавайте им имя рабочей коллекции.

```shell
docker compose up -d --wait
uv run pytest -m integration tests/integration -q
uv run pytest -m e2e tests/e2e -q
```

Проверка готовности БД и просмотр логов:

```shell
curl --fail http://localhost:8080/v1/.well-known/ready
docker compose logs --tail 100 weaviate
```

Если коллекции нет, выполните `uv run rag-kb`. Если ответ отказной, посмотрите
`rag-query` и `--show-context`: нужного факта может не быть в документах либо он
не прошёл порог. Более высокий порог допускает больше контекста, но может добавить
шум. При ошибке загрузки модели проверьте интернет и `EMBEDDING_MODEL_PATH`.
При ошибке LLM проверьте ключ, endpoint, имя модели и лимиты провайдера.

В UI `http://localhost:7777` выберите `KnowledgeChunk` и проверьте свойства,
UUID и количество объектов. Основные команды работают независимо от UI.
Для остановки: `docker compose stop`. Команда `docker compose down -v` удаляет
контейнеры **и все данные локального Weaviate**.

## Структура и Python API

```text
knowledge_base/                исходные TXT-документы
models/                        локальный snapshot (исключён из Git)
src/rag_service/
  loader.py, chunker.py         чтение и разбиение
  embeddings.py                управление моделью и embeddings
  pipeline.py, cli.py          индексация и rag-kb
  retrieval.py, query_cli.py    retrieval и rag-query
  generation_content.py        prompt и модели ответа
  generation_pipeline.py       шаги генерации и проверки
  generation.py, answer_cli.py  RAGService и rag-answer
  application.py               сборка и закрытие клиентов
  repositories/                порты и адаптеры Weaviate / LLM
  evaluation*.py               оценка качества и rag-evaluate
  settings.py, schemas.py       конфигурация и схемы данных
tests/unit/                    проверки со стабами
tests/integration/             реальные адаптеры
tests/e2e/                     полные pipelines и демо
tests/data/                    размеченные вопросы
docs/                          отчёты дней 4–7
```

```python
from rag_service.application import open_generation

with open_generation() as (service, llm):
    result = service.answer("Как top-k влияет на найденный контекст?")
    print(result.answer, result.insufficient_context)
    for number, source in zip(result.source_ids, result.sources, strict=True):
        print(number, source.metadata.source_name, source.metadata.chunk_id)
```

`RAGAnswer` содержит `answer`, `insufficient_context`, `source_ids`, `sources`,
`context`. Зависимости передаются через интерфейсы; шаги индексации и генерации
логируют прогресс и длительность. Проект намеренно остаётся учебным mini-product
без ingestion UI, re-ranking и распределённой инфраструктуры.

Предыдущие этапы: [retrieval](docs/day04-retrieval.md),
[генерация](docs/day05-generation.md), [качество](docs/day06-quality.md).
Отчёты сохраняют параметры своих прогонов; defaults дня 7 — 800/160, top-k 5.
