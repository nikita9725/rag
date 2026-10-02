# RAG Service

Учебный проект, который шаг за шагом строит RAG-пайплайн: от локальных документов до
ответов LLM, основанных на найденном контексте.

## День 1

На первом этапе проект:

- читает UTF-8 `.txt`-файлы из локальной базы знаний;
- нормализует Unicode и переводы строк;
- удаляет пустые строки и лишние пробелы;
- выводит имена загруженных документов и длину очищенного текста.

## Установка и запуск

Требуются `uv` и Python 3.14. Допустимый диапазон версий указан в `pyproject.toml`,
а точные версии зависимостей зафиксированы в `uv.lock`.

```shell
uv sync
uv run rag-kb
```

Другую папку можно передать позиционным аргументом:

```shell
uv run rag-kb path/to/documents
```

Также доступен модульный запуск:

```shell
uv run python -m rag_service
```

## Проверки

```shell
uv run ruff check .
uv run mypy
uv run pytest
```

## Pre-commit

Установить Git hook один раз после `uv sync`:

```shell
uv run pre-commit install
```

Запустить все проверки вручную:

```shell
uv run pre-commit run --all-files
```

Перед коммитом выполняются Ruff, проверка форматирования и mypy. Тесты запускаются
отдельно командой `uv run pytest`.

## Структура

```text
knowledge_base/        учебные документы
src/rag_service/
  cli.py               консольная команда
  loader.py            чтение и очистка документов
  schemas.py           модель документа
tests/                  автоматические тесты
```

Следующие этапы добавят chunking, embeddings, Weaviate, semantic retrieval, grounded
generation и оценку качества. OpenAI API и Weaviate намеренно ещё не подключены.
