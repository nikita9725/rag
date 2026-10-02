"""Загрузка и базовая очистка локальных текстовых документов."""

import logging
import unicodedata
from pathlib import Path

from rag_service.schemas import Document

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Нормализовать Unicode и пробелы, удалить пустые строки."""

    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = (" ".join(line.split()) for line in normalized.split("\n"))
    return "\n".join(line for line in cleaned_lines if line)


def load_documents(directory: Path) -> list[Document]:
    """Загрузить непустые UTF-8 TXT-файлы в детерминированном порядке."""

    if not directory.exists():
        raise FileNotFoundError(f"Папка базы знаний не найдена: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Путь базы знаний не является папкой: {directory}")

    documents: list[Document] = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.lower() != ".txt":
            continue

        try:
            content = clean_text(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as error:
            logger.warning("Не удалось прочитать %s: %s", path, error)
            continue

        if not content:
            logger.warning("Пропущен пустой документ: %s", path)
            continue

        documents.append(Document(source=path, content=content))

    if not documents:
        raise ValueError(f"В папке {directory} нет доступных непустых UTF-8 TXT-документов")

    return documents
