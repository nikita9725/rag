"""Консольный интерфейс первого этапа RAG-проекта."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from rag_service.loader import load_documents

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Загрузить и проверить локальную базу знаний")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("knowledge_base"),
        help="папка с UTF-8 TXT-документами (по умолчанию: knowledge_base)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Загрузить документы и вывести краткий отчёт."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        documents = load_documents(args.directory)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        parser.error(str(error))

    print("Загруженные документы:")
    for document in documents:
        print(f"- {document.source.name}: {document.char_count} символов")
    total_characters = sum(document.char_count for document in documents)
    print(f"Итого: {len(documents)} документов, {total_characters} символов")
