"""Консольный интерфейс загрузки и разбиения базы знаний."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from rag_service.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"максимальный размер чанка в символах (по умолчанию: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=(f"перекрытие соседних чанков в символах (по умолчанию: {DEFAULT_CHUNK_OVERLAP})"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Загрузить документы, разбить их и вывести примеры чанков."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        documents = load_documents(args.directory)
        chunks = chunk_documents(documents, args.chunk_size, args.chunk_overlap)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        parser.error(str(error))

    print("Загруженные документы:")
    for document in documents:
        print(f"- {document.source.name}: {document.char_count} символов")
    total_characters = sum(document.char_count for document in documents)
    print(f"Итого: {len(documents)} документов, {total_characters} символов")

    print(
        f"Создано чанков: {len(chunks)} "
        f"(chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap})"
    )
    print("\nПримеры чанков:")
    for chunk in chunks[:3]:
        metadata = chunk.metadata
        print(
            f"\n[document_id={metadata.document_id}, source_name={metadata.source_name}, "
            f"chunk_id={metadata.chunk_id}, chars={chunk.char_count}]"
        )
        print(chunk.content)
