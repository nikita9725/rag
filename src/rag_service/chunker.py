"""Разбиение документов на перекрывающиеся смысловые фрагменты."""

import re
from collections.abc import Sequence

from rag_service.schemas import Chunk, ChunkMetadata, Document

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 160
_MIN_BOUNDARY_POSITION_RATIO = 0.6
_SENTENCE_END_PATTERN = re.compile(r"[.!?](?=\s)")


def _validate_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("Размер чанка должен быть больше нуля")
    if chunk_overlap < 0:
        raise ValueError("Перекрытие чанков не может быть отрицательным")
    if chunk_overlap >= chunk_size:
        raise ValueError("Перекрытие чанков должно быть меньше размера чанка")


def _choose_chunk_end(text: str, start: int, chunk_size: int) -> int:
    """Выбрать смысловую границу, не превышая лимит размера."""

    maximum_end = min(start + chunk_size, len(text))
    if maximum_end == len(text):
        return maximum_end

    minimum_end = start + max(1, int(chunk_size * _MIN_BOUNDARY_POSITION_RATIO))
    search_area = text[minimum_end:maximum_end]

    newline_position = search_area.rfind("\n")
    if newline_position >= 0:
        return minimum_end + newline_position

    sentence_matches = list(_SENTENCE_END_PATTERN.finditer(search_area))
    if sentence_matches:
        return minimum_end + sentence_matches[-1].end()

    for position in range(maximum_end - 1, minimum_end - 1, -1):
        if text[position].isspace():
            return position

    return maximum_end


def _choose_next_start(text: str, current_start: int, end: int, overlap: int) -> int:
    """Найти начало следующего чанка около заданного overlap и на границе слова."""

    if overlap == 0:
        candidate = end
    else:
        desired_start = max(current_start + 1, end - overlap)
        boundary = desired_start
        while boundary > current_start and not text[boundary - 1].isspace():
            boundary -= 1
        candidate = boundary if boundary > current_start else desired_start

    while candidate < len(text) and text[candidate].isspace():
        candidate += 1
    return max(current_start + 1, candidate)


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Разбить документ на чанки и добавить метаданные источника."""

    _validate_settings(chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    start = 0
    while start < len(document.content):
        end = _choose_chunk_end(document.content, start, chunk_size)
        content = document.content[start:end].strip()
        if content:
            chunks.append(
                Chunk(
                    content=content,
                    metadata=ChunkMetadata(
                        document_id=document.source.stem,
                        source_name=document.source.name,
                        chunk_id=len(chunks),
                    ),
                )
            )

        if end == len(document.content):
            break
        start = _choose_next_start(document.content, start, end, chunk_overlap)

    return chunks


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Разбить документы, сохранив их исходный порядок."""

    _validate_settings(chunk_size, chunk_overlap)
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_size, chunk_overlap)
    ]
