from pathlib import Path

import pytest

from rag_service.chunker import chunk_document, chunk_documents
from rag_service.schemas import Document


def test_short_document_becomes_one_chunk_with_metadata() -> None:
    document = Document(source=Path("knowledge_base/02_chunking.txt"), content="Короткий текст")

    chunks = chunk_document(document, chunk_size=100, chunk_overlap=20)

    assert len(chunks) == 1
    assert chunks[0].content == document.content
    assert chunks[0].char_count == len(document.content)
    assert chunks[0].metadata.document_id == "02_chunking"
    assert chunks[0].metadata.source_name == "02_chunking.txt"
    assert chunks[0].metadata.chunk_id == 0


def test_chunker_prefers_newline_to_a_later_sentence_boundary() -> None:
    text = f"{'А' * 45}\nКоротко. {'Б' * 50}"
    document = Document(source=Path("source.txt"), content=text)

    chunks = chunk_document(document, chunk_size=70, chunk_overlap=0)

    assert chunks[0].content == "А" * 45
    assert chunks[1].content.startswith("Коротко")


def test_chunker_uses_sentence_boundary_without_newline() -> None:
    first_sentence = "Это достаточно длинное первое предложение."
    text = f"{first_sentence} Второе предложение тоже содержит много текста для разбиения."
    document = Document(source=Path("source.txt"), content=text)

    chunks = chunk_document(document, chunk_size=60, chunk_overlap=0)

    assert chunks[0].content == first_sentence


def test_chunker_uses_word_boundary_and_respects_size_limit() -> None:
    document = Document(
        source=Path("source.txt"),
        content="альфа бета гамма дельта эпсилон дзета эта тета йота каппа лямбда",
    )

    chunks = chunk_document(document, chunk_size=30, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(chunk.char_count <= 30 for chunk in chunks)
    assert all(not chunk.content.startswith(" ") for chunk in chunks)
    assert all(not chunk.content.endswith(" ") for chunk in chunks)


def test_chunker_falls_back_to_hard_slices_for_text_without_boundaries() -> None:
    document = Document(source=Path("source.txt"), content="а" * 25)

    chunks = chunk_document(document, chunk_size=10, chunk_overlap=0)

    assert [chunk.char_count for chunk in chunks] == [10, 10, 5]
    assert "".join(chunk.content for chunk in chunks) == document.content


def test_adjacent_chunks_overlap_on_complete_words() -> None:
    text = "один два три четыре пять шесть семь восемь девять десять одиннадцать"
    document = Document(source=Path("source.txt"), content=text)

    chunks = chunk_document(document, chunk_size=35, chunk_overlap=10)

    assert len(chunks) > 1
    first, second = chunks[:2]
    shared_suffixes = [
        size
        for size in range(1, min(first.char_count, second.char_count) + 1)
        if first.content[-size:] == second.content[:size]
    ]
    assert max(shared_suffixes) >= 10
    assert second.content[0] != " "


def test_zero_overlap_does_not_repeat_text() -> None:
    text = "Первое предложение. Второе предложение. Третье предложение."
    document = Document(source=Path("source.txt"), content=text)

    chunks = chunk_document(document, chunk_size=25, chunk_overlap=0)

    assert " ".join(chunk.content for chunk in chunks) == text


def test_multiple_documents_keep_order_and_restart_chunk_ids() -> None:
    documents = [
        Document(source=Path("first.txt"), content="а" * 25),
        Document(source=Path("second.txt"), content="б" * 25),
    ]

    chunks = chunk_documents(documents, chunk_size=10, chunk_overlap=0)

    assert [chunk.metadata.document_id for chunk in chunks] == [
        "first",
        "first",
        "first",
        "second",
        "second",
        "second",
    ]
    assert [chunk.metadata.chunk_id for chunk in chunks] == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "больше нуля"),
        (-1, 0, "больше нуля"),
        (10, -1, "не может быть отрицательным"),
        (10, 10, "должно быть меньше"),
        (10, 11, "должно быть меньше"),
    ],
)
def test_chunker_rejects_invalid_settings(
    chunk_size: int, chunk_overlap: int, message: str
) -> None:
    document = Document(source=Path("source.txt"), content="Текст")

    with pytest.raises(ValueError, match=message):
        chunk_document(document, chunk_size, chunk_overlap)
