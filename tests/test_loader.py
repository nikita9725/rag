import logging
from pathlib import Path

import pytest

from rag_service.loader import clean_text, load_documents


def test_clean_text_normalizes_whitespace_newlines_and_unicode() -> None:
    decomposed = "е\u0308"
    raw = f"  Первая\t строка  \r\n\r\n  {decomposed}жик   идёт \rПоследняя  "

    assert clean_text(raw) == "Первая строка\nёжик идёт\nПоследняя"


def test_clean_text_returns_empty_string_for_whitespace() -> None:
    assert clean_text(" \n\t\r\n ") == ""


def test_load_documents_filters_files_and_sorts_names(tmp_path: Path) -> None:
    (tmp_path / "b.TXT").write_text("  Второй   файл ", encoding="utf-8")
    (tmp_path / "A.txt").write_text("Первый\n\nфайл", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("Не загружать", encoding="utf-8")
    (tmp_path / "nested").mkdir()

    documents = load_documents(tmp_path)

    assert [document.source.name for document in documents] == ["A.txt", "b.TXT"]
    assert [document.content for document in documents] == ["Первый\nфайл", "Второй файл"]
    assert [document.char_count for document in documents] == [11, 11]


def test_load_documents_skips_invalid_and_empty_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "valid.txt").write_text("Рабочий документ", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("\n \t", encoding="utf-8")
    (tmp_path / "invalid.txt").write_bytes(b"\xff\xfe")

    with caplog.at_level(logging.WARNING):
        documents = load_documents(tmp_path)

    assert [document.source.name for document in documents] == ["valid.txt"]
    assert "Пропущен пустой документ" in caplog.text
    assert "Не удалось прочитать" in caplog.text


def test_load_documents_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="не найдена"):
        load_documents(tmp_path / "missing")


def test_load_documents_rejects_file_path(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("Текст", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="не является папкой"):
        load_documents(path)


def test_load_documents_rejects_directory_without_valid_documents(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="нет доступных"):
        load_documents(tmp_path)
