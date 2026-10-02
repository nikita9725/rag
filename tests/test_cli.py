from pathlib import Path

import pytest

from rag_service.cli import main


def test_cli_prints_document_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "one.txt").write_text("Один документ", encoding="utf-8")
    (tmp_path / "two.txt").write_text("Второй документ", encoding="utf-8")

    main([str(tmp_path)])
    captured = capsys.readouterr()

    assert "one.txt: 13 символов" in captured.out
    assert "two.txt: 15 символов" in captured.out
    assert "Итого: 2 документов, 28 символов" in captured.out
    assert "Создано чанков: 2 (chunk_size=500, chunk_overlap=100)" in captured.out
    assert "document_id=one, source_name=one.txt, chunk_id=0, chars=13" in captured.out
    assert "document_id=two, source_name=two.txt, chunk_id=0, chars=15" in captured.out


def test_cli_reports_error_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing")])

    assert error.value.code == 2


def test_cli_accepts_custom_chunk_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "long.txt").write_text("слово " * 30, encoding="utf-8")

    main([str(tmp_path), "--chunk-size", "50", "--chunk-overlap", "10"])
    captured = capsys.readouterr()

    assert "chunk_size=50, chunk_overlap=10" in captured.out
    assert "Создано чанков:" in captured.out


def test_cli_reports_error_for_invalid_chunk_settings(tmp_path: Path) -> None:
    (tmp_path / "document.txt").write_text("Текст", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), "--chunk-size", "10", "--chunk-overlap", "10"])

    assert error.value.code == 2
