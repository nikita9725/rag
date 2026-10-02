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


def test_cli_reports_error_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing")])

    assert error.value.code == 2
