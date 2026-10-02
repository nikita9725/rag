from unittest.mock import Mock, patch

import pytest

from rag_service.query_cli import main
from rag_service.repository import RepositoryError
from rag_service.schemas import ChunkMetadata, ChunkSearchResult


@pytest.mark.parametrize(
    "argv",
    [
        [" "],
        ["q", "--top-k", "0"],
        ["q", "--alpha", "0.5"],
        ["q", "--mode", "hybrid", "--alpha", "2"],
    ],
)
def test_invalid_arguments_do_not_load_model(argv: list[str]) -> None:
    with patch("rag_service.query_cli.LocalModelManager") as manager:
        with pytest.raises(SystemExit) as error:
            main(argv)
        assert error.value.code == 2
        manager.assert_not_called()


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_cli_shows_context_and_closes_client(mode: str, capsys: pytest.CaptureFixture[str]) -> None:
    result = ChunkSearchResult(
        uuid="one",
        content="Проверяемый контекст",
        metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=2),
        distance=0.2 if mode == "semantic" else None,
        score=0.8 if mode == "hybrid" else None,
    )
    client = Mock()
    with (
        patch("rag_service.query_cli.LocalModelManager"),
        patch("rag_service.query_cli.LocalEmbeddingProvider"),
        patch("rag_service.query_cli.connect_to_weaviate", return_value=client),
        patch("rag_service.query_cli.RetrievalService") as service,
    ):
        service.return_value.retrieve.return_value = [result]
        main(["вопрос", "--mode", mode])
    output = capsys.readouterr().out
    assert "source_name=doc.txt chunk_id=2" in output
    assert "Проверяемый контекст" in output
    assert ("distance=0.200000" if mode == "semantic" else "score=0.800000") in output
    client.close.assert_called_once()


def test_cli_closes_client_on_search_error(capsys: pytest.CaptureFixture[str]) -> None:
    client = Mock()
    with (
        patch("rag_service.query_cli.LocalModelManager"),
        patch("rag_service.query_cli.LocalEmbeddingProvider"),
        patch("rag_service.query_cli.connect_to_weaviate", return_value=client),
        patch("rag_service.query_cli.RetrievalService") as service,
    ):
        service.return_value.retrieve.side_effect = RepositoryError("Коллекция отсутствует")
        with pytest.raises(SystemExit) as error:
            main(["вопрос"])
        assert error.value.code == 2
    assert "Коллекция отсутствует" in capsys.readouterr().err
    client.close.assert_called_once()


def test_cli_empty_results(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("rag_service.query_cli.LocalModelManager"),
        patch("rag_service.query_cli.LocalEmbeddingProvider"),
        patch("rag_service.query_cli.connect_to_weaviate"),
        patch("rag_service.query_cli.RetrievalService") as service,
    ):
        service.return_value.retrieve.return_value = []
        main(["вопрос"])
    assert "Контекст не найден" in capsys.readouterr().out
