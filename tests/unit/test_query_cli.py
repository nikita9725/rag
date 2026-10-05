from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from rag_service.query_cli import main
from rag_service.repositories import RepositoryError
from rag_service.retrieval import RetrievalService
from rag_service.schemas import ChunkMetadata, ChunkSearchResult


class RetrievalResources:
    def __init__(self) -> None:
        self.service = Mock(spec=RetrievalService)
        self.service.max_distance = 0.2
        self.service.retrieve.return_value = []
        self.opened = False
        self.closed = False

    @contextmanager
    def open(self) -> Iterator[RetrievalService]:
        self.opened = True
        try:
            yield self.service
        finally:
            self.closed = True


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
    resources = RetrievalResources()
    with pytest.raises(SystemExit) as error:
        main(argv, service_factory=resources.open)
    assert error.value.code == 2
    assert not resources.opened


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_cli_shows_context_and_closes_resources(
    mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    resources = RetrievalResources()
    resources.service.retrieve.return_value = [
        ChunkSearchResult(
            uuid="one",
            content="Проверяемый контекст",
            metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=2),
            distance=0.2,
            score=0.8 if mode == "hybrid" else None,
        )
    ]
    main(["вопрос", "--mode", mode], service_factory=resources.open)
    output = capsys.readouterr().out
    assert "source_name=doc.txt chunk_id=2" in output
    assert "Проверяемый контекст" in output
    assert ("distance=0.200000" if mode == "semantic" else "score=0.800000") in output
    assert "Порог cosine distance: 0.2" in output
    assert "distance=0.200000" in output
    assert resources.closed


def test_cli_closes_resources_on_search_error(capsys: pytest.CaptureFixture[str]) -> None:
    resources = RetrievalResources()
    resources.service.retrieve.side_effect = RepositoryError("Коллекция отсутствует")
    with pytest.raises(SystemExit) as error:
        main(["вопрос"], service_factory=resources.open)
    assert error.value.code == 2
    assert "Коллекция отсутствует" in capsys.readouterr().err
    assert resources.closed


def test_cli_empty_results(capsys: pytest.CaptureFixture[str]) -> None:
    resources = RetrievalResources()
    main(["вопрос"], service_factory=resources.open)
    assert "Контекст не найден" in capsys.readouterr().out
