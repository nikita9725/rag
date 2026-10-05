from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from rag_service.answer_cli import main
from rag_service.generation import RAGAnswer, RAGService
from rag_service.repositories import LLMError, LLMRepository
from rag_service.schemas import ChunkMetadata, ChunkSearchResult


class GenerationResources:
    def __init__(self) -> None:
        self.service = Mock(spec=RAGService)
        self.service.answer.return_value = RAGAnswer(
            answer="Нет сведений", insufficient_context=True
        )
        self.llm = Mock(spec=LLMRepository)
        self.llm.complete.return_value = '{"answer":"Baseline"}'
        self.opened = False
        self.closed = False

    @contextmanager
    def open(self) -> Iterator[tuple[RAGService, LLMRepository]]:
        self.opened = True
        try:
            yield self.service, self.llm
        finally:
            self.closed = True


@pytest.mark.parametrize("argv", [[" "], ["q", "--top-k", "0"], ["q", "--alpha", "0.5"]])
def test_invalid_input_skips_dependencies(argv: list[str]) -> None:
    resources = GenerationResources()
    with pytest.raises(SystemExit):
        main(argv, service_factory=resources.open)
    assert not resources.opened


@pytest.mark.parametrize("fails", [False, True])
def test_resources_closed_and_comparison(fails: bool, capsys: pytest.CaptureFixture[str]) -> None:
    resources = GenerationResources()
    if fails:
        resources.service.answer.side_effect = LLMError("Ошибка модели")
        with pytest.raises(SystemExit):
            main(["вопрос", "--compare"], service_factory=resources.open)
    else:
        main(["вопрос", "--compare"], service_factory=resources.open)
        assert "Baseline" in capsys.readouterr().out
    assert resources.closed


def test_without_comparison_skips_baseline() -> None:
    resources = GenerationResources()
    main(["вопрос"], service_factory=resources.open)
    resources.llm.complete.assert_not_called()


@pytest.mark.parametrize("show_context", [False, True])
def test_context_keeps_source_numbers_and_includes_unused_chunks(
    show_context: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    resources = GenerationResources()
    chunks = tuple(
        ChunkSearchResult(
            uuid=f"uuid-{i}",
            content=f"Полный текст {i}",
            distance=0.1,
            score=0.75,
            metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=i),
        )
        for i in range(2)
    )
    resources.service.answer.return_value = RAGAnswer(
        answer="Факт [2]",
        insufficient_context=False,
        source_ids=(2,),
        sources=(chunks[1],),
        context=chunks,
    )
    main(["вопрос", *(["--show-context"] if show_context else [])], service_factory=resources.open)
    output = capsys.readouterr().out
    assert "[2] source_name=doc.txt chunk_id=1 uuid=uuid-1" in output
    assert ("Полный текст 0" in output) == show_context
    assert ("Полный текст 1" in output) == show_context
    if show_context:
        assert (
            "[1] source_name=doc.txt document_id=doc chunk_id=0 uuid=uuid-0 distance=0.100000"
            in output
        )
        assert "[2] source_name=doc.txt document_id=doc chunk_id=1 uuid=uuid-1" in output
        assert "score=0.750000" in output


def test_empty_context_is_visible(capsys: pytest.CaptureFixture[str]) -> None:
    resources = GenerationResources()
    main(["вопрос", "--show-context"], service_factory=resources.open)
    assert "Контекст пуст; LLM не вызывалась для RAG-ответа." in capsys.readouterr().out
