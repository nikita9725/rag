from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from rag_service.answer_cli import main
from rag_service.generation import RAGAnswer, RAGService
from rag_service.repositories import LLMError, LLMRepository


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
