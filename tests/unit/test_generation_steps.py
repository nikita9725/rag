import json
from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest

from rag_service.generation import RAGService
from rag_service.generation_pipeline import (
    BuildContextStep,
    GenerateAnswerStep,
    GenerationContext,
    GenerationPipeline,
    GenerationStep,
    RetrieveChunksStep,
    ValidateAnswerStep,
    ValidateQueryStep,
    build_generation_pipeline,
)
from rag_service.pipeline import PipelineError
from rag_service.repositories import LLMRepository
from rag_service.retrieval import RetrievalService
from rag_service.schemas import ChunkMetadata, ChunkSearchResult


class RecordingStep:
    def __init__(self, name: str, calls: list[str], *, fail: bool = False) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail

    def execute(self, context: GenerationContext) -> GenerationContext:
        self.calls.append(self.name)
        if self.fail:
            raise PipelineError(self.name)
        return context


@pytest.mark.parametrize("fails", [False, True])
def test_runner_preserves_order_and_stops_on_error(fails: bool) -> None:
    calls: list[str] = []
    pipeline = GenerationPipeline(
        [
            RecordingStep("first", calls),
            RecordingStep("second", calls, fail=fails),
            RecordingStep("third", calls),
        ]
    )
    context = GenerationContext(query="вопрос")
    if fails:
        with pytest.raises(PipelineError, match="second"):
            pipeline.run(context)
        assert calls == ["first", "second"]
    else:
        assert pipeline.run(context) is context
        assert calls == ["first", "second", "third"]


def test_runner_requires_steps() -> None:
    with pytest.raises(ValueError, match="хотя бы один шаг"):
        GenerationPipeline([])


def test_query_step_returns_new_immutable_context() -> None:
    original = GenerationContext(query="  вопрос  ", top_k=2, mode="hybrid", alpha=0.7)
    validated = ValidateQueryStep().execute(original)
    assert original.query == "  вопрос  " and original.validated_query is None
    assert validated.query == validated.validated_query == "вопрос"
    assert (validated.top_k, validated.mode, validated.alpha) == (2, "hybrid", 0.7)
    with pytest.raises(FrozenInstanceError):
        setattr(validated, "query", "другой вопрос")


@pytest.mark.parametrize("name", ["retrieval", "context", "generate", "validate"])
def test_step_requires_preceding_result(name: str) -> None:
    retrieval = Mock(spec=RetrievalService)
    repository = Mock(spec=LLMRepository)
    steps: dict[str, GenerationStep] = {
        "retrieval": RetrieveChunksStep(retrieval),
        "context": BuildContextStep(),
        "generate": GenerateAnswerStep(repository),
        "validate": ValidateAnswerStep(),
    }
    with pytest.raises(PipelineError, match="до"):
        steps[name].execute(GenerationContext(query="вопрос"))
    retrieval.retrieve.assert_not_called()
    repository.complete.assert_not_called()


def test_empty_retrieval_reaches_final_refusal_without_llm() -> None:
    retrieval = Mock(spec=RetrievalService)
    retrieval.retrieve.return_value = []
    repository = Mock(spec=LLMRepository)
    context = build_generation_pipeline(retrieval, repository).run(
        GenerationContext(query="вопрос")
    )
    assert context.chunks == ()
    assert context.prompt is None and context.raw_answer is None
    assert context.result is not None and context.result.insufficient_context
    assert GenerateAnswerStep(repository).execute(context) is context
    assert ValidateAnswerStep().execute(context) is context
    repository.complete.assert_not_called()


def test_production_steps_populate_context_and_log_duration(
    caplog: pytest.LogCaptureFixture,
) -> None:
    retrieval = Mock(spec=RetrievalService)
    chunk = ChunkSearchResult(
        uuid="uuid",
        content="Ответ — 42.",
        metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=0),
    )
    retrieval.retrieve.return_value = [chunk]
    repository = Mock(spec=LLMRepository)
    raw = json.dumps({"answer": "42 [1]", "source_ids": [1], "insufficient_context": False})
    repository.complete.return_value = raw
    original = GenerationContext(query="  вопрос  ")
    with caplog.at_level("INFO", logger="rag_service.generation_pipeline"):
        context = build_generation_pipeline(retrieval, repository).run(original)
    assert original.chunks is None and original.result is None
    assert context.chunks == (chunk,)
    assert (
        context.prompt is not None
        and json.loads(context.prompt)["context"][0]["text"] == chunk.content
    )
    assert context.raw_answer == raw
    assert context.result is not None and context.result.sources == (chunk,)
    started = [record.message for record in caplog.records if record.message.endswith(": начало")]
    assert started == [
        f"Шаг {name}: начало" for name in ("query", "retrieval", "context", "generate", "validate")
    ]
    assert sum("завершён за" in record.message for record in caplog.records) == 5


def test_service_rejects_pipeline_without_final_result() -> None:
    pipeline = Mock(spec=GenerationPipeline)
    pipeline.run.return_value = GenerationContext(query="вопрос")
    with pytest.raises(PipelineError, match="не сформировал ответ"):
        RAGService(Mock(spec=RetrievalService), Mock(spec=LLMRepository), pipeline=pipeline).answer(
            "вопрос"
        )
