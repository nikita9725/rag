"""Явные шаги генерации ответа по retrieved context."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Protocol

from pydantic import ValidationError

from rag_service.generation_content import (
    INSUFFICIENT_CONTEXT,
    SYSTEM_PROMPT,
    ModelAnswer,
    RAGAnswer,
    build_context_prompt,
)
from rag_service.pipeline import PipelineError
from rag_service.repositories import LLMError, LLMRepository
from rag_service.retrieval import RetrievalService, SearchMode, validate_query
from rag_service.schemas import ChunkSearchResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """Состояние запроса; None означает, что этап ещё не завершён."""

    query: str
    top_k: int = 3
    mode: SearchMode = "semantic"
    alpha: float = 0.5
    validated_query: str | None = None
    chunks: tuple[ChunkSearchResult, ...] | None = None
    prompt: str | None = None
    raw_answer: str | None = None
    result: RAGAnswer | None = None


class GenerationStep(Protocol):
    @property
    def name(self) -> str:
        """Имя наблюдаемого шага."""

    def execute(self, context: GenerationContext) -> GenerationContext:
        """Вернуть новое состояние после выполнения шага."""


class ValidateQueryStep(GenerationStep):
    name = "query"

    def execute(self, context: GenerationContext) -> GenerationContext:
        query = validate_query(context.query, context.top_k, context.mode, context.alpha)
        return replace(context, query=query, validated_query=query)


class RetrieveChunksStep(GenerationStep):
    name = "retrieval"

    def __init__(self, retrieval: RetrievalService) -> None:
        self._retrieval = retrieval

    def execute(self, context: GenerationContext) -> GenerationContext:
        if context.validated_query is None:
            raise PipelineError("Шаг retrieval запущен до проверки вопроса")
        chunks = self._retrieval.retrieve(
            context.validated_query, context.top_k, context.mode, context.alpha
        )
        return replace(context, chunks=tuple(chunks))


class BuildContextStep(GenerationStep):
    name = "context"

    def execute(self, context: GenerationContext) -> GenerationContext:
        if context.chunks is None or context.validated_query is None:
            raise PipelineError("Шаг context запущен до retrieval")
        if not context.chunks:
            return replace(
                context,
                result=RAGAnswer(
                    answer=INSUFFICIENT_CONTEXT,
                    insufficient_context=True,
                ),
            )
        return replace(
            context, prompt=build_context_prompt(context.validated_query, list(context.chunks))
        )


class GenerateAnswerStep(GenerationStep):
    name = "generate"

    def __init__(self, repository: LLMRepository) -> None:
        self._repository = repository

    def execute(self, context: GenerationContext) -> GenerationContext:
        if context.result is not None:
            return context
        if context.prompt is None or context.chunks is None or not context.chunks:
            raise PipelineError("Шаг generate запущен до подготовки контекста")
        return replace(context, raw_answer=self._repository.complete(SYSTEM_PROMPT, context.prompt))


class ValidateAnswerStep(GenerationStep):
    name = "validate"

    def execute(self, context: GenerationContext) -> GenerationContext:
        if context.result is not None:
            return context
        if context.raw_answer is None or context.chunks is None:
            raise PipelineError("Шаг validate запущен до генерации ответа")
        try:
            result = ModelAnswer.model_validate_json(context.raw_answer)
        except ValidationError:
            raise LLMError("LLM вернула некорректный JSON RAG-ответа") from None
        if any(source_id > len(context.chunks) for source_id in result.source_ids):
            raise LLMError("LLM сослалась на неизвестный источник")
        source_ids = tuple(dict.fromkeys(result.source_ids))
        if not result.insufficient_context and not source_ids:
            raise LLMError("Содержательный ответ LLM не содержит источников")
        cited_ids = {int(value) for value in re.findall(r"\[(\d+)\]", result.answer)}
        if any(source_id < 1 or source_id > len(context.chunks) for source_id in cited_ids):
            raise LLMError("Текст ответа LLM содержит неизвестную ссылку")
        # Даже в отказе модель может сослаться на нерелевантный фрагмент.
        # Такие ссылки проверяем и включаем в итоговые источники приложения.
        source_ids = tuple(dict.fromkeys((*source_ids, *sorted(cited_ids))))
        logger.info("Шаг validate: ответ проверен, источников %d", len(source_ids))
        answer = RAGAnswer(
            answer=result.answer,
            insufficient_context=result.insufficient_context,
            source_ids=source_ids,
            sources=tuple(context.chunks[source_id - 1] for source_id in source_ids),
            context=context.chunks,
        )
        return replace(context, result=answer)


class GenerationPipeline:
    """Выполнить шаги по порядку с замером времени."""

    def __init__(self, steps: Sequence[GenerationStep]) -> None:
        if not steps:
            raise ValueError("Pipeline должен содержать хотя бы один шаг")
        self._steps = tuple(steps)

    def run(self, context: GenerationContext) -> GenerationContext:
        for step in self._steps:
            started_at = perf_counter()
            logger.info("Шаг %s: начало", step.name)
            context = step.execute(context)
            logger.info("Шаг %s: завершён за %.3f с", step.name, perf_counter() - started_at)
        return context


def build_generation_pipeline(
    retrieval: RetrievalService,
    repository: LLMRepository,
) -> GenerationPipeline:
    """Собрать production-последовательность генерации ответа."""
    return GenerationPipeline(
        [
            ValidateQueryStep(),
            RetrieveChunksStep(retrieval),
            BuildContextStep(),
            GenerateAnswerStep(repository),
            ValidateAnswerStep(),
        ]
    )
