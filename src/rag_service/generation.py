"""Сервис запуска pipeline генерации."""

from rag_service.generation_content import RAGAnswer as RAGAnswer
from rag_service.generation_content import answer_without_retrieval as answer_without_retrieval
from rag_service.generation_pipeline import (
    GenerationContext,
    GenerationPipeline,
    build_generation_pipeline,
)
from rag_service.pipeline import PipelineError
from rag_service.repositories import LLMRepository
from rag_service.retrieval import RetrievalService, SearchMode


class RAGService:
    def __init__(
        self,
        retrieval: RetrievalService,
        repository: LLMRepository,
        *,
        pipeline: GenerationPipeline | None = None,
    ) -> None:
        self._pipeline = (
            pipeline if pipeline is not None else build_generation_pipeline(retrieval, repository)
        )

    def answer(
        self, query: str, top_k: int = 3, mode: SearchMode = "semantic", alpha: float = 0.5
    ) -> RAGAnswer:
        context = self._pipeline.run(
            GenerationContext(query=query, top_k=top_k, mode=mode, alpha=alpha)
        )
        if context.result is None:
            raise PipelineError("Pipeline генерации не сформировал ответ")
        return context.result
