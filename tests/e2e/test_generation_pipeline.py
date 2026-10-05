"""Сравнение реальной LLM с retrieval и без него на восьми вопросах."""

import json
import os
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from rag_service.cli import index_knowledge_base
from rag_service.embeddings import LocalEmbeddingProvider
from rag_service.generation import RAGAnswer, RAGService, answer_without_retrieval
from rag_service.repositories import OpenAILLMRepository, WeaviateChunkRepository
from rag_service.retrieval import RetrievalService
from rag_service.retrieval_quality import DEFAULT_MAX_DISTANCE
from rag_service.settings import PROJECT_ROOT, LLMSettings, Settings

pytestmark = pytest.mark.e2e


class GenerationCase(BaseModel):
    query: str
    expected_sources: list[str]
    insufficient_context: bool
    min_source_documents: int = 1


class Comparison(BaseModel):
    model_config = ConfigDict(frozen=True)
    query: str
    baseline: str
    rag: RAGAnswer
    expected_sources: list[str]
    expected_insufficient_context: bool
    min_source_documents: int = 1


def test_eight_questions_with_and_without_retrieval(
    real_embedding_provider: LocalEmbeddingProvider,
    e2e_repository: WeaviateChunkRepository,
) -> None:
    # Проверяем полные ответы с контекстом, выбранным экспериментом Дня 6.
    # 500/100, top-k 3 может не захватить объяснение и дать честный частичный ответ.
    chunk_size, chunk_overlap, top_k = 800, 160, 5
    index_knowledge_base(
        PROJECT_ROOT / "knowledge_base",
        chunk_size,
        chunk_overlap,
        real_embedding_provider,
        e2e_repository,
    )
    cases = [
        GenerationCase.model_validate(case)
        for case in json.loads(
            (PROJECT_ROOT / "tests/data/generation_questions.json").read_text(encoding="utf-8")
        )
    ]
    settings = LLMSettings()
    repository = OpenAILLMRepository(settings)
    comparisons: list[Comparison] = []
    try:
        service = RAGService(RetrievalService(real_embedding_provider, e2e_repository), repository)
        for case in cases:
            rag = service.answer(case.query, top_k=top_k)
            baseline = answer_without_retrieval(case.query, repository)
            comparisons.append(
                Comparison(
                    query=case.query,
                    baseline=baseline,
                    rag=rag,
                    expected_sources=case.expected_sources,
                    expected_insufficient_context=case.insufficient_context,
                    min_source_documents=case.min_source_documents,
                )
            )
    finally:
        repository.close()
    report_path = os.environ.get("GENERATION_REPORT_PATH")
    if report_path:
        payload = {
            "llm_model": settings.llm_model,
            "embedding_model": Settings().embedding_model_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "max_distance": DEFAULT_MAX_DISTANCE,
            "mode": "semantic",
            "comparisons": [item.model_dump(mode="json") for item in comparisons],
        }
        Path(report_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for item in comparisons:
        assert item.rag.insufficient_context == item.expected_insufficient_context, item.query
        if item.expected_sources:
            sources = {source.metadata.source_name for source in item.rag.sources}
            assert sources.intersection(item.expected_sources), item.query
            assert len(sources) >= item.min_source_documents, item.query
        assert all(source in item.rag.context for source in item.rag.sources)
