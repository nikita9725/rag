"""Явные шаги индексации базы знаний."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Protocol

from rag_service.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_documents
from rag_service.interfaces import ChunkRepository, EmbeddingProvider
from rag_service.loader import load_documents
from rag_service.schemas import (
    Chunk,
    ChunkSearchResult,
    Document,
    RepositorySyncResult,
    VectorizedChunk,
)

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Нарушение контракта или результата одного из шагов pipeline."""


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Состояние, явно передаваемое между шагами индексации."""

    directory: Path
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    documents: tuple[Document, ...] = field(default_factory=tuple)
    chunks: tuple[Chunk, ...] = field(default_factory=tuple)
    vectorized_chunks: tuple[VectorizedChunk, ...] = field(default_factory=tuple)
    sync_result: RepositorySyncResult | None = None
    verified_count: int | None = None
    verification_results: tuple[ChunkSearchResult, ...] = field(default_factory=tuple)


class PipelineStep(Protocol):
    """Один наблюдаемый шаг pipeline."""

    @property
    def name(self) -> str:
        """Человекочитаемое имя шага."""

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Выполнить шаг и вернуть новое состояние."""


class LoadDocumentsStep(PipelineStep):
    name = "load"

    def __init__(self, loader: Callable[[Path], list[Document]] = load_documents) -> None:
        self._loader = loader

    def execute(self, context: PipelineContext) -> PipelineContext:
        return replace(context, documents=tuple(self._loader(context.directory)))


class ChunkDocumentsStep(PipelineStep):
    name = "chunk"

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.documents:
            raise PipelineError("Шаг chunk не получил документы")
        chunks = chunk_documents(context.documents, context.chunk_size, context.chunk_overlap)
        if not chunks:
            raise PipelineError("После разбиения не создано ни одного чанка")
        return replace(context, chunks=tuple(chunks))


class EmbedChunksStep(PipelineStep):
    name = "embed"

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.chunks:
            raise PipelineError("Шаг embed не получил чанки")
        vectors = self._provider.embed_documents([chunk.content for chunk in context.chunks])
        if len(vectors) != len(context.chunks):
            raise PipelineError(
                f"Ожидалось {len(context.chunks)} embeddings, получено {len(vectors)}"
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1 or dimensions == {0}:
            raise PipelineError("Embedding-векторы должны иметь одинаковую ненулевую размерность")
        vectorized = tuple(
            VectorizedChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                vector=tuple(vector),
            )
            for chunk, vector in zip(context.chunks, vectors, strict=True)
        )
        return replace(context, vectorized_chunks=vectorized)


class SyncChunksStep(PipelineStep):
    name = "sync"

    def __init__(self, repository: ChunkRepository) -> None:
        self._repository = repository

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.vectorized_chunks:
            raise PipelineError("Шаг sync не получил векторизованные чанки")
        return replace(context, sync_result=self._repository.sync(context.vectorized_chunks))


class VerifyIndexStep(PipelineStep):
    name = "verify"

    def __init__(self, provider: EmbeddingProvider, repository: ChunkRepository) -> None:
        self._provider = provider
        self._repository = repository

    def execute(self, context: PipelineContext) -> PipelineContext:
        if context.sync_result is None or not context.chunks:
            raise PipelineError("Шаг verify запущен до синхронизации")
        count = self._repository.count()
        if count != len(context.chunks):
            raise PipelineError(
                f"В Weaviate ожидалось {len(context.chunks)} объектов, получено {count}"
            )
        query_vector = self._provider.embed_query(context.chunks[0].content)
        results = self._repository.search(query_vector, limit=1)
        if not results:
            raise PipelineError("Контрольный vector search не вернул результатов")
        return replace(
            context,
            verified_count=count,
            verification_results=tuple(results),
        )


class KnowledgeBasePipeline:
    """Последовательно выполнить настроенные шаги индексации."""

    def __init__(self, steps: Sequence[PipelineStep]) -> None:
        if not steps:
            raise ValueError("Pipeline должен содержать хотя бы один шаг")
        self._steps = tuple(steps)

    def run(self, context: PipelineContext) -> PipelineContext:
        for step in self._steps:
            started_at = perf_counter()
            logger.info("Шаг %s: начало", step.name)
            context = step.execute(context)
            logger.info("Шаг %s: завершён за %.3f с", step.name, perf_counter() - started_at)
        return context
