"""Сборка зависимостей приложения и управление жизненным циклом клиентов."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from rag_service.embeddings import LocalEmbeddingProvider, LocalModelManager
from rag_service.generation import RAGService
from rag_service.repositories import (
    LLMRepository,
    OpenAILLMRepository,
    WeaviateChunkRepository,
    connect_to_weaviate,
)
from rag_service.retrieval import RetrievalService
from rag_service.settings import LLMSettings, Settings

RetrievalFactory = Callable[[], AbstractContextManager[RetrievalService]]
GenerationFactory = Callable[[], AbstractContextManager[tuple[RAGService, LLMRepository]]]


@contextmanager
def open_retrieval() -> Iterator[RetrievalService]:
    settings = Settings()
    manager = LocalModelManager(
        settings.embedding_model_id,
        settings.embedding_model_revision,
        settings.embedding_model_path,
    )
    provider = LocalEmbeddingProvider(
        manager.ensure_downloaded(),
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    client = connect_to_weaviate(settings.weaviate_url, settings.weaviate_grpc_port)
    try:
        yield RetrievalService(
            provider,
            WeaviateChunkRepository(client, settings.weaviate_collection),
            max_distance=settings.retrieval_max_distance,
        )
    finally:
        client.close()


@contextmanager
def open_generation() -> Iterator[tuple[RAGService, LLMRepository]]:
    settings = LLMSettings()
    with open_retrieval() as retrieval:
        llm = OpenAILLMRepository(settings)
        try:
            yield RAGService(retrieval, llm), llm
        finally:
            llm.close()
