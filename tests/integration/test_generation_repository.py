"""Проверка границ Weaviate и LLM на факте, отсутствующем в памяти модели."""

import pytest

from rag_service.embeddings import LocalEmbeddingProvider
from rag_service.generation import RAGService
from rag_service.repositories import OpenAILLMRepository, WeaviateChunkRepository
from rag_service.retrieval import RetrievalService
from rag_service.schemas import ChunkMetadata, VectorizedChunk
from rag_service.settings import LLMSettings

pytestmark = pytest.mark.integration


def test_real_database_context_reaches_real_llm(
    real_embedding_provider: LocalEmbeddingProvider,
    integration_repository: WeaviateChunkRepository,
) -> None:
    text = "В учебном проекте Кедр проверочный код доступа — CEDAR-4827."
    vector = real_embedding_provider.embed_documents([text])[0]
    integration_repository.sync(
        [
            VectorizedChunk(
                content=text,
                vector=tuple(vector),
                metadata=ChunkMetadata(
                    document_id="cedar",
                    source_name="cedar.txt",
                    chunk_id=0,
                ),
            )
        ]
    )
    llm = OpenAILLMRepository(LLMSettings())
    try:
        result = RAGService(
            RetrievalService(real_embedding_provider, integration_repository), llm
        ).answer("Какой проверочный код у учебного проекта Кедр?", top_k=1)
    finally:
        llm.close()
    assert not result.insufficient_context
    assert "CEDAR-4827" in result.answer
    assert result.sources[0].metadata.source_name == "cedar.txt"
    assert result.sources[0].content == text
