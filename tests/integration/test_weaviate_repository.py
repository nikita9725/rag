from collections.abc import Sequence

import pytest

from rag_service.embeddings import LocalEmbeddingProvider
from rag_service.repository import WeaviateChunkRepository
from rag_service.schemas import ChunkMetadata, VectorizedChunk

pytestmark = pytest.mark.integration


def vectorize(provider: LocalEmbeddingProvider, texts: Sequence[str]) -> list[VectorizedChunk]:
    vectors = provider.embed_documents(texts)
    return [
        VectorizedChunk(
            content=text,
            metadata=ChunkMetadata(
                document_id=f"doc-{index}",
                source_name=f"doc-{index}.txt",
                chunk_id=0,
            ),
            vector=tuple(vector),
        )
        for index, (text, vector) in enumerate(zip(texts, vectors, strict=True))
    ]


def test_repository_syncs_updates_deletes_and_searches_real_vectors(
    real_embedding_provider: LocalEmbeddingProvider,
    integration_repository: WeaviateChunkRepository,
) -> None:
    initial = vectorize(
        real_embedding_provider,
        ["Weaviate хранит векторные представления.", "Чанки создаются из документов."],
    )

    first = integration_repository.sync(initial)

    assert first.inserted == 2
    assert first.updated == 0
    assert first.total == 2

    changed = vectorize(
        real_embedding_provider,
        ["Weaviate хранит embeddings и метаданные для поиска."],
    )
    second = integration_repository.sync(changed)

    assert second.inserted == 0
    assert second.updated == 1
    assert second.deleted == 1
    assert second.total == 1

    query = real_embedding_provider.embed_query("Где хранятся embeddings?")
    results = integration_repository.search(query, limit=1)

    assert len(results) == 1
    assert results[0].content == changed[0].content
    assert results[0].metadata.source_name == "doc-0.txt"
    assert results[0].distance >= 0
