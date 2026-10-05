from collections.abc import Sequence

import pytest
from weaviate import WeaviateClient

from rag_service.embeddings import LocalEmbeddingProvider
from rag_service.repositories import RepositoryError, WeaviateChunkRepository
from rag_service.repositories.weaviate import cosine_distance
from rag_service.retrieval import RetrievalService
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
    assert results[0].distance is not None
    assert results[0].distance >= 0

    hybrid = integration_repository.hybrid_search("Weaviate embeddings", query, 3, 0.5)
    assert len(hybrid) == 1
    assert hybrid[0].metadata.source_name == "doc-0.txt"
    assert hybrid[0].score is not None
    assert hybrid[0].distance == pytest.approx(cosine_distance(query, changed[0].vector), abs=1e-6)
    for mode in ("semantic", "hybrid"):
        service = RetrievalService(real_embedding_provider, integration_repository, max_distance=0)
        assert service.retrieve("Weaviate embeddings", mode=mode) == []


def test_search_does_not_create_missing_collection(
    integration_repository: WeaviateChunkRepository,
    weaviate_client: WeaviateClient,
) -> None:

    with pytest.raises(RepositoryError, match="rag-kb"):
        integration_repository.search([1.0] * 384, 3)
    with pytest.raises(RepositoryError, match="rag-kb"):
        integration_repository.hybrid_search("Weaviate", [1.0] * 384, 3, 0.5)
    assert not weaviate_client.collections.exists(integration_repository.collection_name)


def test_existing_empty_collection_returns_no_results(
    integration_repository: WeaviateChunkRepository,
) -> None:
    integration_repository.sync([])
    assert integration_repository.search([1.0] * 384, 3) == []
    assert integration_repository.hybrid_search("Weaviate", [1.0] * 384, 3, 0.5) == []
