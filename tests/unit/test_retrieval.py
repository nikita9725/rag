from unittest.mock import Mock

import pytest

from rag_service.interfaces import EmbeddingProvider
from rag_service.repositories import ChunkRepository, RepositoryError
from rag_service.retrieval import RetrievalService, SearchMode
from rag_service.schemas import ChunkMetadata, ChunkSearchResult


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_retrieval_preserves_results_and_passes_query_vector(mode: SearchMode) -> None:
    provider = Mock(spec=EmbeddingProvider)
    repository = Mock(spec=ChunkRepository)
    provider.embed_query.return_value = [0.1, 0.2]
    results = [
        ChunkSearchResult(
            uuid=str(i),
            content=f"text {i}",
            metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=i),
            distance=0.1,
            score=0.9 if mode == "hybrid" else None,
        )
        for i in [2, 1]
    ]
    repository.search.return_value = results
    repository.hybrid_search.return_value = results

    assert RetrievalService(provider, repository).retrieve("  вопрос  ", 2, mode, 0.7) == results
    provider.embed_query.assert_called_once_with("вопрос")
    if mode == "semantic":
        repository.search.assert_called_once_with([0.1, 0.2], 2)
        repository.hybrid_search.assert_not_called()
    else:
        repository.hybrid_search.assert_called_once_with("вопрос", [0.1, 0.2], 2, 0.7)
        repository.search.assert_not_called()


@pytest.mark.parametrize(
    ("query", "top_k", "alpha"),
    [
        (" \n", 3, 0.5),
        ("q", 0, 0.5),
        ("q", -1, 0.5),
        ("q", 3, -0.1),
        ("q", 3, 1.1),
        ("q", 3, float("nan")),
    ],
)
def test_invalid_input_does_not_call_dependencies(query: str, top_k: int, alpha: float) -> None:
    provider = Mock(spec=EmbeddingProvider)
    repository = Mock(spec=ChunkRepository)
    with pytest.raises(ValueError):
        RetrievalService(provider, repository).retrieve(query, top_k, alpha=alpha)
    assert provider.mock_calls == []
    assert repository.mock_calls == []


def test_empty_search_returns_empty_context() -> None:
    provider = Mock(spec=EmbeddingProvider)
    repository = Mock(spec=ChunkRepository)
    repository.search.return_value = []
    assert RetrievalService(provider, repository).retrieve("вопрос") == []


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
@pytest.mark.parametrize("threshold,expected", [(0.2, [0, 2]), (0.0, []), (None, [0, 1, 2])])
def test_threshold_filters_without_reordering(
    mode: SearchMode, threshold: float | None, expected: list[int]
) -> None:
    provider = Mock(spec=EmbeddingProvider)
    repository = Mock(spec=ChunkRepository)
    chunks = [
        ChunkSearchResult(
            uuid=str(i),
            content=f"fact {i}",
            metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=i),
            distance=distance,
            score=1.0 if mode == "hybrid" else None,
        )
        for i, distance in enumerate([0.2, 0.21, 0.1])
    ]
    repository.search.return_value = repository.hybrid_search.return_value = chunks
    results = RetrievalService(provider, repository, max_distance=threshold).retrieve(
        "q", mode=mode
    )
    assert [result.metadata.chunk_id for result in results] == expected


@pytest.mark.parametrize("threshold", [-1, 2.1, float("nan"), float("inf")])
def test_invalid_threshold_rejected(threshold: float) -> None:
    with pytest.raises(ValueError):
        RetrievalService(
            Mock(spec=EmbeddingProvider), Mock(spec=ChunkRepository), max_distance=threshold
        )


@pytest.mark.parametrize("distance", [None, -0.1, 2.1, float("nan"), float("inf")])
def test_invalid_distance_fails_closed(distance: float | None) -> None:
    repository = Mock(spec=ChunkRepository)
    repository.search.return_value = [
        ChunkSearchResult(
            uuid="bad",
            content="bad",
            distance=distance,
            metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=0),
        )
    ]
    with pytest.raises(RepositoryError):
        RetrievalService(Mock(spec=EmbeddingProvider), repository).retrieve("q")
