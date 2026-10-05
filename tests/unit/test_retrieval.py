from unittest.mock import Mock

import pytest

from rag_service.interfaces import EmbeddingProvider
from rag_service.repositories import ChunkRepository
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
            distance=0.1 if mode == "semantic" else None,
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
