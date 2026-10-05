"""Получение контекста по пользовательскому вопросу."""

from typing import Literal

from rag_service.interfaces import EmbeddingProvider
from rag_service.repositories import ChunkRepository
from rag_service.schemas import ChunkSearchResult

SearchMode = Literal["semantic", "hybrid"]


def validate_query(query: str, top_k: int, mode: str, alpha: float) -> str:
    """Проверить параметры до обращения к модели или хранилищу."""
    query = query.strip()
    if not query:
        raise ValueError("Вопрос не может быть пустым")
    if top_k <= 0:
        raise ValueError("Top-k должен быть больше нуля")
    if mode not in {"semantic", "hybrid"}:
        raise ValueError("Режим должен быть semantic или hybrid")
    if not 0 <= alpha <= 1:
        raise ValueError("Alpha должен находиться в диапазоне [0, 1]")
    return query


class RetrievalService:
    """Построить embedding вопроса и вернуть упорядоченные чанки."""

    def __init__(self, provider: EmbeddingProvider, repository: ChunkRepository) -> None:
        self._provider = provider
        self._repository = repository

    def retrieve(
        self, query: str, top_k: int = 3, mode: SearchMode = "semantic", alpha: float = 0.5
    ) -> list[ChunkSearchResult]:
        query = validate_query(query, top_k, mode, alpha)
        vector = self._provider.embed_query(query)
        if mode == "hybrid":
            return self._repository.hybrid_search(query, vector, top_k, alpha)
        return self._repository.search(vector, top_k)
