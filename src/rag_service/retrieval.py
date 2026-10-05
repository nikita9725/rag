"""Получение контекста по пользовательскому вопросу."""

import logging
from math import isfinite
from typing import Literal

from rag_service.interfaces import EmbeddingProvider
from rag_service.repositories import ChunkRepository, RepositoryError
from rag_service.retrieval_quality import DEFAULT_MAX_DISTANCE
from rag_service.schemas import ChunkSearchResult

SearchMode = Literal["semantic", "hybrid"]
logger = logging.getLogger(__name__)


def validate_max_distance(value: float | None) -> None:
    if value is not None and (not isfinite(value) or not 0 <= value <= 2):
        raise ValueError("Порог distance должен быть конечным числом в диапазоне [0, 2]")


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

    def __init__(
        self,
        provider: EmbeddingProvider,
        repository: ChunkRepository,
        *,
        max_distance: float | None = DEFAULT_MAX_DISTANCE,
    ) -> None:
        validate_max_distance(max_distance)
        self._provider = provider
        self._repository = repository
        self.max_distance = max_distance

    def retrieve(
        self, query: str, top_k: int = 3, mode: SearchMode = "semantic", alpha: float = 0.5
    ) -> list[ChunkSearchResult]:
        query = validate_query(query, top_k, mode, alpha)
        vector = self._provider.embed_query(query)
        if mode == "hybrid":
            results = self._repository.hybrid_search(query, vector, top_k, alpha)
        else:
            results = self._repository.search(vector, top_k)
        for result in results:
            distance = result.distance
            if distance is None or not isfinite(distance) or not 0 <= distance <= 2:
                raise RepositoryError("Результат retrieval не содержит корректный cosine distance")
        accepted = [
            result
            for result in results
            if self.max_distance is None
            or (result.distance is not None and result.distance <= self.max_distance)
        ]
        logger.info(
            "Retrieval: max_distance=%s, найдено=%d, оставлено=%d, отброшено=%d",
            self.max_distance,
            len(results),
            len(accepted),
            len(results) - len(accepted),
        )
        return accepted
