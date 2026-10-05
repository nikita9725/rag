"""Порты хранения чанков и генерации ответа."""

from collections.abc import Sequence
from typing import Protocol

from rag_service.schemas import ChunkSearchResult, RepositorySyncResult, VectorizedChunk


class ChunkRepository(Protocol):
    """Порт хранения и поиска векторизованных чанков."""

    def sync(self, chunks: Sequence[VectorizedChunk]) -> RepositorySyncResult:
        """Привести коллекцию к точному состоянию переданного набора."""

    def count(self) -> int:
        """Вернуть число объектов в коллекции."""

    def search(self, vector: Sequence[float], limit: int) -> list[ChunkSearchResult]:
        """Найти ближайшие чанки по готовому вектору."""

    def hybrid_search(
        self, query: str, vector: Sequence[float], limit: int, alpha: float
    ) -> list[ChunkSearchResult]:
        """Совместить поиск по тексту и готовому вектору."""


class LLMRepository(Protocol):
    """Порт генерации структурированного ответа."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Вернуть JSON-ответ модели."""
