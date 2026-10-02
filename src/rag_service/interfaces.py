"""Интерфейсы сменяемых стратегий pipeline."""

from collections.abc import Sequence
from typing import Protocol

from rag_service.schemas import ChunkSearchResult, RepositorySyncResult, VectorizedChunk


class EmbeddingProvider(Protocol):
    """Стратегия получения совместимых embeddings документов и запросов."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Построить embeddings индексируемых текстов."""

    def embed_query(self, text: str) -> list[float]:
        """Построить embedding поискового запроса."""


class ChunkRepository(Protocol):
    """Порт хранения и поиска векторизованных чанков."""

    def sync(self, chunks: Sequence[VectorizedChunk]) -> RepositorySyncResult:
        """Привести коллекцию к точному состоянию переданного набора."""

    def count(self) -> int:
        """Вернуть число объектов в коллекции."""

    def search(self, vector: Sequence[float], limit: int) -> list[ChunkSearchResult]:
        """Найти ближайшие чанки по готовому вектору."""
