"""Интерфейсы сменяемых стратегий pipeline."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Стратегия получения совместимых embeddings документов и запросов."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Построить embeddings индексируемых текстов."""

    def embed_query(self, text: str) -> list[float]:
        """Построить embedding поискового запроса."""
