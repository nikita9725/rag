"""Weaviate-реализация репозитория векторизованных чанков."""

from collections.abc import Sequence
from typing import TypedDict
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, UUID, uuid5

import weaviate
from weaviate import WeaviateClient
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import HybridFusion, MetadataQuery
from weaviate.collections import Collection

from rag_service.interfaces import ChunkRepository
from rag_service.schemas import (
    ChunkMetadata,
    ChunkSearchResult,
    RepositorySyncResult,
    VectorizedChunk,
)


class ChunkProperties(TypedDict):
    """Свойства объекта чанка в Weaviate."""

    document_id: str
    source_name: str
    chunk_id: int
    text: str


ChunkCollection = Collection[ChunkProperties, None]


class RepositoryError(RuntimeError):
    """Ошибка схемы, записи или проверки векторного хранилища."""


def chunk_uuid(collection_name: str, chunk: VectorizedChunk) -> UUID:
    """Получить стабильный UUID позиции чанка внутри выделенной коллекции."""

    metadata = chunk.metadata
    key = f"{collection_name}:{metadata.document_id}:{metadata.chunk_id}"
    return uuid5(NAMESPACE_URL, key)


def connect_to_weaviate(url: str, grpc_port: int) -> WeaviateClient:
    """Создать v4 client для HTTP URL и соответствующего gRPC endpoint."""

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RepositoryError(f"Некорректный WEAVIATE_URL: {url}")
    secure = parsed.scheme == "https"
    return weaviate.connect_to_custom(
        http_host=parsed.hostname,
        http_port=parsed.port or (443 if secure else 80),
        http_secure=secure,
        grpc_host=parsed.hostname,
        grpc_port=grpc_port,
        grpc_secure=secure,
    )


class WeaviateChunkRepository(ChunkRepository):
    """Хранилище с полной синхронизацией выделенной collection."""

    def __init__(self, client: WeaviateClient, collection_name: str = "KnowledgeChunk") -> None:
        self._client = client
        self.collection_name = collection_name

    def sync(self, chunks: Sequence[VectorizedChunk]) -> RepositorySyncResult:
        collection = self._ensure_collection()
        desired = {chunk_uuid(self.collection_name, chunk): chunk for chunk in chunks}
        if len(desired) != len(chunks):
            raise RepositoryError("Набор чанков содержит повторяющиеся document_id/chunk_id")

        existing = {item.uuid for item in collection.iterator(include_vector=False)}
        desired_ids = set(desired)

        # Удаление выполняется только после того, как все актуальные объекты записаны.
        for object_id, chunk in desired.items():
            properties = self._properties(chunk)
            vector = list(chunk.vector)
            if not vector:
                raise RepositoryError(f"Чанк {object_id} не содержит embedding")
            if object_id in existing:
                collection.data.replace(
                    uuid=object_id,
                    properties=properties,
                    vector=vector,
                )
            else:
                collection.data.insert(
                    uuid=object_id,
                    properties=properties,
                    vector=vector,
                )

        stale_ids = existing - desired_ids
        for object_id in stale_ids:
            collection.data.delete_by_id(object_id)

        total = self.count()
        if total != len(desired):
            raise RepositoryError(
                f"После синхронизации ожидалось {len(desired)} объектов, получено {total}"
            )
        return RepositorySyncResult(
            inserted=len(desired_ids - existing),
            updated=len(desired_ids & existing),
            deleted=len(stale_ids),
            total=total,
        )

    def count(self) -> int:
        collection = self._ensure_collection()
        result = collection.aggregate.over_all(total_count=True)
        if result.total_count is None:
            raise RepositoryError("Weaviate не вернул total_count")
        return int(result.total_count)

    def search(self, vector: Sequence[float], limit: int) -> list[ChunkSearchResult]:
        if not vector:
            raise ValueError("Поисковый вектор не может быть пустым")
        if limit <= 0:
            raise ValueError("Лимит поиска должен быть больше нуля")

        collection = self._get_collection()
        response = collection.query.near_vector(
            near_vector=list(vector),
            limit=limit,
            return_metadata=MetadataQuery(distance=True),
        )
        results: list[ChunkSearchResult] = []
        for item in response.objects:
            distance = item.metadata.distance
            if distance is None:
                raise RepositoryError("Weaviate не вернул distance для результата поиска")
            properties = item.properties
            results.append(
                ChunkSearchResult(
                    uuid=str(item.uuid),
                    content=str(properties["text"]),
                    metadata=ChunkMetadata(
                        document_id=str(properties["document_id"]),
                        source_name=str(properties["source_name"]),
                        chunk_id=int(properties["chunk_id"]),
                    ),
                    distance=float(distance),
                )
            )
        return results

    def hybrid_search(
        self, query: str, vector: Sequence[float], limit: int, alpha: float
    ) -> list[ChunkSearchResult]:
        if not query.strip() or not vector:
            raise ValueError("Вопрос и поисковый вектор не могут быть пустыми")
        if limit <= 0:
            raise ValueError("Лимит поиска должен быть больше нуля")
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha должен находиться в диапазоне [0, 1]")
        collection = self._get_collection()
        response = collection.query.hybrid(
            query=query,
            vector=list(vector),
            limit=limit,
            alpha=alpha,
            query_properties=["text"],
            fusion_type=HybridFusion.RELATIVE_SCORE,
            return_metadata=MetadataQuery(score=True),
        )
        results: list[ChunkSearchResult] = []
        for item in response.objects:
            if item.metadata.score is None:
                raise RepositoryError("Weaviate не вернул score для hybrid результата")
            properties = item.properties
            results.append(
                ChunkSearchResult(
                    uuid=str(item.uuid),
                    content=str(properties["text"]),
                    metadata=ChunkMetadata(
                        document_id=str(properties["document_id"]),
                        source_name=str(properties["source_name"]),
                        chunk_id=int(properties["chunk_id"]),
                    ),
                    score=float(item.metadata.score),
                )
            )
        return results

    def _get_collection(self) -> ChunkCollection:
        """Получить существующий индекс без создания коллекции при поиске."""
        if not self._client.is_ready():
            raise RepositoryError("Weaviate недоступен или ещё не готов")
        if not self._client.collections.exists(self.collection_name):
            raise RepositoryError("Коллекция отсутствует; выполните rag-kb для индексации")
        collection = self._client.collections.use(
            self.collection_name, data_model_properties=ChunkProperties
        )
        self._validate_schema(collection)
        return collection

    def _ensure_collection(self) -> ChunkCollection:
        if not self._client.is_ready():
            raise RepositoryError("Weaviate недоступен или ещё не готов")
        if not self._client.collections.exists(self.collection_name):
            self._client.collections.create(
                name=self.collection_name,
                properties=[
                    Property(name="document_id", data_type=DataType.TEXT),
                    Property(name="source_name", data_type=DataType.TEXT),
                    Property(name="chunk_id", data_type=DataType.INT),
                    Property(name="text", data_type=DataType.TEXT),
                ],
                vector_config=Configure.Vectors.self_provided(
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE
                    )
                ),
            )

        collection = self._client.collections.use(
            self.collection_name,
            data_model_properties=ChunkProperties,
        )
        self._validate_schema(collection)
        return collection

    @staticmethod
    def _properties(chunk: VectorizedChunk) -> ChunkProperties:
        metadata = chunk.metadata
        return {
            "document_id": metadata.document_id,
            "source_name": metadata.source_name,
            "chunk_id": metadata.chunk_id,
            "text": chunk.content,
        }

    @staticmethod
    def _validate_schema(collection: ChunkCollection) -> None:
        config = collection.config.get()
        actual = {property_.name: property_.data_type for property_ in config.properties}
        expected = {
            "document_id": DataType.TEXT,
            "source_name": DataType.TEXT,
            "chunk_id": DataType.INT,
            "text": DataType.TEXT,
        }
        if actual != expected:
            raise RepositoryError(
                "Существующая collection имеет несовместимую схему; "
                "удалите или переименуйте её вручную"
            )
        if not config.vector_config:
            raise RepositoryError("Collection не настроена для хранения supplied vectors")
