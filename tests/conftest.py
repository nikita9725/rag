from collections.abc import Generator

import pytest
from weaviate import WeaviateClient

from rag_service.embeddings import LocalEmbeddingProvider, LocalModelManager
from rag_service.repository import WeaviateChunkRepository, connect_to_weaviate
from rag_service.settings import Settings


@pytest.fixture(scope="session")
def real_embedding_provider() -> LocalEmbeddingProvider:
    settings = Settings()
    manager = LocalModelManager(
        settings.embedding_model_id,
        settings.embedding_model_revision,
        settings.embedding_model_path,
    )
    path = manager.ensure_downloaded()
    provider = LocalEmbeddingProvider(
        path,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    # Загружаем модель сразу, чтобы ошибка окружения была причиной падения fixture.
    provider.embed_query("проверка локальной модели")
    return provider


@pytest.fixture(scope="session")
def weaviate_client() -> Generator[WeaviateClient]:
    settings = Settings()
    try:
        client = connect_to_weaviate(settings.weaviate_url, settings.weaviate_grpc_port)
        if not client.is_ready():
            pytest.fail("Weaviate не готов; выполните docker compose up -d")
    except Exception as error:
        pytest.fail(f"Не удалось подключиться к Weaviate: {error}")
    try:
        yield client
    finally:
        client.close()


def _repository_fixture(client: WeaviateClient, name: str) -> Generator[WeaviateChunkRepository]:
    if client.collections.exists(name):
        client.collections.delete(name)
    try:
        yield WeaviateChunkRepository(client, name)
    finally:
        if client.collections.exists(name):
            client.collections.delete(name)


@pytest.fixture
def integration_repository(
    weaviate_client: WeaviateClient,
) -> Generator[WeaviateChunkRepository]:
    yield from _repository_fixture(
        weaviate_client,
        Settings().weaviate_integration_collection,
    )


@pytest.fixture
def e2e_repository(weaviate_client: WeaviateClient) -> Generator[WeaviateChunkRepository]:
    yield from _repository_fixture(
        weaviate_client,
        Settings().weaviate_e2e_collection,
    )
