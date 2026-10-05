"""CLI загрузки модели и пошаговой индексации базы знаний."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from rag_service.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from rag_service.embeddings import LocalEmbeddingProvider, LocalModelError, LocalModelManager
from rag_service.interfaces import EmbeddingProvider
from rag_service.pipeline import (
    ChunkDocumentsStep,
    EmbedChunksStep,
    KnowledgeBasePipeline,
    LoadDocumentsStep,
    PipelineContext,
    PipelineError,
    SyncChunksStep,
    VerifyIndexStep,
)
from rag_service.repositories import (
    ChunkRepository,
    RepositoryError,
    WeaviateChunkRepository,
    connect_to_weaviate,
)
from rag_service.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Индексировать локальную базу знаний")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("knowledge_base"),
        help="папка с UTF-8 TXT-документами (по умолчанию: knowledge_base)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"максимальный размер чанка (по умолчанию: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"перекрытие чанков (по умолчанию: {DEFAULT_CHUNK_OVERLAP})",
    )
    return parser


def build_pipeline(
    provider: EmbeddingProvider, repository: ChunkRepository
) -> KnowledgeBasePipeline:
    """Собрать production-последовательность шагов."""

    return KnowledgeBasePipeline(
        [
            LoadDocumentsStep(),
            ChunkDocumentsStep(),
            EmbedChunksStep(provider),
            SyncChunksStep(repository),
            VerifyIndexStep(provider, repository),
        ]
    )


def index_knowledge_base(
    directory: Path,
    chunk_size: int,
    chunk_overlap: int,
    provider: EmbeddingProvider,
    repository: ChunkRepository,
) -> PipelineContext:
    """Выполнить полный pipeline с переданными стратегиями."""

    return build_pipeline(provider, repository).run(
        PipelineContext(
            directory=directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Индексировать базу знаний, при необходимости скачав модель."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    try:
        settings = Settings()
        manager = LocalModelManager(
            settings.embedding_model_id,
            settings.embedding_model_revision,
            settings.embedding_model_path,
        )
        args = parser.parse_args(argv)
        model_path = manager.ensure_downloaded()
        provider = LocalEmbeddingProvider(
            model_path,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
        client = connect_to_weaviate(settings.weaviate_url, settings.weaviate_grpc_port)
        try:
            repository = WeaviateChunkRepository(client, settings.weaviate_collection)
            result = index_knowledge_base(
                args.directory,
                args.chunk_size,
                args.chunk_overlap,
                provider,
                repository,
            )
        finally:
            client.close()
    except (
        ValidationError,
        FileNotFoundError,
        NotADirectoryError,
        ValueError,
        LocalModelError,
        PipelineError,
        RepositoryError,
    ) as error:
        parser.error(str(error))

    sync = result.sync_result
    if sync is None:
        raise AssertionError("Успешный pipeline обязан вернуть статистику синхронизации")
    total_characters = sum(document.char_count for document in result.documents)
    print(f"Документов: {len(result.documents)}, символов: {total_characters}")
    print(
        f"Чанков: {len(result.chunks)} "
        f"(chunk_size={result.chunk_size}, chunk_overlap={result.chunk_overlap})"
    )
    print(
        f"Weaviate: inserted={sync.inserted}, updated={sync.updated}, "
        f"deleted={sync.deleted}, total={sync.total}"
    )
    print(f"Проверка retrieval: найдено {len(result.verification_results)}")
