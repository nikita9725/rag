"""CLI поиска контекста в существующем индексе."""

import argparse
from collections.abc import Sequence

from pydantic import ValidationError
from weaviate.exceptions import WeaviateBaseError

from rag_service.embeddings import LocalEmbeddingProvider, LocalModelError, LocalModelManager
from rag_service.repository import RepositoryError, WeaviateChunkRepository, connect_to_weaviate
from rag_service.retrieval import RetrievalService, validate_query
from rag_service.settings import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Найти контекст в Weaviate")
    parser.add_argument("query", help="вопрос пользователя")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mode", choices=["semantic", "hybrid"], default="semantic")
    parser.add_argument("--alpha", type=float, default=None, help="вес vector search в hybrid")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.alpha is not None and args.mode != "hybrid":
            raise ValueError("--alpha разрешён только для --mode hybrid")
        alpha = args.alpha if args.alpha is not None else 0.5
        query = validate_query(args.query, args.top_k, args.mode, alpha)
        settings = Settings()
        manager = LocalModelManager(
            settings.embedding_model_id,
            settings.embedding_model_revision,
            settings.embedding_model_path,
        )
        provider = LocalEmbeddingProvider(
            manager.ensure_downloaded(),
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
        client = connect_to_weaviate(settings.weaviate_url, settings.weaviate_grpc_port)
        try:
            repository = WeaviateChunkRepository(client, settings.weaviate_collection)
            results = RetrievalService(provider, repository).retrieve(
                query, args.top_k, args.mode, alpha
            )
        finally:
            client.close()
    except (
        ValueError,
        ValidationError,
        LocalModelError,
        RepositoryError,
        WeaviateBaseError,
        OSError,
    ) as error:
        parser.error(str(error))

    print(f"Вопрос: {query}\nРежим: {args.mode}, top-k: {args.top_k}")
    if args.mode == "hybrid":
        print(f"Alpha: {alpha}")
    if not results:
        print("Контекст не найден: коллекция пуста или поиск не вернул результатов.")
    for rank, result in enumerate(results, 1):
        metric = (
            f"distance={result.distance:.6f}"
            if result.distance is not None
            else f"score={result.score:.6f}"
            if result.score is not None
            else "метрика отсутствует"
        )
        print(
            f"\n{rank}. source_name={result.metadata.source_name} "
            f"chunk_id={result.metadata.chunk_id} {metric}\n{result.content}"
        )


if __name__ == "__main__":
    main()
