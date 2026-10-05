"""CLI поиска контекста в существующем индексе."""

import argparse
from collections.abc import Sequence

from pydantic import ValidationError
from weaviate.exceptions import WeaviateBaseError

from rag_service.application import RetrievalFactory, open_retrieval
from rag_service.embeddings import LocalModelError
from rag_service.repositories import RepositoryError
from rag_service.retrieval import DEFAULT_TOP_K, validate_query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Найти контекст в Weaviate")
    parser.add_argument("query", help="вопрос пользователя")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"число кандидатов (по умолчанию: {DEFAULT_TOP_K})",
    )
    parser.add_argument("--mode", choices=["semantic", "hybrid"], default="semantic")
    parser.add_argument("--alpha", type=float, default=None, help="вес vector search в hybrid")
    return parser


def main(
    argv: Sequence[str] | None = None, *, service_factory: RetrievalFactory = open_retrieval
) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.alpha is not None and args.mode != "hybrid":
            raise ValueError("--alpha разрешён только для --mode hybrid")
        alpha = args.alpha if args.alpha is not None else 0.5
        query = validate_query(args.query, args.top_k, args.mode, alpha)
        with service_factory() as service:
            results = service.retrieve(query, args.top_k, args.mode, alpha)
            max_distance = service.max_distance
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
    print(f"Порог cosine distance: {max_distance}")
    if args.mode == "hybrid":
        print(f"Alpha: {alpha}")
    if not results:
        print("Контекст не найден: нет фрагментов, прошедших порог надёжности.")
    for rank, result in enumerate(results, 1):
        metric = " ".join(
            f"{name}={value:.6f}"
            for name, value in (("distance", result.distance), ("score", result.score))
            if value is not None
        )
        print(
            f"\n{rank}. source_name={result.metadata.source_name} "
            f"chunk_id={result.metadata.chunk_id} {metric}\n{result.content}"
        )


if __name__ == "__main__":
    main()
