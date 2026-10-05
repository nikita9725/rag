"""CLI полного RAG pipeline и сравнения с ответом без retrieval."""

import logging
from collections.abc import Sequence

from pydantic import ValidationError
from weaviate.exceptions import WeaviateBaseError

from rag_service.application import GenerationFactory, open_generation
from rag_service.embeddings import LocalModelError
from rag_service.generation import answer_without_retrieval
from rag_service.pipeline import PipelineError
from rag_service.query_cli import build_parser
from rag_service.repositories import (
    LLMError,
    RepositoryError,
)
from rag_service.retrieval import validate_query


def main(
    argv: Sequence[str] | None = None, *, service_factory: GenerationFactory = open_generation
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    parser.description = "Ответить на вопрос по контексту из Weaviate"
    parser.add_argument("--compare", action="store_true", help="сравнить с ответом без retrieval")
    args = parser.parse_args(argv)
    try:
        if args.alpha is not None and args.mode != "hybrid":
            raise ValueError("--alpha разрешён только для --mode hybrid")
        alpha = args.alpha if args.alpha is not None else 0.5
        query = validate_query(args.query, args.top_k, args.mode, alpha)
        with service_factory() as (service, llm):
            result = service.answer(query, args.top_k, args.mode, alpha)
            baseline = answer_without_retrieval(query, llm) if args.compare else None
    except (
        ValueError,
        ValidationError,
        LocalModelError,
        RepositoryError,
        WeaviateBaseError,
        LLMError,
        PipelineError,
        OSError,
    ) as error:
        parser.error(str(error))
    print(f"Вопрос: {query}\nРежим: {args.mode}, top-k: {args.top_k}")
    if baseline is not None:
        print(f"\nБез retrieval:\n{baseline}")
    print(f"\nОтвет с retrieval:\n{result.answer}")
    if result.insufficient_context:
        print("Контекст недостаточен для полного ответа.")
    print("\nИсточники:")
    for source_id, source in zip(result.source_ids, result.sources, strict=True):
        print(
            f"[{source_id}] source_name={source.metadata.source_name} "
            f"chunk_id={source.metadata.chunk_id} uuid={source.uuid}"
        )
    if not result.sources:
        print("Нет использованных источников.")
