import json
from pathlib import Path

import pytest

from rag_service.cli import index_knowledge_base
from rag_service.embeddings import LocalEmbeddingProvider
from rag_service.repository import WeaviateChunkRepository
from rag_service.retrieval import RetrievalService, SearchMode
from rag_service.settings import PROJECT_ROOT

pytestmark = pytest.mark.e2e


def test_five_questions_in_both_modes(
    real_embedding_provider: LocalEmbeddingProvider,
    e2e_repository: WeaviateChunkRepository,
) -> None:
    index_knowledge_base(
        PROJECT_ROOT / "knowledge_base", 500, 100, real_embedding_provider, e2e_repository
    )
    cases = json.loads(
        (Path(__file__).parents[1] / "data/retrieval_questions.json").read_text(encoding="utf-8")
    )
    service = RetrievalService(real_embedding_provider, e2e_repository)
    modes: tuple[SearchMode, ...] = ("semantic", "hybrid")
    for mode in modes:
        for case in cases:
            results = service.retrieve(case["query"], top_k=3, mode=mode)
            assert len(results) == 3
            assert all(result.content and result.metadata.source_name for result in results)
            assert any(
                result.metadata.source_name == case["source_name"]
                and case["expected_fragment"] in result.content
                for result in results
            ), (mode, case["query"], results)
            if mode == "semantic":
                distances = [result.distance for result in results]
                assert all(distance is not None for distance in distances)
                assert distances == sorted(distances, key=lambda distance: float(distance or 0))
            else:
                assert all(result.score is not None for result in results)
