from pathlib import Path

import pytest

from rag_service.cli import index_knowledge_base
from rag_service.embeddings import LocalEmbeddingProvider
from rag_service.repositories import WeaviateChunkRepository

pytestmark = pytest.mark.e2e


def test_full_pipeline_is_idempotent_and_removes_stale_chunks(
    tmp_path: Path,
    real_embedding_provider: LocalEmbeddingProvider,
    e2e_repository: WeaviateChunkRepository,
) -> None:
    first_path = tmp_path / "first.txt"
    second_path = tmp_path / "second.txt"
    first_path.write_text("Weaviate — векторная база данных.", encoding="utf-8")
    second_path.write_text("Embedding описывает смысл текста.", encoding="utf-8")

    first = index_knowledge_base(tmp_path, 500, 100, real_embedding_provider, e2e_repository)
    repeated = index_knowledge_base(tmp_path, 500, 100, real_embedding_provider, e2e_repository)

    assert first.sync_result is not None
    assert first.sync_result.inserted == 2
    assert repeated.sync_result is not None
    assert repeated.sync_result.inserted == 0
    assert repeated.sync_result.updated == 2
    assert repeated.verified_count == 2

    first_path.write_text("Weaviate хранит чанки и их embeddings.", encoding="utf-8")
    second_path.unlink()
    (tmp_path / "third.txt").write_text(
        "Повторная индексация не создаёт дубликаты.", encoding="utf-8"
    )

    changed = index_knowledge_base(tmp_path, 500, 100, real_embedding_provider, e2e_repository)

    assert changed.sync_result is not None
    assert changed.sync_result.inserted == 1
    assert changed.sync_result.updated == 1
    assert changed.sync_result.deleted == 1
    assert changed.sync_result.total == 2
    assert changed.verified_count == 2
    assert changed.verification_results
