"""День 7: восемь demo-вопросов через CLI с реальными Weaviate и LLM."""

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel

from rag_service.answer_cli import main
from rag_service.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from rag_service.cli import index_knowledge_base
from rag_service.embeddings import LocalEmbeddingProvider
from rag_service.generation import RAGAnswer, RAGService
from rag_service.generation_content import SYSTEM_PROMPT
from rag_service.repositories import LLMRepository, OpenAILLMRepository, WeaviateChunkRepository
from rag_service.retrieval import DEFAULT_TOP_K, RetrievalService
from rag_service.settings import PROJECT_ROOT, LLMSettings, Settings

pytestmark = pytest.mark.e2e


class DemoCase(BaseModel):
    id: str
    query: str
    kind: Literal["answerable", "partial", "unanswerable"]
    expected_sources: list[str]


class RecordingLLM:
    def __init__(self, repository: LLMRepository) -> None:
        self.repository = repository
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.repository.complete(system_prompt, user_prompt)


class RecordingRAG(RAGService):
    result: RAGAnswer | None = None

    def answer(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        mode: Literal["semantic", "hybrid"] = "semantic",
        alpha: float = 0.5,
    ) -> RAGAnswer:
        self.result = super().answer(query, top_k, mode, alpha)
        return self.result


def test_demo_cli_with_real_context(
    real_embedding_provider: LocalEmbeddingProvider,
    e2e_repository: WeaviateChunkRepository,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    settings, llm_settings = Settings(), LLMSettings()
    first = index_knowledge_base(
        PROJECT_ROOT / "knowledge_base",
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
        real_embedding_provider,
        e2e_repository,
    )
    second = index_knowledge_base(
        PROJECT_ROOT / "knowledge_base",
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
        real_embedding_provider,
        e2e_repository,
    )
    assert first.sync_result is not None and second.sync_result is not None
    assert second.sync_result.inserted == second.sync_result.deleted == 0
    assert first.sync_result.total == second.sync_result.total == len(first.chunks)
    cases = [
        DemoCase.model_validate(item)
        for item in json.loads(
            (PROJECT_ROOT / "tests/data/demo_questions.json").read_text(encoding="utf-8")
        )
    ]
    repository = OpenAILLMRepository(llm_settings)
    llm = RecordingLLM(repository)
    service = RecordingRAG(
        RetrievalService(
            real_embedding_provider,
            e2e_repository,
            max_distance=settings.retrieval_max_distance,
        ),
        llm,
    )

    @contextmanager
    def resources() -> Iterator[tuple[RAGService, LLMRepository]]:
        yield service, llm

    observations: list[dict[str, object]] = []
    try:
        for case in cases:
            calls = llm.calls
            main([case.query, "--show-context"], service_factory=resources)
            output = capsys.readouterr().out
            assert service.result is not None
            observations.append(
                {
                    "case": case.model_dump(mode="json"),
                    "result": service.result.model_dump(mode="json"),
                    "llm_calls": llm.calls - calls,
                    "cli_output": output,
                }
            )
    finally:
        repository.close()
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "embedding_model": settings.embedding_model_id,
        "embedding_revision": settings.embedding_model_revision,
        "llm_model": llm_settings.llm_model,
        "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "corpus_sha256": {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in sorted((PROJECT_ROOT / "knowledge_base").glob("*.txt"))
        },
        "configuration": {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
            "top_k": DEFAULT_TOP_K,
            "mode": "semantic",
            "max_distance": settings.retrieval_max_distance,
        },
        "index_sync": {
            "first": first.sync_result.model_dump(),
            "second": second.sync_result.model_dump(),
        },
        "observations": observations,
    }
    path = Path(os.environ.get("DAY07_REPORT_PATH", str(tmp_path / "day07-demo.json")))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for item in observations:
        case = DemoCase.model_validate(item["case"])
        result = RAGAnswer.model_validate(item["result"])
        assert result.insufficient_context == (case.kind != "answerable"), case.id
        assert item["llm_calls"] == int(bool(result.context)), case.id
        assert set(case.expected_sources) <= {
            source.metadata.source_name for source in result.sources
        }, case.id
        assert all(
            result.context[n - 1] == source
            for n, source in zip(result.source_ids, result.sources, strict=True)
        )
        assert all(
            chunk.distance is not None and chunk.distance <= settings.retrieval_max_distance
            for chunk in result.context
        )
        assert set(map(int, re.findall(r"\[(\d+)\]", result.answer))) <= set(result.source_ids)
        if case.kind == "unanswerable":
            assert not result.sources and not result.source_ids, case.id
        if case.id == "version":
            assert result.context, "Проверяем отказ даже при тематически близком контексте"
        if case.id == "cooking":
            assert not result.context and item["llm_calls"] == 0
