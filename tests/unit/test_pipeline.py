from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_service.interfaces import EmbeddingProvider
from rag_service.pipeline import (
    EmbedChunksStep,
    KnowledgeBasePipeline,
    PipelineContext,
    PipelineError,
)
from rag_service.schemas import Chunk, ChunkMetadata


class WrongCountProvider(EmbeddingProvider):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return []

    def embed_query(self, text: str) -> list[float]:
        return [1.0]


class RecordingStep:
    def __init__(self, name: str, calls: list[str], *, fail: bool = False) -> None:
        self.name = name
        self.calls = calls
        self.fail = fail

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.calls.append(self.name)
        if self.fail:
            raise PipelineError(self.name)
        return context


def test_pipeline_runs_steps_in_order_and_stops_on_error() -> None:
    calls: list[str] = []
    pipeline = KnowledgeBasePipeline(
        [
            RecordingStep("first", calls),
            RecordingStep("second", calls, fail=True),
            RecordingStep("third", calls),
        ]
    )

    with pytest.raises(PipelineError, match="second"):
        pipeline.run(PipelineContext(directory=Path("knowledge_base")))

    assert calls == ["first", "second"]


def test_embed_step_validates_number_of_vectors() -> None:
    context = PipelineContext(
        directory=Path("knowledge_base"),
        chunks=(
            Chunk(
                content="text",
                metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=0),
            ),
        ),
    )

    with pytest.raises(PipelineError, match="Ожидалось 1"):
        EmbedChunksStep(WrongCountProvider()).execute(context)
