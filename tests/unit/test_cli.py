from collections.abc import Sequence
from pathlib import Path

from rag_service.cli import build_parser, index_knowledge_base
from rag_service.interfaces import ChunkRepository, EmbeddingProvider
from rag_service.schemas import ChunkSearchResult, RepositorySyncResult, VectorizedChunk


class StubEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.document_inputs: list[str] = []
        self.query_inputs: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_inputs.extend(texts)
        return [[1.0, float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        self.query_inputs.append(text)
        return [1.0, 0.0]


class StubChunkRepository(ChunkRepository):
    def __init__(self) -> None:
        self.chunks: list[VectorizedChunk] = []

    def sync(self, chunks: Sequence[VectorizedChunk]) -> RepositorySyncResult:
        self.chunks = list(chunks)
        return RepositorySyncResult(inserted=len(chunks), updated=0, deleted=0, total=len(chunks))

    def count(self) -> int:
        return len(self.chunks)

    def hybrid_search(
        self, query: str, vector: Sequence[float], limit: int, alpha: float
    ) -> list[ChunkSearchResult]:
        raise NotImplementedError

    def search(self, vector: Sequence[float], limit: int) -> list[ChunkSearchResult]:
        chunk = self.chunks[0]
        return [
            ChunkSearchResult(
                uuid="00000000-0000-0000-0000-000000000001",
                content=chunk.content,
                metadata=chunk.metadata,
                distance=0.0,
            )
        ][:limit]


def test_parser_accepts_directory_and_chunk_settings() -> None:
    args = build_parser().parse_args(["custom-kb", "--chunk-size", "300", "--chunk-overlap", "60"])

    assert args.directory == Path("custom-kb")
    assert args.chunk_size == 300
    assert args.chunk_overlap == 60


def test_index_pipeline_uses_stub_repository(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("Один документ", encoding="utf-8")
    provider = StubEmbeddingProvider()
    repository = StubChunkRepository()

    result = index_knowledge_base(tmp_path, 500, 100, provider, repository)

    assert len(result.documents) == 1
    assert len(result.chunks) == 1
    assert result.verified_count == 1
    assert repository.chunks[0].metadata.document_id == "one"
    assert repository.chunks[0].vector == (1.0, 0.0)
    assert provider.document_inputs == ["Один документ"]
    assert provider.query_inputs == ["Один документ"]
