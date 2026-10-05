import pytest

from rag_service.repositories import RepositoryError, chunk_uuid
from rag_service.repositories.weaviate import cosine_distance
from rag_service.schemas import ChunkMetadata, VectorizedChunk


def test_chunk_uuid_is_stable_and_depends_on_chunk_identity() -> None:
    first = VectorizedChunk(
        content="old text",
        metadata=ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=0),
        vector=(1.0, 0.0),
    )
    changed = first.model_copy(update={"content": "new text", "vector": (0.0, 1.0)})
    next_chunk = first.model_copy(
        update={"metadata": ChunkMetadata(document_id="doc", source_name="doc.txt", chunk_id=1)}
    )

    assert chunk_uuid("KnowledgeChunk", first) == chunk_uuid("KnowledgeChunk", changed)
    assert chunk_uuid("KnowledgeChunk", first) != chunk_uuid("KnowledgeChunk", next_chunk)


@pytest.mark.parametrize(
    "left,right,expected",
    [([1.0, 0.0], [2.0, 0.0], 0.0), ([1.0, 0.0], [0.0, 1.0], 1.0), ([1.0], [-1.0], 2.0)],
)
def test_cosine_distance(left: list[float], right: list[float], expected: float) -> None:
    assert cosine_distance(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    "left,right",
    [
        ([], []),
        ([1.0], [1.0, 2.0]),
        ([0.0], [1.0]),
        ([float("nan")], [1.0]),
        ([float("inf")], [1.0]),
    ],
)
def test_invalid_vectors_fail(left: list[float], right: list[float]) -> None:
    with pytest.raises(RepositoryError):
        cosine_distance(left, right)
