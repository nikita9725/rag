from rag_service.repository import chunk_uuid
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
