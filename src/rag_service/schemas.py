"""Модели данных для документов базы знаний."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, computed_field


class Document(BaseModel):
    """Локальный документ после очистки."""

    model_config = ConfigDict(frozen=True)

    source: Path
    content: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def char_count(self) -> int:
        """Количество символов в очищенном тексте."""

        return len(self.content)


class ChunkMetadata(BaseModel):
    """Метаданные, связывающие чанк с исходным документом."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source_name: str
    chunk_id: int


class Chunk(BaseModel):
    """Фрагмент документа, подготовленный для последующего retrieval."""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: ChunkMetadata

    @computed_field  # type: ignore[prop-decorator]
    @property
    def char_count(self) -> int:
        """Количество символов в чанке."""

        return len(self.content)
