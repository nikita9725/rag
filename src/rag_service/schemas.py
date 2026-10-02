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
