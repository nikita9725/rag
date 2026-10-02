"""Типизированная конфигурация приложения из окружения и локального ``.env``."""

from pathlib import Path
from typing import Annotated

from pydantic import AnyHttpUrl, Field, StringConstraints, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Settings(BaseSettings):
    """Настройки с приоритетом переменных окружения над корневым ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    embedding_model_id: NonEmptyString = "intfloat/multilingual-e5-small"
    embedding_model_revision: NonEmptyString = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    embedding_model_path: Path = PROJECT_ROOT / "models/multilingual-e5-small"
    embedding_device: NonEmptyString = "cpu"
    embedding_batch_size: int = Field(default=32, gt=0)

    weaviate_url: str = "http://localhost:8080"
    weaviate_grpc_port: int = Field(default=50051, ge=1, le=65535)
    weaviate_collection: NonEmptyString = "KnowledgeChunk"
    weaviate_integration_collection: NonEmptyString = "KnowledgeChunkIntegration"
    weaviate_e2e_collection: NonEmptyString = "KnowledgeChunkE2E"

    @field_validator("embedding_model_path")
    @classmethod
    def resolve_model_path(cls, value: Path) -> Path:
        """Разрешить относительный путь от корня проекта, а не текущего каталога."""

        if value.is_absolute():
            return value
        return PROJECT_ROOT / value

    @field_validator("weaviate_url")
    @classmethod
    def validate_weaviate_url(cls, value: str) -> str:
        """Принять только непустой HTTP(S) URL с хостом."""

        normalized = value.strip()
        AnyHttpUrl(normalized)
        return normalized.rstrip("/")
