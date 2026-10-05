"""Типизированная конфигурация приложения из окружения и локального ``.env``."""

from pathlib import Path
from typing import Annotated

from pydantic import AnyHttpUrl, Field, SecretStr, StringConstraints, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_service.retrieval_quality import DEFAULT_MAX_DISTANCE

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
    retrieval_max_distance: float = Field(
        default=DEFAULT_MAX_DISTANCE, ge=0, le=2, allow_inf_nan=False
    )

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


class LLMSettings(BaseSettings):
    """Обязательные настройки только для команд генерации."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    llm_api_key: SecretStr
    llm_base_url: AnyHttpUrl
    llm_model: NonEmptyString
    llm_retry_max_attempts: int = Field(default=3, ge=1)
    llm_retry_base_delay_seconds: float = Field(default=0.5, ge=0, allow_inf_nan=False)

    @field_validator("llm_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("LLM_API_KEY не может быть пустым")
        return value
