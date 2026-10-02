from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_service.settings import PROJECT_ROOT, Settings


def test_settings_resolve_relative_model_path_from_project_root() -> None:
    settings = Settings(
        embedding_model_path=Path("models/test-model"),
        weaviate_integration_collection="IntegrationChunks",
    )

    assert settings.embedding_model_path == PROJECT_ROOT / Path("models/test-model")
    assert settings.weaviate_integration_collection == "IntegrationChunks"


def test_environment_has_priority_over_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEAVIATE_COLLECTION", "CollectionFromEnvironment")

    settings = Settings()

    assert settings.weaviate_collection == "CollectionFromEnvironment"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"embedding_batch_size": 0}, "greater than 0"),
        ({"weaviate_grpc_port": 65536}, "less than or equal to 65535"),
        ({"embedding_device": "  "}, "at least 1 character"),
        ({"weaviate_url": "localhost:8080"}, "URL scheme should be 'http' or 'https'"),
    ],
)
def test_settings_reject_invalid_values(values: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings.model_validate(values)
