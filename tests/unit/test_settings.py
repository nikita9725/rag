import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from rag_service.settings import PROJECT_ROOT, LLMSettings, Settings


def test_settings_resolve_relative_model_path_from_project_root() -> None:
    settings = Settings(
        embedding_model_path=Path("models/test-model"),
        weaviate_integration_collection="IntegrationChunks",
    )

    assert settings.embedding_model_path == PROJECT_ROOT / Path("models/test-model")
    assert settings.weaviate_integration_collection == "IntegrationChunks"


def test_environment_has_priority_over_dotenv() -> None:
    environment = dict(os.environ, WEAVIATE_COLLECTION="CollectionFromEnvironment")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rag_service.settings import Settings; print(Settings().weaviate_collection)",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "CollectionFromEnvironment"


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


def test_llm_settings_hide_secret_and_require_configuration() -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"}
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rag_service.settings import LLMSettings; LLMSettings(_env_file=None)",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Field required" in result.stderr
    settings = LLMSettings(
        llm_api_key=SecretStr("private-key"),
        llm_base_url="https://example.com",
        llm_model="test",
    )
    assert "private-key" not in repr(settings)
    with pytest.raises(ValidationError):
        LLMSettings(
            llm_api_key=SecretStr(" "),
            llm_base_url="https://example.com",
            llm_model="test",
        )


@pytest.mark.parametrize(
    "values",
    [
        {"llm_retry_max_attempts": 0},
        {"llm_retry_base_delay_seconds": -1},
        {"llm_retry_base_delay_seconds": float("inf")},
        {"llm_retry_base_delay_seconds": float("nan")},
    ],
)
def test_retry_settings_reject_invalid_values(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LLMSettings.model_validate(
            {
                "llm_api_key": "test",
                "llm_base_url": "https://example.com",
                "llm_model": "test",
                **values,
            }
        )
