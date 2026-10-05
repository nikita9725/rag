from unittest.mock import Mock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, OpenAI
from pydantic import SecretStr

from rag_service.repositories import LLMError, OpenAILLMRepository
from rag_service.settings import LLMSettings


def config() -> LLMSettings:
    return LLMSettings(
        llm_api_key=SecretStr("test-secret"),
        llm_base_url="https://example.com/v1",
        llm_model="test-model",
    )


def test_request_and_close() -> None:
    client = Mock(spec=OpenAI)
    client.chat.completions.create.return_value.choices = [Mock(message=Mock(content=" {} "))]
    repository = OpenAILLMRepository(config(), client=client)
    assert repository.complete("system", "user") == "{}"
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "test-model"
    assert request["messages"][1]["content"] == "user"
    assert request["response_format"] == {"type": "json_object"}
    repository.close()
    client.close.assert_called_once()


@pytest.mark.parametrize("status", [429, 500, 401])
def test_status_retry_policy(status: int) -> None:
    client = Mock(spec=OpenAI)
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.com"))
    client.chat.completions.create.side_effect = APIStatusError(
        "secret upstream details", response=response, body=None
    )
    sleeper = Mock()
    repository = OpenAILLMRepository(config(), client=client, sleeper=sleeper)
    with pytest.raises(LLMError) as error:
        repository.complete("system", "user")
    assert "secret" not in str(error.value)
    assert client.chat.completions.create.call_count == (1 if status == 401 else 3)
    assert sleeper.call_count == (0 if status == 401 else 2)


def test_connection_recovers() -> None:
    client = Mock(spec=OpenAI)
    client.chat.completions.create.side_effect = [
        APIConnectionError(request=httpx.Request("POST", "https://example.com")),
        Mock(choices=[Mock(message=Mock(content="{}"))]),
    ]
    sleeper = Mock()
    assert OpenAILLMRepository(config(), client=client, sleeper=sleeper).complete("s", "u") == "{}"
    sleeper.assert_called_once_with(0.5)


def test_empty_response_is_not_retried() -> None:
    client = Mock(spec=OpenAI)
    client.chat.completions.create.return_value.choices = []
    with pytest.raises(LLMError, match="пустой"):
        OpenAILLMRepository(config(), client=client).complete("s", "u")
    client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("attempts", [1, 2, 4])
def test_retry_settings_control_attempts_and_backoff(attempts: int) -> None:
    settings = config().model_copy(
        update={
            "llm_retry_max_attempts": attempts,
            "llm_retry_base_delay_seconds": 0.25,
        }
    )
    client = Mock(spec=OpenAI)
    client.chat.completions.create.side_effect = APIConnectionError(
        request=httpx.Request("POST", "https://example.com")
    )
    delays: list[float] = []
    repository = OpenAILLMRepository(settings, client=client, sleeper=delays.append)
    with pytest.raises(LLMError):
        repository.complete("system", "user")
    assert client.chat.completions.create.call_count == attempts
    assert delays == [0.25 * 2**i for i in range(attempts - 1)]
