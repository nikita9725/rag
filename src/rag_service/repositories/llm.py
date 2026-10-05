"""OpenAI-совместимый адаптер генерации с ограниченными повторами."""

import logging
import time
from collections.abc import Callable

from openai import APIConnectionError, APIError, APIStatusError, OpenAI

from rag_service.repositories.interfaces import LLMRepository
from rag_service.settings import LLMSettings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Ошибка вызова модели или проверки её ответа."""


class OpenAILLMRepository(LLMRepository):
    def __init__(
        self,
        settings: LLMSettings,
        *,
        client: OpenAI | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._max_attempts = settings.llm_retry_max_attempts
        self._base_delay = settings.llm_retry_base_delay_seconds
        self._model = settings.llm_model
        self._client = client or OpenAI(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=str(settings.llm_base_url),
            timeout=60.0,
            max_retries=0,
        )
        self._sleeper = sleeper

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    stream=False,
                )
            except APIError as error:
                transient = isinstance(error, APIConnectionError) or (
                    isinstance(error, APIStatusError)
                    and (error.status_code == 429 or error.status_code >= 500)
                )
                if not transient or attempt == self._max_attempts:
                    raise LLMError(f"Ошибка LLM API после {attempt} попыток") from None
                logger.warning(
                    "Временная ошибка LLM API: попытка %d/%d", attempt, self._max_attempts
                )
                self._sleeper(self._base_delay * 2 ** (attempt - 1))
                continue
            content = response.choices[0].message.content if response.choices else None
            if not content or not content.strip():
                raise LLMError("LLM вернула пустой ответ")
            return content.strip()
        raise AssertionError("Цикл повторов должен вернуть ответ или ошибку")

    def close(self) -> None:
        self._client.close()
