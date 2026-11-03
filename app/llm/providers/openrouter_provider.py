from typing import Any, Optional
from langchain_openai import ChatOpenAI

from app.llm.config import config
from app.llm.providers.base import BaseLLMProvider


class OpenRouterProvider(BaseLLMProvider):
    """Cloud fallback provider 3: OpenRouter API."""

    @property
    def name(self) -> str:
        return "openrouter"

    def is_configured(self) -> bool:
        return bool(config.openrouter_api_key)

    def get_model(
        self,
        temperature: float = 0,
        json_mode: bool = False,
        format_schema: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if not self.is_configured():
            raise RuntimeError("OpenRouterProvider is not configured: OPENROUTER_API_KEY is missing.")

        timeout_val = timeout if timeout is not None else config.timeout

        kwargs = {
            "model": config.openrouter_model,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": config.openrouter_api_key,
            "temperature": temperature,
            "timeout": timeout_val,
            "max_retries": 1,
            "default_headers": {
                "HTTP-Referer": "https://videomentor.ai",
                "X-Title": "VideoMentor AI",
            },
        }

        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        return ChatOpenAI(**kwargs)

