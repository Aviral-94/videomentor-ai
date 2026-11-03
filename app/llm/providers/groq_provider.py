from typing import Any, Optional
from langchain_openai import ChatOpenAI

from app.llm.config import config
from app.llm.providers.base import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """Cloud fallback provider 1: Groq API."""

    @property
    def name(self) -> str:
        return "groq"

    def is_configured(self) -> bool:
        return bool(config.groq_api_key)

    def get_model(
        self,
        temperature: float = 0,
        json_mode: bool = False,
        format_schema: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if not self.is_configured():
            raise RuntimeError("GroqProvider is not configured: GROQ_API_KEY is missing.")

        timeout_val = timeout if timeout is not None else config.timeout

        kwargs = {
            "model": config.groq_model,
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": config.groq_api_key,
            "temperature": temperature,
            "timeout": timeout_val,
            "max_retries": 1,
        }

        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

        return ChatOpenAI(**kwargs)

