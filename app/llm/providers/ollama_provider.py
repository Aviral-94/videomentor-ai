from typing import Any, Optional
from langchain_ollama import ChatOllama

from app.llm.config import config
from app.llm.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Primary local inference provider via Ollama."""

    @property
    def name(self) -> str:
        return "ollama"

    def is_configured(self) -> bool:
        return bool(config.ollama_base_url)

    def get_model(
        self,
        temperature: float = 0,
        json_mode: bool = False,
        format_schema: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        timeout_val = timeout if timeout is not None else config.timeout

        kwargs = {
            "model": config.ollama_model,
            "base_url": config.ollama_base_url,
            "temperature": temperature,
            "client_kwargs": {"timeout": timeout_val},
        }

        if json_mode:
            if format_schema is not None:
                kwargs["format"] = format_schema
            else:
                kwargs["format"] = "json"

        return ChatOllama(**kwargs)

