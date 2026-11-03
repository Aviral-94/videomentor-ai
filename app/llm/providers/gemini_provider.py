from typing import Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

from app.llm.config import config
from app.llm.providers.base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """Cloud fallback provider 2: Google Gemini API."""

    @property
    def name(self) -> str:
        return "gemini"

    def is_configured(self) -> bool:
        return bool(config.gemini_api_key)

    def get_model(
        self,
        temperature: float = 0,
        json_mode: bool = False,
        format_schema: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        if not self.is_configured():
            raise RuntimeError("GeminiProvider is not configured: GEMINI_API_KEY / GOOGLE_API_KEY is missing.")

        timeout_val = timeout if timeout is not None else config.timeout

        kwargs = {
            "model": config.gemini_model,
            "google_api_key": config.gemini_api_key,
            "temperature": temperature,
            "timeout": timeout_val,
            "max_retries": 1,
        }

        # Handle JSON mode if requested
        if json_mode:
            # ChatGoogleGenerativeAI supports response_mime_type via generation_config
            kwargs["generation_config"] = {"response_mime_type": "application/json"}

        return ChatGoogleGenerativeAI(**kwargs)

