from typing import Dict, Type
from app.llm.providers.base import BaseLLMProvider, is_retryable_error
from app.llm.providers.ollama_provider import OllamaProvider
from app.llm.providers.groq_provider import GroqProvider
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.openrouter_provider import OpenRouterProvider

PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {
    "ollama": OllamaProvider,
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
}

__all__ = [
    "BaseLLMProvider",
    "is_retryable_error",
    "OllamaProvider",
    "GroqProvider",
    "GeminiProvider",
    "OpenRouterProvider",
    "PROVIDER_REGISTRY",
]

