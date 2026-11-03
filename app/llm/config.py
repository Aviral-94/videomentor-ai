import os
from typing import List


def _to_bool(val: str, default: bool = True) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "y", "on")


class LLMConfig:
    """Configuration for Hybrid Multi-LLM provider orchestration."""

    def __init__(self):
        self.reload()

    def reload(self):
        # Primary provider selection
        self.primary_provider = os.getenv("LLM_PRIMARY", "ollama").strip().lower()

        # Fallback configuration
        self.fallback_enabled = _to_bool(os.getenv("LLM_FALLBACK_ENABLED", "true"), default=True)
        fallback_str = os.getenv("LLM_FALLBACK_ORDER", "groq,gemini,openrouter")
        self.fallback_order: List[str] = [
            p.strip().lower() for p in fallback_str.split(",") if p.strip()
        ]

        # Ollama configuration
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b").strip()
        self.ollama_base_url = (
            os.getenv("OLLAMA_BASE_URL")
            or os.getenv("OLLAMA_HOST")
            or "http://localhost:11434"
        ).strip().rstrip("/")

        # Groq configuration
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

        # Google Gemini configuration
        self.gemini_api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        ).strip()
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

        # OpenRouter configuration
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openrouter_model = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()

        # Timeouts and retries
        try:
            self.timeout = float(os.getenv("LLM_TIMEOUT", "30.0"))
        except ValueError:
            self.timeout = 30.0

        try:
            self.total_deadline = float(os.getenv("TOTAL_DEADLINE", "40.0"))
        except ValueError:
            self.total_deadline = 40.0

        try:
            self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "1"))
        except ValueError:
            self.max_retries = 1

    def is_provider_configured(self, provider_name: str) -> bool:
        """Check if provider credentials/prerequisites are available."""
        p = provider_name.strip().lower()
        if p == "ollama":
            # Ollama is always considered configured (local or tunnel URL)
            return bool(self.ollama_base_url)
        elif p == "groq":
            return bool(self.groq_api_key)
        elif p == "gemini":
            return bool(self.gemini_api_key)
        elif p == "openrouter":
            return bool(self.openrouter_api_key)
        return False

    def get_provider_chain(self) -> List[str]:
        """
        Returns the ordered list of providers to attempt:
        Primary provider first, followed by configured fallback providers.
        """
        chain = [self.primary_provider]
        if self.fallback_enabled:
            for fb in self.fallback_order:
                if fb not in chain and self.is_provider_configured(fb):
                    chain.append(fb)
        return chain


config = LLMConfig()

