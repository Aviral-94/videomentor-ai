from abc import ABC, abstractmethod
from typing import Any, Optional


def is_retryable_error(exc: Exception) -> bool:
    """
    Determines whether an exception is an infrastructure/availability failure
    warranting fallback to the next provider.

    Fallback triggers on:
    - Connection refused / host unreachable
    - Request or socket timeouts
    - HTTP 429 Rate limits / quota exhaustion
    - HTTP 500, 502, 503, 504 server outages

    Does NOT trigger on:
    - Validation errors (ValueError, TypeError)
    - User input formatting errors
    - Normal model refusals
    """
    exc_name = type(exc).__name__
    exc_str = str(exc).lower()

    # Known transient error class names
    retryable_class_names = (
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "RemoteProtocolError",
        "ConnectionError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
        "TimeoutError",
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailable",
        "ResourceExhausted",
        "TooManyRequests",
        "BadGateway",
        "GatewayTimeout",
        "OverloadedError",
    )

    if any(cls_name in exc_name for cls_name in retryable_class_names):
        return True

    # Error message heuristic inspection
    retryable_substrings = (
        "connection refused",
        "failed to connect",
        "connect error",
        "timed out",
        "timeout",
        "rate limit",
        "rate_limit",
        "quota exceeded",
        "resource exhausted",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "server error",
        "overloaded",
        "status code: 429",
        "status code: 500",
        "status code: 502",
        "status code: 503",
        "status code: 504",
        "429",
        "502",
        "503",
        "504",
    )

    return any(sub in exc_str for sub in retryable_substrings)


class BaseLLMProvider(ABC):
    """Abstract interface for an LLM provider."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if the provider has all required credentials."""
        pass

    @abstractmethod
    def get_model(
        self,
        temperature: float = 0,
        json_mode: bool = False,
        format_schema: Optional[Any] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """Constructs and returns the LangChain BaseChatModel instance."""
        pass

