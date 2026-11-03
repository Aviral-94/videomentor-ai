from typing import Any, Optional
from app.llm.router import HybridLLMRouter


def get_llm(
    temperature: float = 0,
    json_mode: bool = False,
    format_schema: Optional[Any] = None,
    timeout: Optional[float] = None,
    total_deadline: Optional[float] = None,
) -> HybridLLMRouter:
    """
    Centralized factory for obtaining an LLM instance with hybrid multi-provider
    fallback capability.

    Parameters:
    - temperature: Sampling temperature (default 0).
    - json_mode: Whether to request JSON output from the model.
    - format_schema: Specific schema (e.g. QUIZ_SCHEMA) for structured generation.
    - timeout: Per-provider timeout in seconds.
    - total_deadline: Maximum total budget across all providers (default 40s).

    Returns:
    A HybridLLMRouter instance exposing a standard LangChain `.invoke(prompt)` interface.
    """
    return HybridLLMRouter(
        temperature=temperature,
        json_mode=json_mode,
        format_schema=format_schema,
        timeout=timeout,
        total_deadline=total_deadline,
    )

