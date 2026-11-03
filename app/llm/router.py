import time
import logging
from typing import Any, List, Optional

from app.llm.config import config
from app.llm.providers import PROVIDER_REGISTRY, is_retryable_error

logger = logging.getLogger("VideoMentor.LLMRouter")


class HybridLLMRouter:
    """
    Orchestrates hybrid inference with primary Ollama and cloud fallbacks:
    Ollama (Primary) -> Groq (Fallback 1) -> Gemini (Fallback 2) -> OpenRouter (Fallback 3).

    Enforces a strict total request deadline so that sequential fallbacks
    never exceed the frontend request timeout.
    """

    def __init__(
        self,
        temperature: float = 0,
        json_mode: bool = False,
        format_schema: Optional[Any] = None,
        timeout: Optional[float] = None,
        total_deadline: Optional[float] = None,
    ):
        self.temperature = temperature
        self.json_mode = json_mode
        self.format_schema = format_schema
        self.per_call_timeout = timeout if timeout is not None else config.timeout
        self.total_deadline = (
            total_deadline if total_deadline is not None else config.total_deadline
        )

    def _get_provider_chain(self) -> List[str]:
        """Returns the ordered list of providers eligible for this request."""
        return config.get_provider_chain()

    def invoke(self, prompt: Any, **kwargs) -> Any:
        """
        Executes prompt inference across the provider chain with failover protection.
        Compatible with LangChain chat model `.invoke()` signature.
        """
        chain = self._get_provider_chain()
        if not chain:
            raise RuntimeError("No LLM providers are configured.")

        start_time = time.monotonic()
        attempted_errors = []

        for idx, provider_name in enumerate(chain):
            elapsed = time.monotonic() - start_time
            remaining_time = self.total_deadline - elapsed

            # Guard against exceeding total request deadline
            if remaining_time <= 1.5:
                deadline_msg = (
                    f"Total request deadline ({self.total_deadline:.1f}s) reached before "
                    f"attempting '{provider_name}'. Attempted: {attempted_errors}"
                )
                logger.error(f"[HybridLLM] {deadline_msg}")
                raise TimeoutError("The AI is taking longer than expected. Please try again.")

            provider_cls = PROVIDER_REGISTRY.get(provider_name)
            if not provider_cls:
                logger.warning(f"[HybridLLM] Unknown provider '{provider_name}'. Skipping.")
                continue

            provider_instance = provider_cls()
            if not provider_instance.is_configured():
                logger.info(
                    f"[HybridLLM] Provider '{provider_name}' lacks API credentials. Skipping."
                )
                continue

            # Cap timeout to remaining budget
            current_timeout = min(self.per_call_timeout, remaining_time)

            logger.info(
                f"[HybridLLM] Attempting '{provider_name}' (timeout={current_timeout:.1f}s, "
                f"remaining_deadline={remaining_time:.1f}s)..."
            )

            try:
                model = provider_instance.get_model(
                    temperature=self.temperature,
                    json_mode=self.json_mode,
                    format_schema=self.format_schema,
                    timeout=current_timeout,
                )

                response = model.invoke(prompt, **kwargs)

                # Tag metadata with the successful provider
                if hasattr(response, "response_metadata") and isinstance(
                    response.response_metadata, dict
                ):
                    response.response_metadata["provider"] = provider_name

                logger.info(
                    f"[HybridLLM] Successfully generated response using '{provider_name}'."
                )
                return response

            except Exception as exc:
                exc_summary = f"{type(exc).__name__}: {str(exc)[:200]}"
                attempted_errors.append(f"{provider_name} ({exc_summary})")

                # Check if this error qualifies for fallback to next provider
                if not is_retryable_error(exc):
                    logger.error(
                        f"[HybridLLM] Non-retryable error from '{provider_name}': {exc_summary}. "
                        "Aborting fallback chain."
                    )
                    raise

                has_next = idx < len(chain) - 1
                if has_next:
                    next_provider = chain[idx + 1]
                    logger.warning(
                        f"[HybridLLM] Provider '{provider_name}' failed with {exc_summary}. "
                        f"Failing over to next provider: '{next_provider}'."
                    )
                else:
                    logger.error(
                        f"[HybridLLM] Provider '{provider_name}' failed with {exc_summary}. "
                        "No more providers in fallback chain."
                    )

        # All providers in chain failed
        error_details = "; ".join(attempted_errors)
        raise RuntimeError(
            f"All configured LLM providers failed. Details: {error_details}"
        )

