"""
Fallback orchestrator: tries providers in priority order, retrying each
with backoff before moving to the next. Degrades gracefully if all fail.
"""
import logging
from typing import Any, Dict, List

from app.llm.providers import LLMProvider
from app.reliability.retry import build_retry_decorator

logger = logging.getLogger("ai_assistant.fallback")


class AllProvidersFailedError(Exception):
    def __init__(self, errors: Dict[str, str]):
        self.errors = errors
        super().__init__(f"All providers failed: {errors}")


class FallbackOrchestrator:
    def __init__(
        self,
        providers: Dict[str, LLMProvider],
        order: List[str],
        retry_max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ):
        self.providers = providers
        self.order = [p for p in order if p in providers]
        self._retry_decorator = build_retry_decorator(retry_max_attempts, retry_backoff_seconds)

    async def chat(self, **kwargs) -> Dict[str, Any]:
        errors: Dict[str, str] = {}

        for provider_name in self.order:
            provider = self.providers[provider_name]
            try:
                call = self._retry_decorator(provider.chat)
                result = await call(**kwargs)
                result["provider_used"] = provider_name
                return result
            except Exception as e:  # noqa: BLE001
                logger.error("Provider '%s' failed after retries: %s", provider_name, e)
                errors[provider_name] = str(e)
                continue

        raise AllProvidersFailedError(errors)

    @staticmethod
    def graceful_degradation_response(errors: Dict[str, str]) -> Dict[str, Any]:
        return {
            "content": (
                "I'm temporarily unable to reach any language model provider "
                "(cloud or local). Please try again shortly. "
                f"[debug: {errors}]"
            ),
            "tool_calls": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "provider_used": "none",
            "degraded": True,
        }
