"""
Retry policy for transient LLM-provider failures (timeouts, 429s, 5xxs).
"""
import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
)

logger = logging.getLogger("ai_assistant.retry")
RETRYABLE_EXCEPTIONS = (Exception,)


def build_retry_decorator(max_attempts: int, backoff_seconds: float):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=backoff_seconds, max=10),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
