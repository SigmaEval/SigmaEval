"""
LiteLLM client utilities with Tenacity-based retries.

This module centralizes outbound LLM calls to apply consistent, resilient
retry behavior across the framework, with configurable settings.
"""

import logging
from typing import Any, Dict, List, Optional

from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
from litellm import acompletion as _litellm_acompletion
from .models import RetryConfig


logger = logging.getLogger("sigmaeval")




async def _acompletion_with_retry(
    *,
    model: str,
    messages: List[Dict[str, str]],
    retry_config: Optional[RetryConfig] = None,
    **kwargs: Any,
):
    """
    Call LiteLLM's async completion with resilient, configurable retries.

    Args:
        model: LiteLLM model identifier (e.g., "openai/gpt-4o")
        messages: Chat messages following the OpenAI-style schema
        retry_config: Optional RetryConfig; if disabled or attempts<=1, no retries
        **kwargs: Additional parameters forwarded to LiteLLM

    Raises:
        Propagates the last exception after exhausting retries.

    Returns:
        The LiteLLM completion response object.
    """
    cfg = retry_config or RetryConfig()

    if not cfg.enabled or cfg.max_attempts <= 1:
        return await _litellm_acompletion(model=model, messages=messages, **kwargs)

    retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(cfg.max_attempts),
        wait=wait_random_exponential(multiplier=cfg.backoff_multiplier, max=cfg.max_backoff_seconds),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    async for attempt in retrying:
        with attempt:
            return await _litellm_acompletion(model=model, messages=messages, **kwargs)


