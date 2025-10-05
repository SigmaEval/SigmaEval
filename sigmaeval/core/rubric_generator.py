"""
Rubric generation logic for Phase 1 of SigmaEval evaluation.
"""

import logging
from typing import Dict, Any
from litellm import acompletion as _litellm_acompletion
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

from .models import ScenarioTest, RetryConfig
from .prompts import _build_rubric_generation_prompt, RUBRIC_GENERATOR_SYSTEM_PROMPT
from .exceptions import LLMCommunicationError

logger = logging.getLogger("sigmaeval")


async def _generate_rubric(
    scenario: ScenarioTest,
    model: str,
    retry_config: RetryConfig | None = None,
) -> str:
    """
    Generate a 1-10 scoring rubric based on the expected behavior.
    
    Internal implementation detail - API may change without backward compatibility.
    
    This is Phase 1, Step 2 of the evaluation process. The rubric provides
    detailed criteria for the Judge LLM to evaluate interactions consistently.
    
    Args:
        scenario: The behavioral test case containing the expected behavior
        model: The LLM model identifier to use (e.g., "openai/gpt-4o")
        
    Returns:
        A string containing the generated 1-10 rubric
        
    Example:
        A rubric might look like:
        
        1: Bot gives no answer or ignores the question.
        2: Bot answers irrelevantly, with no mention of its functions.
        3: Bot gives vague or incomplete information, missing most functions.
        ...
        10: Bot names all required functions clearly, concisely, in order, 
            and with natural, helpful phrasing.
    """
    prompt = _build_rubric_generation_prompt(scenario)
    logger.debug(f"Rubric generation prompt: {prompt}")
    
    cfg = retry_config or RetryConfig()
    retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(cfg.max_attempts if cfg.enabled else 1),
        wait=wait_random_exponential(
            multiplier=cfg.backoff_multiplier, max=cfg.max_backoff_seconds
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    async for attempt in retrying:
        with attempt:
            try:
                response = await _litellm_acompletion(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": RUBRIC_GENERATOR_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.7,
                    drop_params=True,
                )
            except Exception as e:
                raise LLMCommunicationError("Rubric generation LLM call failed") from e

            rubric = response.choices[0].message.content
            if not isinstance(rubric, str) or not rubric.strip():
                raise LLMCommunicationError("Rubric generation returned empty content")

            logger.debug(f"Generated rubric: {rubric}")
            return rubric
    
    # This path should not be reachable if reraise=True is set
    raise LLMCommunicationError("Exhausted all retries for rubric generation.")


def _parse_behavioral_test(scenario: ScenarioTest) -> Dict[str, Any]:
    """
    Parse a ScenarioTest into its constituent components.
    
    This is Phase 1, Step 1 of the evaluation process.
    
    Internal implementation detail - API may change without backward compatibility.
    
    Args:
        scenario: The behavioral test case to parse
        
    Returns:
        Dictionary containing parsed components:
        - title: Test case title
        - given: Context and prerequisites
        - when: User goal or action
        - expected_behavior: Expected outcome description
        - evaluator: Statistical evaluator instance
        - sample_size: Number of samples to collect
    """
    return {
        "title": scenario.title,
        "given": scenario.given,
        "when": scenario.when,
        "expected_behavior": scenario.then.expected_behavior,
        "evaluator": scenario.then.evaluator,
        "sample_size": scenario.then.evaluator.sample_size,
    }

