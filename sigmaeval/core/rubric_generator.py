"""
Rubric generation logic for Phase 1 of SigmaEval evaluation.
"""

from typing import Dict, Any
from litellm import acompletion

from .models import BehavioralTest
from .prompts import _build_rubric_generation_prompt, RUBRIC_GENERATOR_SYSTEM_PROMPT
from .exceptions import LLMCommunicationError


async def _generate_rubric(
    scenario: BehavioralTest,
    model: str
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
    
    try:
        response = await acompletion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": RUBRIC_GENERATOR_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
        )
    except Exception as e:
        raise LLMCommunicationError("Rubric generation LLM call failed") from e
    
    rubric = response.choices[0].message.content
    if not isinstance(rubric, str) or not rubric.strip():
        raise LLMCommunicationError("Rubric generation returned empty content")
    return rubric


def _parse_behavioral_test(scenario: BehavioralTest) -> Dict[str, Any]:
    """
    Parse a BehavioralTest into its constituent components.
    
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
        - sample_size: Number of samples to collect (defaults to 20 if not specified)
    """
    return {
        "title": scenario.title,
        "given": scenario.given,
        "when": scenario.when,
        "expected_behavior": scenario.then.expected_behavior,
        "evaluator": scenario.then.evaluator,
        "sample_size": getattr(scenario.then.evaluator, "sample_size", 20),
    }

