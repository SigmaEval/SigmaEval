"""
Rubric generation logic for Phase 1 of SigmaEval evaluation.
"""

from typing import Dict, Any
from litellm import acompletion

from .models import BehavioralTest


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
    
    response = await acompletion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating detailed evaluation rubrics for AI system behavior."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7,
    )
    
    rubric = response.choices[0].message.content
    return rubric


def _build_rubric_generation_prompt(scenario: BehavioralTest) -> str:
    """
    Build the prompt for generating a rubric from a BehavioralTest.
    
    Args:
        scenario: The behavioral test case
        
    Returns:
        A formatted prompt string for the LLM
    """
    prompt = f"""You are creating an evaluation rubric for judging AI system behavior.

Given the following test scenario:

**Context (Given):** {scenario.given}

**User Goal (When):** {scenario.when}

**Expected Behavior (Then):** {scenario.then.expected_behavior}

Create a detailed 1-10 scoring rubric that will be used to evaluate whether the AI system's behavior meets the expected outcome. The rubric should:

1. Provide clear criteria for each rating level from 1 to 10
2. Rating of 1-5 should represent varying degrees of failure to meet expectations
3. Rating of 6-10 should represent varying degrees of success in meeting expectations
4. Be specific to the expected behavior described
5. Consider both what the system does AND how well it does it (clarity, completeness, helpfulness)
6. Use gradual progression - each level should be meaningfully different from adjacent levels

Format your rubric as follows:
**1:** [Description of worst possible response]
**2:** [Description]
**3:** [Description]
**4:** [Description]
**5:** [Description]
**6:** [Description - minimum acceptable]
**7:** [Description]
**8:** [Description]
**9:** [Description]
**10:** [Description of ideal response]

Be concise but specific. Each rating description should be 1-2 sentences maximum."""
    
    return prompt


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

