"""
Prompt templates for SigmaEval LLM interactions.

This module contains all prompt templates used for:
- Rubric generation
- User simulation
- Judge evaluation
"""

from typing import List, Dict
from .models import BehavioralTest


def _build_rubric_generation_prompt(scenario: BehavioralTest) -> str:
    """
    Build the prompt for generating a rubric from a BehavioralTest.
    
    Internal implementation detail - API may change without backward compatibility.
    
    Args:
        scenario: The behavioral test case
        
    Returns:
        A formatted prompt string for the LLM
    """
    return f"""You are creating an evaluation rubric for judging AI system behavior.

Given the following test scenario:

**Context (Given):** {scenario.given}

**Scenario (When):** {scenario.when}

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


def _build_user_simulator_prompt(
    scenario: BehavioralTest,
    conversation_history: List[Dict[str, str]],
    writing_style: str | None = None,
) -> str:
    """
    Build the prompt for simulating a user turn.
    
    Internal implementation detail - API may change without backward compatibility.
    
    Args:
        scenario: The behavioral test case
        conversation_history: List of previous conversation turns
        writing_style: Optional writing style instruction
        
    Returns:
        A formatted prompt string for the user simulator LLM
    """
    # Build conversation context
    conversation_context = ""
    if conversation_history:
        conversation_context = "\n\n**Conversation so far:**\n"
        for turn in conversation_history:
            if turn["role"] == "user":
                conversation_context += f"You (the user): {turn['content']}\n"
            else:
                conversation_context += f"Assistant: {turn['content']}\n"
    
    # Build instructions list
    instructions = [
        "- Be realistic and natural in your conversation",
    ]
    if writing_style:
        instructions.append(writing_style)

    instructions.extend(
        [
            "- If the scenario's objective has been fulfilled or completed, politely end the conversation",
            "- If you're stuck or the assistant isn't helping after multiple turns, end the conversation",
        ]
    )
    instructions_str = "\n".join(instructions)

    return f"""You are simulating a user interacting with an AI assistant.

**Background information/context (Given):** {scenario.given}

**The scenario (When):** {scenario.when}
{conversation_context}
Your task is to naturally continue the conversation as the user according to the scenario described above. 

{instructions_str}

After each message, you must decide whether to continue the conversation or end it.

Respond in the following JSON format:
{{
    "message": "Your next message to the assistant",
    "continue": true/false
}}

Set "continue" to false when:
- The scenario's objective has been achieved or completed
- You've decided to end the conversation
- The assistant has clearly failed to help after several attempts
- You've reached a natural stopping point"""


def _build_judge_prompt(
    scenario: BehavioralTest,
    conversation_text: str,
    rubric: str
) -> str:
    """
    Build the prompt for judging an interaction.
    
    Internal implementation detail - API may change without backward compatibility.
    
    Args:
        scenario: The behavioral test case
        conversation_text: Formatted conversation to evaluate
        rubric: The scoring rubric (1-10 scale)
        
    Returns:
        A formatted prompt string for the judge LLM
    """
    return f"""You are an expert evaluator judging an AI assistant's performance.

**Context (Given):** {scenario.given}

**Action/Trigger (When):** {scenario.when}

**Expected Behavior (Then):** {scenario.then.expected_behavior}

**Scoring Rubric:**
{rubric}

**Conversation to Evaluate:**
{conversation_text}

Based on the rubric above, rate this conversation on a scale of 1-10. Consider how well the assistant's behavior matched the expected behavior in the given context and scenario.

Respond in the following JSON format:
{{
    "score": <number from 1-10>,
    "reasoning": "<brief explanation of your score>"
}}"""


# System prompts for different LLM roles
RUBRIC_GENERATOR_SYSTEM_PROMPT = "You are an expert at creating detailed evaluation rubrics for AI system behavior."

USER_SIMULATOR_SYSTEM_PROMPT = "You are simulating a user interacting with an AI assistant."

JUDGE_SYSTEM_PROMPT = "You are an expert evaluator. Provide fair, consistent judgments based on the rubric."

