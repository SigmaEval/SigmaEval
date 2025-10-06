"""
Prompt templates for SigmaEval LLM interactions.

This module contains all prompt templates used for:
- Rubric generation
- User simulation
- Judge evaluation
"""

from typing import List, Dict
from .models import ScenarioTest, ConversationTurn, BehavioralExpectation


def _build_rubric_generation_prompt(
    scenario: ScenarioTest,
    expectation: BehavioralExpectation,
) -> str:
    """
    Build the prompt for generating a rubric from a ScenarioTest.
    
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

**Expected Behavior (Then):** {expectation.expected_behavior}

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
    scenario: ScenarioTest,
    conversation_history: List[Dict[str, str]],
    writing_style: Dict[str, str] | None = None,
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
    conversation_header = ""
    action_verb = "start"
    if conversation_history:
        action_verb = "continue"
        conversation_header = "The conversation so far is provided in the <conversation_history> XML block. Each <turn> tag represents a single turn of the conversation, with the speaker attribute indicating whether it was the 'user' or the 'assistant'."
        conversation_context = "\n<conversation_history>\n"
        for turn in conversation_history:
            speaker = turn["role"]
            content = turn["content"]
            conversation_context += f'<turn speaker="{speaker}">\n{content}\n</turn>\n'
        conversation_context += "</conversation_history>"
    
    # Build instructions list
    instructions = [
        "- Be realistic and natural in your conversation",
    ]
    if writing_style:
        style_str = (
            "- Adopt the following writing style for the user. But your responses "
            "should naturally vary and not strictly adhere to these at all times (e.g., "
            "an extremely verbose person might still say 'thanks' or 'ok' as a short full response).\n"
        )
        for key, value in writing_style.items():
            style_str += f"    - {key}: {value}\n"
        style_str += (
            "    (Note: If any aspect of this writing style conflicts with the 'Given' "
            "(background) or 'When' (scenario) instructions noted above, you "
            "must prioritize those instructions and disregard the conflicting "
            "aspects of this writing style.)"
        )
        instructions.append(style_str)

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

{conversation_header}
{conversation_context}

Your task is to naturally {action_verb} the conversation as the user according to the scenario described above. 


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
    scenario: ScenarioTest,
    expectation: BehavioralExpectation,
    conversation_history: list[ConversationTurn],
    rubric: str,
) -> str:
    """
    Build the prompt for judging an interaction.
    
    Internal implementation detail - API may change without backward compatibility.
    
    Args:
        scenario: The behavioral test case
        conversation_history: List of conversation turns to evaluate
        rubric: The scoring rubric (1-10 scale)
        
    Returns:
        A formatted prompt string for the judge LLM
    """
    conversation_text = ""
    conversation_header = ""
    if conversation_history:
        conversation_header = "The conversation to evaluate is provided in the <conversation_history> XML block. Each <turn> tag represents a single turn of the conversation, with the speaker attribute indicating whether it was the 'user' or the 'assistant'."
        conversation_text = "\n<conversation_history>\n"
        for turn in conversation_history:
            speaker = turn.role
            content = turn.content
            conversation_text += f'<turn speaker="{speaker}">\n{content}\n</turn>\n'
        conversation_text += "</conversation_history>"

    return f"""You are an expert evaluator judging an AI assistant's performance.

**Context (Given):** {scenario.given}

**Action/Trigger (When):** {scenario.when}

**Expected Behavior (Then):** {expectation.expected_behavior}

**Scoring Rubric:**
{rubric}

**Conversation to Evaluate:**
{conversation_header}
{conversation_text}

Based on the rubric above, rate this conversation on a scale of 1-10. Consider how well the assistant's behavior matched the expected behavior in the given context and scenario.

Respond in the following JSON format:
{{
    "score": <number from 1-10>,
    "reasoning": "<brief explanation of your score>"
}}"""


# System prompts for different LLM roles
RUBRIC_GENERATOR_SYSTEM_PROMPT = "You are an expert at creating detailed evaluation rubrics for AI system behavior."

USER_SIMULATOR_SYSTEM_PROMPT = "You are simulating a user interacting with an AI assistant. You will be given a scenario and a conversation history in XML format. Follow the instructions to generate the user's next message."

JUDGE_SYSTEM_PROMPT = "You are an expert evaluator. Provide fair, consistent judgments based on the rubric."

