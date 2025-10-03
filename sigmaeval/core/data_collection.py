"""
Data collection logic for Phase 2 of SigmaEval evaluation.

This module handles:
- User simulation with User Simulator LLM
- Recording interactions with the application under test
- Judging outcomes with Judge LLM using the rubric
"""

import asyncio
from typing import Callable, Awaitable, Any, Dict, List
from litellm import acompletion

from .models import AppResponse, BehavioralTest
from .prompts import (
    _build_user_simulator_prompt,
    _build_judge_prompt,
    JUDGE_SYSTEM_PROMPT,
)


class ConversationRecord:
    """
    Record of a single conversation between user simulator and app.
    
    This class stores the turn-by-turn interaction between the simulated user
    and the application under test.
    
    Attributes:
        turns: List of conversation turns, each a dict with 'role' and 'content'
    """
    
    def __init__(self):
        self.turns: List[Dict[str, str]] = []
    
    def add_user_message(self, message: str):
        """Add a user message to the conversation."""
        self.turns.append({"role": "user", "content": message})
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the conversation."""
        self.turns.append({"role": "assistant", "content": message})
    
    def to_formatted_string(self) -> str:
        """
        Format the conversation as a human-readable string.
        
        Returns:
            A string with each turn formatted as "User: ..." or "Assistant: ..."
        """
        lines = []
        for turn in self.turns:
            if turn["role"] == "user":
                lines.append(f"User: {turn['content']}")
            else:
                lines.append(f"Assistant: {turn['content']}")
        return "\n\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation to a dictionary format.
        
        Returns:
            Dictionary with 'turns' key containing the conversation turns
        """
        return {"turns": self.turns}
    
    def __repr__(self) -> str:
        """String representation of the conversation record."""
        return f"ConversationRecord(turns={len(self.turns)})"


async def _simulate_user_turn(
    scenario: BehavioralTest,
    conversation_history: List[Dict[str, str]],
    model: str,
    max_turns: int = 10
) -> tuple[str, bool]:
    """
    Simulate a single user turn using the User Simulator LLM.
    
    Args:
        scenario: The behavioral test case
        conversation_history: List of previous conversation turns
        model: The LLM model identifier
        max_turns: Maximum number of turns before ending conversation
        
    Returns:
        Tuple of (user_message, should_continue)
        - user_message: The simulated user's message
        - should_continue: Whether the conversation should continue
    """
    prompt = _build_user_simulator_prompt(scenario, conversation_history)
    
    messages = [{"role": "system", "content": prompt}]
    
    # Check if we've exceeded max turns
    turn_count = len([m for m in conversation_history if m["role"] == "user"])
    if turn_count >= max_turns:
        return "[Conversation ended - max turns reached]", False
    
    response = await acompletion(
        model=model,
        messages=messages,
        temperature=0.8,
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    
    # Parse JSON response
    import json
    try:
        parsed = json.loads(content)
        user_message = parsed.get("message", "")
        should_continue = parsed.get("continue", False)
        return user_message, should_continue
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return content, False


async def _run_single_interaction(
    scenario: BehavioralTest,
    app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
    model: str,
    max_turns: int = 10
) -> ConversationRecord:
    """
    Run a single interaction between user simulator and the app.
    
    This is Phase 2, Steps 3-4: Simulate user and record interaction.
    
    Args:
        scenario: The behavioral test case
        app_handler: Async callback to interact with the app under test
        model: The LLM model identifier for user simulation
        max_turns: Maximum conversation turns
        
    Returns:
        ConversationRecord containing the full interaction
    """
    conversation = ConversationRecord()
    # History for User Simulator LLM - only the actual conversation content
    simulator_conversation_history: List[Dict[str, str]] = []
    app_state: Dict[str, Any] = {}
    
    should_continue = True
    
    while should_continue:
        # Simulate user message based on current conversation history
        user_message, should_continue = await _simulate_user_turn(
            scenario,
            simulator_conversation_history,
            model,
            max_turns
        )
        
        # Check if conversation should end
        if not user_message or user_message.startswith("[Conversation ended"):
            break
        
        # Record user message
        conversation.add_user_message(user_message)
        
        # Get app response for this user message
        app_response = await app_handler(user_message, app_state)
        
        # Record app response
        conversation.add_assistant_message(app_response.response)
        
        # Update histories for next iteration
        # The simulator needs to see: what it said (user), what app replied (assistant)
        simulator_conversation_history.append({"role": "user", "content": user_message})
        simulator_conversation_history.append({"role": "assistant", "content": app_response.response})
        
        # Update app state for next turn
        app_state = app_response.state
    
    return conversation


async def _judge_interaction(
    scenario: BehavioralTest,
    conversation: ConversationRecord,
    rubric: str,
    model: str
) -> tuple[float, str]:
    """
    Judge a single interaction using the Judge LLM.
    
    This is Phase 2, Step 5: Judge expected behavior with Judge LLM.
    
    Args:
        scenario: The behavioral test case
        conversation: The recorded conversation to judge
        rubric: The scoring rubric (1-10 scale)
        model: The LLM model identifier for judging
        
    Returns:
        Tuple of (score, reasoning)
        - score: Score from 1-10 based on the rubric
        - reasoning: Judge's explanation for the score
    """
    conversation_text = conversation.to_formatted_string()
    prompt = _build_judge_prompt(scenario, conversation_text, rubric)
    
    response = await acompletion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    content = response.choices[0].message.content
    
    # Parse JSON response
    import json
    try:
        parsed = json.loads(content)
        score = float(parsed.get("score", 1))
        reasoning = parsed.get("reasoning", "No reasoning provided")
        # Clamp score to valid range
        score = max(1.0, min(10.0, score))
        return score, reasoning
    except (json.JSONDecodeError, ValueError):
        # Fallback to minimum score if parsing fails
        return 1.0, "Error parsing judge response"


async def _run_single_evaluation(
    scenario: BehavioralTest,
    app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
    rubric: str,
    model: str,
    max_turns: int = 10
) -> tuple[float, str, ConversationRecord]:
    """
    Run a complete single evaluation: simulate, interact, and judge.
    
    Args:
        scenario: The behavioral test case
        app_handler: Async callback to interact with the app under test
        rubric: The scoring rubric
        model: The LLM model identifier
        max_turns: Maximum conversation turns
        
    Returns:
        Tuple of (score, reasoning, conversation_record)
        - score: Judge's score (1-10)
        - reasoning: Judge's explanation for the score
        - conversation_record: Full conversation transcript
    """
    # Steps 3-4: Simulate and record
    conversation = await _run_single_interaction(
        scenario,
        app_handler,
        model,
        max_turns
    )
    
    # Step 5: Judge
    score, reasoning = await _judge_interaction(
        scenario,
        conversation,
        rubric,
        model
    )
    
    return score, reasoning, conversation


async def collect_evaluation_data(
    scenario: BehavioralTest,
    app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
    rubric: str,
    model: str,
    sample_size: int,
    concurrency: int = 10,
    max_turns: int = 10
) -> tuple[List[float], List[str], List[ConversationRecord]]:
    """
    Collect evaluation data by running multiple interactions with controlled concurrency.
    
    This orchestrates Phase 2 of the evaluation process, running steps 3-5
    multiple times (sample_size) using a semaphore to maintain constant concurrency.
    
    Args:
        scenario: The behavioral test case
        app_handler: Async callback to interact with the app under test
        rubric: The scoring rubric from Phase 1
        model: The LLM model identifier
        sample_size: Total number of evaluations to run
        concurrency: Maximum number of evaluations to run concurrently
        max_turns: Maximum conversation turns per interaction
        
    Returns:
        Tuple of (scores, reasoning_list, conversations)
        - scores: List of scores (1-10) from Judge LLM
        - reasoning_list: List of judge's explanations for each score
        - conversations: List of ConversationRecord objects
    """
    semaphore = asyncio.Semaphore(concurrency)
    
    async def _run_with_semaphore() -> tuple[float, str, ConversationRecord]:
        """Run a single evaluation with semaphore control."""
        async with semaphore:
            return await _run_single_evaluation(
                scenario,
                app_handler,
                rubric,
                model,
                max_turns
            )
    
    # Create all tasks at once, semaphore controls concurrency
    tasks = [_run_with_semaphore() for _ in range(sample_size)]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Separate scores, reasoning, and conversations
    scores = [score for score, _, _ in results]
    reasoning_list = [reasoning for _, reasoning, _ in results]
    conversations = [conversation for _, _, conversation in results]
    
    return scores, reasoning_list, conversations

