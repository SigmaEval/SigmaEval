"""
Data collection logic for Phase 2 of SigmaEval evaluation.

This module handles:
- User simulation with User Simulator LLM
- Recording interactions with the application under test
- Judging outcomes with Judge LLM using the rubric
"""

import asyncio
import logging
import secrets
from typing import Callable, Awaitable, Any, Dict, List
import json
from litellm import acompletion
from tqdm.asyncio import tqdm

from .models import AppResponse, BehavioralTest, ConversationRecord
from .prompts import (
    _build_user_simulator_prompt,
    _build_judge_prompt,
    JUDGE_SYSTEM_PROMPT,
)
from .exceptions import LLMCommunicationError

logger = logging.getLogger("sigmaeval")


async def _simulate_user_turn(
    scenario: BehavioralTest,
    conversation_history: List[Dict[str, str]],
    model: str,
    max_turns: int = 10,
    eval_id: str = ""
) -> tuple[str, bool]:
    """
    Simulate a single user turn using the User Simulator LLM.
    
    Args:
        scenario: The behavioral test case
        conversation_history: List of previous conversation turns
        model: The LLM model identifier
        max_turns: Maximum number of turns before ending conversation
        eval_id: Unique identifier for the evaluation run
        
    Returns:
        Tuple of (user_message, should_continue)
        - user_message: The simulated user's message
        - should_continue: Whether the conversation should continue
    """
    prompt = _build_user_simulator_prompt(scenario, conversation_history)
    log_prefix = f"[{eval_id}] " if eval_id else ""
    logger.debug(f"{log_prefix}User simulator prompt: {prompt}")
    
    messages = [{"role": "system", "content": prompt}]
    
    # Check if we've exceeded max turns
    turn_count = len([m for m in conversation_history if m["role"] == "user"])
    if turn_count >= max_turns:
        logger.debug(f"{log_prefix}Max turns reached, ending conversation.")
        return "[Conversation ended - max turns reached]", False
    
    try:
        response = await acompletion(
            model=model,
            messages=messages,
            temperature=0.8,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        raise LLMCommunicationError("User simulator LLM call failed") from e
    
    content = response.choices[0].message.content
    logger.debug(f"{log_prefix}User simulator response: {content}")
    
    # Parse JSON response
    try:
        parsed = json.loads(content)
        user_message = parsed.get("message", "")
        should_continue = parsed.get("continue", False)
        return user_message, should_continue
    except json.JSONDecodeError as e:
        raise LLMCommunicationError(
            f"User simulator returned non-JSON response: {content[:200]}"
        ) from e


async def _run_single_interaction(
    scenario: BehavioralTest,
    app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
    user_simulator_model: str,
    max_turns: int = 10,
    eval_id: str = ""
) -> ConversationRecord:
    """
    Run a single interaction between user simulator and the app.
    
    This is Phase 2, Steps 3-4: Simulate user and record interaction.
    
    Args:
        scenario: The behavioral test case
        app_handler: Async callback to interact with the app under test
        user_simulator_model: The LLM model identifier for user simulation
        max_turns: Maximum conversation turns
        eval_id: Unique identifier for the evaluation run
        
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
            user_simulator_model,
            max_turns,
            eval_id
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
    judge_model: str,
    eval_id: str = ""
) -> tuple[float, str]:
    """
    Judge a single interaction using the Judge LLM.
    
    This is Phase 2, Step 5: Judge expected behavior with Judge LLM.
    
    Args:
        scenario: The behavioral test case
        conversation: The recorded conversation to judge
        rubric: The scoring rubric (1-10 scale)
        judge_model: The LLM model identifier for judging
        eval_id: Unique identifier for the evaluation run
        
    Returns:
        Tuple of (score, reasoning)
        - score: Score from 1-10 based on the rubric
        - reasoning: Judge's explanation for the score
    """
    conversation_text = conversation.to_formatted_string()
    prompt = _build_judge_prompt(scenario, conversation_text, rubric)
    log_prefix = f"[{eval_id}] " if eval_id else ""
    logger.debug(f"{log_prefix}Judge prompt: {prompt}")
    
    try:
        response = await acompletion(
            model=judge_model,
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
    except Exception as e:
        raise LLMCommunicationError("Judge LLM call failed") from e
    
    content = response.choices[0].message.content
    logger.debug(f"{log_prefix}Judge response: {content}")
    
    # Parse JSON response
    try:
        parsed = json.loads(content)
        if "score" not in parsed:
            raise LLMCommunicationError("Judge LLM response missing 'score' field")
        score = float(parsed["score"])
        reasoning = parsed.get("reasoning", "No reasoning provided")
        # Clamp score to valid range
        score = max(1.0, min(10.0, score))
        return score, reasoning
    except json.JSONDecodeError as e:
        raise LLMCommunicationError(
            f"Judge LLM returned non-JSON response: {content[:200]}"
        ) from e
    except ValueError as e:
        raise LLMCommunicationError("Judge LLM response contained non-numeric 'score'") from e


async def _run_single_evaluation(
    scenario: BehavioralTest,
    app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
    rubric: str,
    judge_model: str,
    user_simulator_model: str,
    max_turns: int = 10,
    eval_id: str = ""
) -> tuple[float, str, ConversationRecord]:
    """
    Run a complete single evaluation: simulate, interact, and judge.
    
    Args:
        scenario: The behavioral test case
        app_handler: Async callback to interact with the app under test
        rubric: The scoring rubric
        judge_model: The LLM model identifier for the judge
        user_simulator_model: The LLM model identifier for the user simulator
        max_turns: Maximum conversation turns
        eval_id: Unique identifier for the evaluation run
        
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
        user_simulator_model,
        max_turns,
        eval_id
    )
    
    # Step 5: Judge
    score, reasoning = await _judge_interaction(
        scenario,
        conversation,
        rubric,
        judge_model,
        eval_id
    )
    
    return score, reasoning, conversation


async def collect_evaluation_data(
    scenario: BehavioralTest,
    app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
    rubric: str,
    judge_model: str,
    user_simulator_model: str,
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
        judge_model: The LLM model for the judge
        user_simulator_model: The LLM model for the user simulator
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
        eval_id = secrets.token_hex(9)  # 18-character random hex string
        async with semaphore:
            return await _run_single_evaluation(
                scenario,
                app_handler,
                rubric,
                judge_model,
                user_simulator_model,
                max_turns,
                eval_id=eval_id
            )
    
    # Create all tasks at once, semaphore controls concurrency
    tasks = [_run_with_semaphore() for _ in range(sample_size)]
    
    # Wait for all tasks to complete with progress bar
    results = await tqdm.gather(*tasks, desc="Running evaluations")
    
    # Separate scores, reasoning, and conversations
    scores = [score for score, _, _ in results]
    reasoning_list = [reasoning for _, reasoning, _ in results]
    conversations = [conversation for _, _, conversation in results]
    
    return scores, reasoning_list, conversations

