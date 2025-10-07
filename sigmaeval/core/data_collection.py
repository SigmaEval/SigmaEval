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
from typing import Callable, Awaitable, Any, Dict, List, Tuple, Union
from datetime import datetime, timezone
from litellm import acompletion as _litellm_acompletion
from tqdm.asyncio import tqdm
from tenacity import (
    AsyncRetrying,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

from .models import AppResponse, ScenarioTest, ConversationRecord, RetryConfig
from .prompts import (
    _build_user_simulator_prompt,
    _build_judge_prompt,
    JUDGE_SYSTEM_PROMPT,
    USER_SIMULATOR_SYSTEM_PROMPT,
)
from .exceptions import LLMCommunicationError
from .writing_styles import _generate_writing_style
from .models import WritingStyleConfig
from .utils import _extract_json_from_response
from .models import Expectation

logger = logging.getLogger("sigmaeval")


# Define the flexible app_handler signature and its possible return types
AppHandler = Callable[
    [List[Dict[str, str]], Any],
    Awaitable[Union[AppResponse, str, Tuple[str, Any]]],
]


async def _simulate_user_turn(
    scenario: ScenarioTest,
    conversation_history: List[Dict[str, str]],
    model: str,
    max_turns: int = 10,
    eval_id: str = "",
    retry_config: RetryConfig | None = None,
    writing_style: dict[str, str] | None = None,
) -> tuple[str, bool, datetime | None, datetime | None]:
    """
    Simulate a single user turn using the User Simulator LLM. This function
    includes retries for LLM communication and response parsing errors.

    Args:
        scenario: The behavioral test case
        conversation_history: List of previous conversation turns
        model: The LLM model identifier
        max_turns: Maximum number of turns before ending conversation
        eval_id: Unique identifier for the evaluation run
        retry_config: Configuration for retrying LLM calls on failure.
        writing_style: Optional writing style instruction dictionary

    Returns:
        Tuple of (user_message, should_continue, request_timestamp, response_timestamp)
        - user_message: The simulated user's message
        - should_continue: Whether the conversation should continue
        - request_timestamp: Timestamp when the LLM request was sent
        - response_timestamp: Timestamp when the LLM response was received
    """
    prompt = _build_user_simulator_prompt(
        scenario, conversation_history, writing_style=writing_style
    )
    log_prefix = f"[{eval_id}] " if eval_id else ""
    logger.debug(f"{log_prefix}User simulator prompt: {prompt}")

    messages = [
        {"role": "system", "content": USER_SIMULATOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    # Check if we've exceeded max turns
    turn_count = len([m for m in conversation_history if m["role"] == "user"])
    if turn_count >= max_turns:
        logger.debug(f"{log_prefix}Max turns reached, ending conversation.")
        return "[Conversation ended - max turns reached]", False, None, None

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
                request_timestamp = datetime.now(timezone.utc)
                response = await _litellm_acompletion(
                    model=model,
                    messages=messages,
                    temperature=0.8,
                    response_format={"type": "json_object"},
                    drop_params=True,
                )
                response_timestamp = datetime.now(timezone.utc)
            except Exception as e:
                raise LLMCommunicationError("User simulator LLM call failed") from e

            content = response.choices[0].message.content
            logger.debug(f"{log_prefix}User simulator response: {content}")

            # Parse JSON response
            try:
                parsed = _extract_json_from_response(content)
                if parsed:
                    user_message = parsed.get("message", "")
                    should_continue = parsed.get("continue", False)
                    return user_message, should_continue, request_timestamp, response_timestamp

                # If parsing fails, raise an error to trigger a retry
                raise LLMCommunicationError(
                    f"User simulator returned non-JSON response: {content[:200]}"
                )
            except Exception as e:
                if isinstance(e, LLMCommunicationError):
                    raise
                raise LLMCommunicationError(
                    f"Failed to parse user simulator response: {content[:200]}"
                ) from e

    # This path should not be reachable if reraise=True is set
    raise LLMCommunicationError("Exhausted all retries for user simulator.")


async def _run_single_interaction(
    scenario: ScenarioTest,
    app_handler: AppHandler,
    user_simulator_model: str,
    max_turns: int = 10,
    eval_id: str = "",
    retry_config: RetryConfig | None = None,
    writing_style: dict[str, str] | None = None,
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
        writing_style: Optional writing style instruction dictionary

    Returns:
        ConversationRecord containing the full interaction
    """
    conversation = ConversationRecord(writing_style=writing_style)
    # History for User Simulator LLM - only the actual conversation content
    simulator_conversation_history: List[Dict[str, str]] = []
    app_state: Any = {}

    should_continue = True

    while should_continue:
        # Simulate user message based on current conversation history
        user_message, should_continue, sim_req_ts, sim_resp_ts = await _simulate_user_turn(
            scenario,
            simulator_conversation_history,
            user_simulator_model,
            max_turns,
            eval_id,
            retry_config,
            writing_style,
        )

        # Check if conversation should end
        if not user_message or user_message.startswith("[Conversation ended"):
            break

        # Add the new user message to the history for the app
        app_conversation_history = simulator_conversation_history + [
            {"role": "user", "content": user_message}
        ]

        # Get app response for this user message
        app_req_ts = datetime.now(timezone.utc)
        app_output = await app_handler(app_conversation_history, app_state)
        app_resp_ts = datetime.now(timezone.utc)

        # --- Normalize the app's response ---
        app_response: AppResponse
        if isinstance(app_output, AppResponse):
            app_response = app_output
        elif isinstance(app_output, str):
            app_response = AppResponse(response=app_output, state={})
        elif isinstance(app_output, tuple) and len(app_output) == 2:
            app_response = AppResponse(response=app_output[0], state=app_output[1])
        else:
            raise TypeError(
                f"app_handler returned an unsupported type: {type(app_output)}. "
                "Supported types are: str, Tuple[str, Any], or AppResponse."
            )
        # --- End Normalization ---

        # Record user message
        if sim_req_ts and sim_resp_ts:
            conversation.add_user_message(
                user_message,
                request_timestamp=sim_req_ts,
                response_timestamp=sim_resp_ts,
            )

        # Record app response
        conversation.add_assistant_message(
            app_response.response, request_timestamp=app_req_ts, response_timestamp=app_resp_ts
        )

        # Update histories for next iteration
        # The simulator needs to see: what it said (user), what app replied (assistant)
        simulator_conversation_history.append({"role": "user", "content": user_message})
        simulator_conversation_history.append(
            {"role": "assistant", "content": app_response.response}
        )

        # Update app state for next turn
        app_state = app_response.state

    return conversation


async def _judge_interaction(
    scenario: ScenarioTest,
    expectation: Expectation,
    conversation: ConversationRecord,
    rubric: str,
    judge_model: str,
    eval_id: str = "",
    retry_config: RetryConfig | None = None,
) -> tuple[float, str]:
    """
    Judge a single interaction using the Judge LLM. This function
    includes retries for LLM communication and response parsing errors.

    This is Phase 2, Step 5: Judge expected behavior with Judge LLM.

    Args:
        scenario: The behavioral test case
        conversation: The recorded conversation to judge
        rubric: The scoring rubric (1-10 scale)
        judge_model: The LLM model identifier for judging
        eval_id: Unique identifier for the evaluation run
        retry_config: Configuration for retrying LLM calls on failure.

    Returns:
        Tuple of (score, reasoning)
        - score: Score from 1-10 based on the rubric
        - reasoning: Judge's explanation for the score
    """
    prompt = _build_judge_prompt(scenario, expectation, conversation.turns, rubric)
    log_prefix = f"[{eval_id}] " if eval_id else ""
    logger.debug(f"{log_prefix}Judge prompt: {prompt}")

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
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    drop_params=True,
                )
            except Exception as e:
                raise LLMCommunicationError("Judge LLM call failed") from e

            content = response.choices[0].message.content
            logger.debug(f"{log_prefix}Judge response: {content}")

            # Parse JSON response
            try:
                parsed = _extract_json_from_response(content)
                if not parsed or "score" not in parsed:
                    raise LLMCommunicationError(
                        "Judge LLM response is not valid JSON or is missing the 'score' field."
                    )

                score = float(parsed["score"])
                reasoning = parsed.get("reasoning", "No reasoning provided")
                # Clamp score to valid range
                score = max(1.0, min(10.0, score))
                return score, reasoning
            except (ValueError, TypeError) as e:
                raise LLMCommunicationError(
                    f"Judge LLM response contained non-numeric 'score': {content[:200]}"
                ) from e
            except Exception as e:
                if isinstance(e, LLMCommunicationError):
                    raise
                raise LLMCommunicationError(
                    f"Failed to parse judge response: {content[:200]}"
                ) from e

    # This path should not be reachable if reraise=True is set
    raise LLMCommunicationError("Exhausted all retries for judge.")


async def _collect_conversations(
    scenario: ScenarioTest,
    app_handler: AppHandler,
    user_simulator_model: str,
    sample_size: int,
    concurrency: int = 10,
    max_turns: int = 10,
    retry_config: RetryConfig | None = None,
    writing_style_config: WritingStyleConfig | None = None,
) -> List[ConversationRecord]:
    """
    Simulate and collect conversation data with controlled concurrency.

    This function runs the user simulation `sample_size` times to generate
    a dataset of conversations based on the scenario's `given` and `when`
    clauses. It does not perform any judging.

    Args:
        scenario: The behavioral test case.
        app_handler: Async callback to interact with the app under test.
        user_simulator_model: The LLM model for the user simulator.
        sample_size: Total number of conversations to simulate.
        concurrency: Maximum number of simulations to run concurrently.
        max_turns: Maximum conversation turns per interaction.
        retry_config: Configuration for retrying LLM calls.
        writing_style_config: Configuration for writing style variations.

    Returns:
        A list of `ConversationRecord` objects.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_with_semaphore() -> ConversationRecord:
        """Run a single interaction with semaphore control."""
        eval_id = secrets.token_hex(9)  # 18-character random hex string

        # Generate a random writing style for this specific interaction
        config = writing_style_config or WritingStyleConfig()
        writing_style = _generate_writing_style(axes=config.axes) if config.enabled else None

        async with semaphore:
            return await _run_single_interaction(
                scenario,
                app_handler,
                user_simulator_model,
                max_turns,
                eval_id=eval_id,
                retry_config=retry_config,
                writing_style=writing_style,
            )

    # Create all tasks at once, semaphore controls concurrency
    tasks = [_run_with_semaphore() for _ in range(sample_size)]

    # Wait for all tasks to complete with progress bar
    results = await tqdm.gather(*tasks, desc="Simulating user conversations")

    return results


async def _judge_conversations(
    scenario: ScenarioTest,
    expectation: Expectation,
    conversations: List[ConversationRecord],
    rubric: str,
    judge_model: str,
    concurrency: int = 10,
    retry_config: RetryConfig | None = None,
) -> tuple[List[float], List[str]]:
    """
    Judge a list of conversations against a rubric with controlled concurrency.

    This function takes a list of conversations and evaluates each one against
    the provided rubric and behavioral expectation.

    Args:
        scenario: The behavioral test case.
        expectation: The specific behavioral expectation to judge against.
        conversations: The list of `ConversationRecord` objects to judge.
        rubric: The scoring rubric from Phase 1.
        judge_model: The LLM model for the judge.
        concurrency: Maximum number of judgments to run concurrently.
        retry_config: Configuration for retrying LLM calls.

    Returns:
        A tuple of (scores, reasoning_list).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _judge_with_semaphore(
        conversation: ConversationRecord,
    ) -> tuple[float, str]:
        """Run a single judgment with semaphore control."""
        eval_id = secrets.token_hex(9)
        async with semaphore:
            return await _judge_interaction(
                scenario=scenario,
                expectation=expectation,
                conversation=conversation,
                rubric=rubric,
                judge_model=judge_model,
                eval_id=eval_id,
                retry_config=retry_config,
            )

    tasks = [_judge_with_semaphore(conv) for conv in conversations]

    desc = (
        f"Judging conversations for expectation '{expectation.label}'"
        if expectation.label
        else "Judging conversations"
    )
    results = await tqdm.gather(*tasks, desc=desc)

    scores = [score for score, _ in results]
    reasoning_list = [reasoning for _, reasoning in results]
    return scores, reasoning_list
