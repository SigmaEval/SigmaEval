"""Minimal tests for the SigmaEval framework."""

from typing import Any, Dict
import os
from dotenv import load_dotenv
import logging

import pytest

from sigmaeval import (
    SigmaEval,
    BehavioralTest,
    Expectation,
    AppResponse,
    SuccessRateEvaluator,
    EvaluationResult,
    RetryConfig,
    WritingStyleConfig,
)
from tests.example_apps.simple_chat_app import SimpleChatApp


@pytest.mark.parametrize(
    "valid_model",
    [
        "openai/gpt-4o",
        "anthropic/claude-3-opus",
        "local/test-model",
    ],
)
def test_sigmaeval_init_stores_model(valid_model: str) -> None:
    se = SigmaEval(judge_model=valid_model)
    assert se.judge_model == valid_model
    assert se.user_simulator_model == valid_model


def test_sigmaeval_init_stores_separate_models() -> None:
    se = SigmaEval(
        judge_model="openai/gpt-4o", user_simulator_model="openai/gpt-3.5-turbo"
    )
    assert se.judge_model == "openai/gpt-4o"
    assert se.user_simulator_model == "openai/gpt-3.5-turbo"


@pytest.mark.parametrize("invalid_model", ["", "   "])
def test_sigmaeval_init_invalid_model_raises(invalid_model: str) -> None:
    with pytest.raises(ValueError):
        SigmaEval(judge_model=invalid_model)


def test_sigmaeval_init_non_string_raises() -> None:
    with pytest.raises(ValueError):
        SigmaEval(judge_model=None)  # type: ignore[arg-type]


@pytest.mark.integration
async def test_e2e_evaluation_with_simple_example_app(caplog) -> None:
    """
    Runs a full end-to-end test of SigmaEval against the example chat app.

    This test requires both TEST_EVAL_MODEL (for the judge) and
    TEST_APP_MODEL (for the app under test) to be set in the environment.
    """
    # 1. Setup: Load environment and get model names
    load_dotenv()
    eval_model = os.getenv("TEST_EVAL_MODEL")
    app_model = os.getenv("TEST_APP_MODEL")
    if not eval_model or not app_model:
        pytest.skip(
            "TEST_EVAL_MODEL and TEST_APP_MODEL env vars must be set to run this test."
        )
    
    sample_size = 10 # Keep sample size low for a quick test

    # 2. Define the Behavioral Test
    scenario = BehavioralTest(
        title="Bot handles a product return request",
        given="A user wants to return a recently purchased pair of shoes.",
        when="The user asks how to start a return.",
        then=Expectation(
            expected_behavior=(
                "The bot should acknowledge the user's request, ask for an order "
                "number, and explain the next steps in the return process clearly."
            ),
            evaluator=SuccessRateEvaluator(
                significance_level=0.05,
                min_proportion=0.7,
                sample_size=sample_size,
            ),
        ),
        max_turns=5,
    )

    # 3. Create the app handler to bridge SigmaEval and the target app
    chat_app = SimpleChatApp(model=app_model)

    async def app_handler(message: str, state: Dict[str, Any]) -> AppResponse:
        """
        This bridge function takes messages from SigmaEval's user simulator,
        passes them to the chat app, and manages conversation history
        in the state dictionary.
        """
        history = state.get("history", [])
        response_text, updated_history = await chat_app.respond(
            user_message=message, history=history
        )
        return AppResponse(response=response_text, state={"history": updated_history})

    # 4. Run the evaluation
    sigma_eval = SigmaEval(
        judge_model=eval_model,
        log_level=logging.INFO,
        retry_config=RetryConfig(enabled=False),
    )
    with caplog.at_level(logging.INFO):
        results = await sigma_eval.evaluate(scenario, app_handler)

    # 5. Assert the results to ensure the evaluation ran correctly
    assert isinstance(results, EvaluationResult)
    assert results.test_config["title"] == scenario.title
    assert results.rubric and len(results.rubric) > 0
    assert results.scores and len(results.scores) == sample_size
    assert results.conversations and len(results.conversations) == sample_size

    # Check that 70% of test scores are above 5 (indicating good performance)
    scores = results.scores
    scores_above_5 = [score for score in scores if score > 5]
    proportion_above_5 = len(scores_above_5) / len(scores)
    
    # Check if at least 70% of scores are above 5
    assert proportion_above_5 >= 0.7, f"Only {proportion_above_5:.1%} of scores are above 5. Expected at least 70% of scores to be above 5."
    
    # Print warning if some scores are below 5 but test still passes
    scores_below_5 = [score for score in scores if score <= 5]
    if scores_below_5:
        print(f"WARNING: {len(scores_below_5)} out of {len(scores)} scores are below 5: {scores_below_5}")

    # Check that a multi-turn conversation was recorded
    first_conversation = results.conversations[0]
    assert len(first_conversation.turns) > 1
    assert first_conversation.turns[0]["role"] == "user"
    assert first_conversation.turns[1]["role"] == "assistant"
    assert first_conversation.writing_style is not None

    # 6. Assert logging output
    assert "--- Starting evaluation" in caplog.text
    assert f"Collecting {sample_size} samples..." in caplog.text
    assert "--- Evaluation complete ---" in caplog.text
    assert "Generated rubric" not in caplog.text  # DEBUG message




@pytest.mark.integration
async def test_e2e_evaluation_with_bad_app_returns_low_scores(caplog) -> None:
    """
    Runs an end-to-end test of SigmaEval against a deliberately bad app handler
    that returns gibberish, expecting overall low scores from the judge.

    This test requires both TEST_EVAL_MODEL (for the judge) and
    TEST_APP_MODEL (kept for parity with the other integration test) to be set
    in the environment.
    """
    # 1. Setup: Load environment and get model names
    load_dotenv()
    eval_model = os.getenv("TEST_EVAL_MODEL")
    app_model = os.getenv("TEST_APP_MODEL")
    if not eval_model or not app_model:
        pytest.skip(
            "TEST_EVAL_MODEL and TEST_APP_MODEL env vars must be set to run this test."
        )

    sample_size = 10  # Keep sample size low for a quick test

    # 2. Define the Behavioral Test (same scenario as the good app test)
    scenario = BehavioralTest(
        title="Bot handles a product return request",
        given="A user wants to return a recently purchased pair of shoes.",
        when="The user asks how to start a return.",
        then=Expectation(
            expected_behavior=(
                "The bot should acknowledge the user's request, ask for an order "
                "number, and explain the next steps in the return process clearly."
            ),
            evaluator=SuccessRateEvaluator(
                significance_level=0.05,
                min_proportion=0.7,
                sample_size=sample_size,
            ),
        ),
        max_turns=5,
    )

    # 3. Create a bad app handler that returns gibberish and does NOT call SimpleChatApp
    async def app_handler(message: str, state: Dict[str, Any]) -> AppResponse:
        """
        This intentionally poor handler ignores the input and returns gibberish,
        returning the state unchanged.
        """
        response_text = os.urandom(16).hex()
        return AppResponse(response=response_text, state=state)

    # 4. Run the evaluation
    sigma_eval = SigmaEval(
        judge_model=eval_model,
        log_level=logging.DEBUG,
        retry_config=RetryConfig(enabled=False),
        writing_style_config=WritingStyleConfig(enabled=False),
    )
    with caplog.at_level(logging.DEBUG):
        results = await sigma_eval.evaluate(scenario, app_handler)

    # 5. Assert the results to ensure the evaluation ran and produced low scores
    assert isinstance(results, EvaluationResult)
    assert results.test_config["title"] == scenario.title
    assert results.rubric and len(results.rubric) > 0
    assert results.scores and len(results.scores) == sample_size
    assert results.conversations and len(results.conversations) == sample_size

    # Expect that at most 30% of scores are above 2 for this bad app
    scores = results.scores
    scores_above_2 = [score for score in scores if score > 2]
    proportion_above_2 = len(scores_above_2) / len(scores)
    assert (
        proportion_above_2 <= 0.3
    ), f"Too many high scores for a bad app: {proportion_above_2:.1%} above 2 (expected <= 30%)."

    # Check that a multi-turn conversation was recorded
    first_conversation = results.conversations[0]
    assert len(first_conversation.turns) > 1
    assert first_conversation.turns[0]["role"] == "user"
    assert first_conversation.turns[1]["role"] == "assistant"
    assert first_conversation.writing_style is None

    # 6. Assert logging output
    assert "--- Starting evaluation" in caplog.text
    assert "Generated rubric" in caplog.text
    assert "Collected scores" in caplog.text
    assert "Judge prompt" in caplog.text
