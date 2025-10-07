"""Integration tests for the SigmaEval framework."""

from typing import Any
import os
from dotenv import load_dotenv
import logging

# Suppress excessive logging from LiteLLM during tests
os.environ["LITELLM_LOG"] = "ERROR"

import pytest

from sigmaeval import (
    SigmaEval,
    ScenarioTest,
    AppResponse,
    ScenarioTestResult,
    RetryConfig,
    WritingStyleConfig,
    WritingStyleAxes,
    assertions,
    metrics,
)
from tests.example_apps.simple_chat_app import SimpleChatApp


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
        pytest.skip("TEST_EVAL_MODEL and TEST_APP_MODEL env vars must be set to run this test.")

    sample_size = 10  # Keep sample size low for a quick test

    # 2. Define the Behavioral Test
    scenario = (
        ScenarioTest("Bot handles a product return request")
        .given("A user wants to return a recently purchased pair of shoes.")
        .when("The user asks how to start a return.")
        .sample_size(sample_size)
        .max_turns(3)
        .expect_behavior(
            "The bot should acknowledge the user's request, ask for an order "
            "number, and explain the next steps in the return process clearly.",
            criteria=[
                assertions.scores.proportion_gte(min_score=6, proportion=0.7),
                assertions.scores.median_gte(threshold=6.0),
            ],
            label="Correctness",
        )
        .expect_metric(
            metrics.per_turn.response_latency,
            criteria=assertions.metrics.median_lt(threshold=10.0),
            label="Responsiveness",
        )
    )

    # 3. Create the app handler to bridge SigmaEval and the target app
    chat_app = SimpleChatApp(model=app_model)

    async def app_handler(messages: list[dict[str, str]], state: dict[str, Any]) -> AppResponse:
        """
        This bridge function takes messages from SigmaEval's user simulator,
        extracts the latest message and history, passes them to the chat app
        in the format it expects, and manages conversation history in the
        state dictionary.
        """
        history = state.get("history", [])
        user_message = messages[-1]["content"]

        # The chat_app expects the history *before* the current message
        history_for_app = [msg for msg in messages[:-1] if msg["role"] != "system"]

        response_text, updated_history = await chat_app.respond(
            user_message=user_message, history=history_for_app
        )
        return AppResponse(response=response_text, state={"history": updated_history})

    # 4. Run the evaluation
    sigma_eval = SigmaEval(
        judge_model=eval_model,
        log_level=logging.INFO,
        retry_config=RetryConfig(enabled=False),
        significance_level=0.05,
    )
    with caplog.at_level(logging.INFO):
        results = await sigma_eval.evaluate(scenario, app_handler)

    # 5. Assert the results to ensure the evaluation ran correctly
    assert isinstance(results, ScenarioTestResult)
    assert results.title == scenario.title
    assert results.rubric and len(results.rubric) > 0
    assert results.conversations and len(results.conversations) == sample_size

    # Check that the overall test passed
    assert results.passed is True

    # Check individual expectation results
    assert len(results.expectation_results) == 2
    assert results.expectation_results[0].passed is True
    assert results.expectation_results[1].passed is True

    # Check that the behavioral expectation has two passing assertion results
    assert len(results.expectation_results[0].assertion_results) == 2
    assert results.expectation_results[0].assertion_results[0].passed is True
    assert results.expectation_results[0].assertion_results[1].passed is True

    # Check that at least 70% of scores are above 5 (indicating good performance)
    scores = results.expectation_results[0].scores
    assert scores and len(scores) == sample_size
    scores_above_5 = [score for score in scores if score > 5]
    proportion_above_5 = len(scores_above_5) / len(scores)
    assert proportion_above_5 >= 0.7, f"Only {proportion_above_5:.1%} of scores are above 5."

    # Check that reasoning is available
    assert (
        results.expectation_results[0].reasoning
        and len(results.expectation_results[0].reasoning) == sample_size
    )

    # Check that a multi-turn conversation was recorded
    first_conversation = results.conversations[0]
    assert len(first_conversation.turns) > 0
    assert first_conversation.details["writing_style"] is not None

    # 6. Assert logging output
    assert "--- Starting evaluation" in caplog.text
    assert f"Simulating {sample_size} conversations for '{scenario.title}'..." in caplog.text
    assert f"--- Evaluation complete for: {scenario.title} ---" in caplog.text
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
        pytest.skip("TEST_EVAL_MODEL and TEST_APP_MODEL env vars must be set to run this test.")

    sample_size = 10  # Keep sample size low for a quick test

    # 2. Define the Behavioral Test (same scenario as the good app test)
    scenario = (
        ScenarioTest("Bot handles a product return request")
        .given("A user wants to return a recently purchased pair of shoes.")
        .when("The user asks how to start a return.")
        .sample_size(sample_size)
        .max_turns(3)
        .expect_behavior(
            "The bot should acknowledge the user's request, ask for an order "
            "number, and explain the next steps in the return process clearly.",
            criteria=assertions.scores.proportion_gte(min_score=6, proportion=0.9),
            label="Correctness",
        )
        .expect_metric(
            metrics.per_turn.response_latency,
            criteria=assertions.metrics.median_lt(threshold=0.1),
            label="Responsiveness",
        )
    )

    # 3. Create a bad app handler that returns gibberish and does NOT call SimpleChatApp
    async def app_handler(messages: list[dict[str, str]], state: dict[str, Any]) -> AppResponse:
        """
        This intentionally poor handler ignores the input and returns gibberish,
        returning the state unchanged.
        """
        response_text = "Go away"
        return AppResponse(response=response_text, state=state)

    # 4. Run the evaluation
    sigma_eval = SigmaEval(
        judge_model=eval_model,
        log_level=logging.DEBUG,
        retry_config=RetryConfig(enabled=False),
        writing_style_config=WritingStyleConfig(enabled=False),
        significance_level=0.05,
    )
    with caplog.at_level(logging.DEBUG):
        results = await sigma_eval.evaluate(scenario, app_handler)

    # 5. Assert the results to ensure the evaluation ran and produced low scores
    assert isinstance(results, ScenarioTestResult)
    assert results.title == scenario.title
    assert results.rubric and len(results.rubric) > 0
    assert results.conversations and len(results.conversations) == sample_size

    # Expect that at most 30% of scores are above 2 for this bad app
    scores = results.expectation_results[0].scores
    assert scores and len(scores) == sample_size
    scores_above_2 = [score for score in scores if score > 2]
    proportion_above_2 = len(scores_above_2) / len(scores)
    assert (
        proportion_above_2 <= 0.3
    ), f"Too many high scores for a bad app: {proportion_above_2:.1%}."

    # Check that reasoning is available
    assert (
        results.expectation_results[0].reasoning
        and len(results.expectation_results[0].reasoning) == sample_size
    )

    # The overall test should FAIL because the behavioral expectation is not met.
    assert results.passed is False

    # However, we can assert that the individual expectations behaved as predicted:
    # - The "Correctness" check should fail because the app is bad.
    # - The "Responsiveness" check should pass because the app is fast.
    assert results.expectation_results[0].passed is False
    assert results.expectation_results[1].passed is True

    # Check that a multi-turn conversation was recorded
    first_conversation = results.conversations[0]
    assert len(first_conversation.turns) > 0
    assert first_conversation.details["writing_style"] is None

    # 6. Assert logging output
    assert "--- Starting evaluation" in caplog.text
    assert "Generated rubric" in caplog.text
    assert "Collected scores" in caplog.text
    assert "Judge prompt" in caplog.text


@pytest.mark.integration
async def test_e2e_evaluation_with_custom_writing_style(caplog) -> None:
    """
    Runs an end-to-end test to verify that a custom WritingStyleConfig is
    correctly used during evaluation.
    """
    # 1. Setup: Load environment and get model names
    load_dotenv()
    eval_model = os.getenv("TEST_EVAL_MODEL")
    app_model = os.getenv("TEST_APP_MODEL")
    if not eval_model or not app_model:
        pytest.skip("TEST_EVAL_MODEL and TEST_APP_MODEL env vars must be set to run this test.")

    sample_size = 2  # Keep sample size low for a quick test

    # 2. Define the Behavioral Test
    scenario = (
        ScenarioTest("Bot handles a simple greeting")
        .given("A user starts a conversation.")
        .when("The user says 'hello'.")
        .sample_size(sample_size)
        .max_turns(2)
        .expect_behavior(
            "The bot should respond with a friendly greeting.",
            criteria=assertions.scores.proportion_gte(
                min_score=6,
                proportion=0.6,
                significance_level=0.05,
            ),
        )
    )

    # 3. Create the app handler
    chat_app = SimpleChatApp(model=app_model)

    async def app_handler(messages: list[dict[str, str]], state: dict[str, Any]) -> AppResponse:
        history = state.get("history", [])
        user_message = messages[-1]["content"]
        history_for_app = [msg for msg in messages[:-1] if msg["role"] != "system"]
        response_text, updated_history = await chat_app.respond(
            user_message=user_message, history=history_for_app
        )
        return AppResponse(response=response_text, state={"history": updated_history})

    # 4. Define and run the evaluation with a custom writing style
    custom_axes = WritingStyleAxes(
        proficiency=["TEST_PROFICIENCY"],
        tone=["TEST_TONE"],
        verbosity=["TEST_VERBOSITY"],
        formality=["TEST_FORMALITY"],
    )
    custom_style_config = WritingStyleConfig(axes=custom_axes)

    sigma_eval = SigmaEval(
        judge_model=eval_model,
        log_level=logging.INFO,
        writing_style_config=custom_style_config,
        significance_level=0.05,
    )
    results = await sigma_eval.evaluate(scenario, app_handler)

    # 5. Assert the results
    assert isinstance(results, ScenarioTestResult)
    assert len(results.conversations) == sample_size

    # Key assertion: check that the custom writing style was used for all conversations
    for convo in results.conversations:
        assert convo.details["writing_style"] is not None
        assert convo.details["writing_style"]["Proficiency"] == "TEST_PROFICIENCY"
        assert convo.details["writing_style"]["Tone"] == "TEST_TONE"
        assert convo.details["writing_style"]["Verbosity"] == "TEST_VERBOSITY"
        assert convo.details["writing_style"]["Formality"] == "TEST_FORMALITY"


@pytest.mark.integration
async def test_e2e_evaluation_with_test_suite(caplog) -> None:
    """
    Runs a minimal end-to-end test to verify that evaluating a list of
    scenarios (a test suite) works correctly.
    """
    # 1. Setup: Load environment and get model names
    load_dotenv()
    eval_model = os.getenv("TEST_EVAL_MODEL")
    if not eval_model:
        pytest.skip("TEST_EVAL_MODEL env var must be set to run this test.")

    sample_size = 1  # Keep sample size at 1 for a very fast test

    # 2. Define two minimal Behavioral Tests
    scenario_1 = (
        ScenarioTest("Minimal Test 1")
        .given("A user")
        .when("The user says hi")
        .sample_size(sample_size)
        .max_turns(2)
        .expect_behavior(
            "The bot says hi back.",
            criteria=assertions.scores.proportion_gte(
                min_score=6,
                proportion=0.1,
                significance_level=0.05,
            ),
        )
    )
    scenario_2 = (
        ScenarioTest("Minimal Test 2")
        .given("A user")
        .when("The user says bye")
        .sample_size(sample_size)
        .max_turns(2)
        .expect_behavior(
            "The bot says bye back.",
            criteria=assertions.scores.proportion_gte(
                min_score=6,
                proportion=0.1,
                significance_level=0.05,
            ),
        )
    )
    test_suite = [scenario_1, scenario_2]

    # 3. Create a simple, fixed-response app handler
    async def app_handler(messages: list[dict[str, str]], state: dict[str, Any]) -> AppResponse:
        return AppResponse(response="ok", state=state)

    # 4. Run the evaluation on the full suite
    sigma_eval = SigmaEval(
        judge_model=eval_model,
        log_level=logging.INFO,
        writing_style_config=WritingStyleConfig(enabled=False),
        significance_level=0.05,
    )
    with caplog.at_level(logging.INFO):
        all_results = await sigma_eval.evaluate(test_suite, app_handler)

    # 5. Assert the results
    assert isinstance(all_results, list)
    assert len(all_results) == 2
    assert isinstance(all_results[0], ScenarioTestResult)
    assert all_results[0].title == "Minimal Test 1"
    assert all_results[1].title == "Minimal Test 2"

    # 6. Assert logging output for test suites
    assert "--- Starting evaluation for test suite with 2 scenarios ---" in caplog.text
    assert "--- Test suite evaluation complete ---" in caplog.text
    # Check that individual test logs are also present
    assert "--- Starting evaluation for ScenarioTest: Minimal Test 1 ---" in caplog.text
    assert "--- Evaluation complete for: Minimal Test 1 ---" in caplog.text
