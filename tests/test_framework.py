"""Minimal tests for the SigmaEval framework."""

from typing import Any, Dict
import os
from dotenv import load_dotenv
import logging
from unittest.mock import patch, Mock, AsyncMock

# Suppress excessive logging from LiteLLM during tests
os.environ["LITELLM_LOG"] = "ERROR"

import pytest

from sigmaeval import (
    SigmaEval,
    ScenarioTest,
    BehavioralExpectation,
    AppResponse,
    EvaluationResult,
    RetryConfig,
    WritingStyleConfig,
    WritingStyleAxes,
    ConversationRecord,
    assertions,
    ConversationTurn,
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
    se = SigmaEval(judge_model=valid_model, significance_level=0.05)
    assert se.judge_model == valid_model
    assert se.user_simulator_model == valid_model


def test_sigmaeval_init_stores_separate_models() -> None:
    se = SigmaEval(
        judge_model="openai/gpt-4o",
        user_simulator_model="openai/gpt-3.5-turbo",
        significance_level=0.05,
    )
    assert se.judge_model == "openai/gpt-4o"
    assert se.user_simulator_model == "openai/gpt-3.5-turbo"


@pytest.mark.parametrize("invalid_model", ["", "   "])
def test_sigmaeval_init_invalid_model_raises(invalid_model: str) -> None:
    with pytest.raises(ValueError):
        SigmaEval(judge_model=invalid_model, significance_level=0.05)


def test_sigmaeval_init_non_string_raises() -> None:
    with pytest.raises(ValueError):
        SigmaEval(judge_model=None, significance_level=0.05)  # type: ignore[arg-type]


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
    scenario = ScenarioTest(
        title="Bot handles a product return request",
        given="A user wants to return a recently purchased pair of shoes.",
        when="The user asks how to start a return.",
        sample_size=sample_size,
        then=BehavioralExpectation(
            expected_behavior=(
                "The bot should acknowledge the user's request, ask for an order "
                "number, and explain the next steps in the return process clearly."
            ),
            criteria=assertions.scores.proportion_gte(
                min_score=6,
                proportion=0.7,
                significance_level=0.05,
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
        significance_level=0.05,
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
    assert first_conversation.turns[0].role == "user"
    assert first_conversation.turns[1].role == "assistant"
    assert first_conversation.writing_style is not None

    # 6. Assert logging output
    assert "--- Starting evaluation" in caplog.text
    assert f"Collecting {sample_size} samples for '{scenario.title}'..." in caplog.text
    assert f"--- Evaluation complete for: {scenario.title} ---" in caplog.text
    assert "Generated rubric" not in caplog.text  # DEBUG message


def test_evaluation_result_properties_and_methods():
    """
    Tests the convenience properties and methods of the EvaluationResult class.
    """
    scores = [1.0, 5.0, 7.5, 8.0, 9.5]
    reasoning = ["r1", "r2", "r3", "r4", "r5"]
    
    # Create mock ConversationRecord objects with valid ConversationTurn objects
    # including dummy timestamps, as they are now required fields.
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    conversations = [
        ConversationRecord(turns=[ConversationTurn(role="user", content="hello", request_timestamp=now, response_timestamp=now)]),
        ConversationRecord(turns=[ConversationTurn(role="user", content="world", request_timestamp=now, response_timestamp=now)]),
        ConversationRecord(turns=[ConversationTurn(role="user", content="foo", request_timestamp=now, response_timestamp=now)]),
        ConversationRecord(turns=[ConversationTurn(role="user", content="bar", request_timestamp=now, response_timestamp=now)]),
        ConversationRecord(turns=[ConversationTurn(role="user", content="baz", request_timestamp=now, response_timestamp=now)]),
    ]

    results_pass = EvaluationResult(
        judge_model="test/judge",
        user_simulator_model="test/simulator",
        test_config={"title": "Test Pass"},
        retry_config=RetryConfig(),
        rubric="Test Rubric",
        scores=scores,
        reasoning=reasoning,
        conversations=conversations,
        num_conversations=len(scores),
        results={"passed": True, "p_value": 0.01},
    )

    # Test properties for a passing result
    assert results_pass.passed is True
    assert results_pass.p_value == 0.01
    assert results_pass.average_score == 6.2
    assert results_pass.median_score == 7.5
    assert results_pass.min_score == 1.0
    assert results_pass.max_score == 9.5
    assert results_pass.std_dev_score == pytest.approx(2.977, abs=0.001)

    # Test methods
    worst_score, worst_reasoning, worst_convo = results_pass.get_worst_conversation()
    assert worst_score == 1.0
    assert worst_reasoning == "r1"
    assert worst_convo == conversations[0]

    best_score, best_reasoning, best_convo = results_pass.get_best_conversation()
    assert best_score == 9.5
    assert best_reasoning == "r5"
    assert best_convo == conversations[4]

    # Test __str__ method
    summary_str = str(results_pass)
    assert "--- Test Pass: ✅ PASSED ---" in summary_str
    assert "P-value: 0.0100" in summary_str
    assert "Average Score: 6.20" in summary_str
    assert "Passed: True" in summary_str
    
    # Test a failing result
    results_fail = EvaluationResult(
        judge_model="test/judge",
        user_simulator_model="test/simulator",
        test_config={"title": "Test Fail"},
        retry_config=RetryConfig(),
        rubric="Test Rubric",
        scores=scores,
        reasoning=reasoning,
        conversations=conversations,
        num_conversations=len(scores),
        results={"passed": False, "p_value": 0.88},
    )
    
    assert results_fail.passed is False
    assert "--- Test Fail: ❌ FAILED ---" in str(results_fail)


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
    scenario = ScenarioTest(
        title="Bot handles a product return request",
        given="A user wants to return a recently purchased pair of shoes.",
        when="The user asks how to start a return.",
        sample_size=sample_size,
        then=BehavioralExpectation(
            expected_behavior=(
                "The bot should acknowledge the user's request, ask for an order "
                "number, and explain the next steps in the return process clearly."
            ),
            criteria=assertions.scores.proportion_gte(
                min_score=6,
                proportion=0.7,
                significance_level=0.05,
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
    assert first_conversation.turns[0].role == "user"
    assert first_conversation.turns[1].role == "assistant"
    assert first_conversation.writing_style is None

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
        pytest.skip(
            "TEST_EVAL_MODEL and TEST_APP_MODEL env vars must be set to run this test."
        )

    sample_size = 2  # Keep sample size low for a quick test

    # 2. Define the Behavioral Test
    scenario = ScenarioTest(
        title="Bot handles a simple greeting",
        given="A user starts a conversation.",
        when="The user says 'hello'.",
        sample_size=sample_size,
        then=BehavioralExpectation(
            expected_behavior="The bot should respond with a friendly greeting.",
            criteria=assertions.scores.proportion_gte(
                min_score=6,
                proportion=0.6,
                significance_level=0.05,
            ),
        ),
        max_turns=2,
    )

    # 3. Create the app handler
    chat_app = SimpleChatApp(model=app_model)

    async def app_handler(message: str, state: Dict[str, Any]) -> AppResponse:
        history = state.get("history", [])
        response_text, updated_history = await chat_app.respond(
            user_message=message, history=history
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
    assert isinstance(results, EvaluationResult)
    assert len(results.scores) == sample_size
    assert len(results.conversations) == sample_size

    # Key assertion: check that the custom writing style was used for all conversations
    for convo in results.conversations:
        assert convo.writing_style is not None
        assert convo.writing_style["Proficiency"] == "TEST_PROFICIENCY"
        assert convo.writing_style["Tone"] == "TEST_TONE"
        assert convo.writing_style["Verbosity"] == "TEST_VERBOSITY"
        assert convo.writing_style["Formality"] == "TEST_FORMALITY"


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
    scenario_1 = ScenarioTest(
        title="Minimal Test 1",
        given="A user",
        when="The user says hi",
        sample_size=sample_size,
        then=BehavioralExpectation(
            expected_behavior="The bot says hi back.",
            criteria=assertions.scores.proportion_gte(
                min_score=6,
                proportion=0.1,
                significance_level=0.05,
            ),
        ),
        max_turns=2,
    )
    scenario_2 = ScenarioTest(
        title="Minimal Test 2",
        given="A user",
        when="The user says bye",
        sample_size=sample_size,
        then=BehavioralExpectation(
            expected_behavior="The bot says bye back.",
            criteria=assertions.scores.proportion_gte(
                min_score=6,
                proportion=0.1,
                significance_level=0.05,
            ),
        ),
        max_turns=2,
    )
    test_suite = [scenario_1, scenario_2]

    # 3. Create a simple, fixed-response app handler
    async def app_handler(message: str, state: Dict[str, Any]) -> AppResponse:
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
    assert isinstance(all_results[0], EvaluationResult)
    assert all_results[0].test_config["title"] == "Minimal Test 1"
    assert all_results[1].test_config["title"] == "Minimal Test 2"

    # 6. Assert logging output for test suites
    assert "--- Starting evaluation for test suite with 2 scenarios ---" in caplog.text
    assert "--- Test suite evaluation complete ---" in caplog.text
    # Check that individual test logs are also present
    assert "--- Starting evaluation for ScenarioTest: Minimal Test 1 ---" in caplog.text
    assert "--- Evaluation complete for: Minimal Test 1 ---" in caplog.text


@pytest.mark.asyncio
@patch("sigmaeval.core.framework.collect_evaluation_data")
@patch("sigmaeval.core.framework._generate_rubric")
@patch("sigmaeval.core.framework.RatingProportionEvaluator")
async def test_assertion_significance_level_overrides_constructor(
    mock_evaluator_class, mock_generate_rubric, mock_collect_evaluation_data
) -> None:
    """
    Tests that the significance_level provided in an assertion overrides the
    default value provided in the SigmaEval constructor.
    """
    # Mock setup
    mock_evaluator_instance = Mock()
    mock_evaluator_instance.evaluate.return_value = {"passed": True}
    mock_evaluator_class.return_value = mock_evaluator_instance
    mock_generate_rubric.return_value = "Test Rubric"
    mock_collect_evaluation_data.return_value = ([10.0], ["reason"], [])

    # Define a scenario with a specific significance level in the assertion
    scenario = ScenarioTest(
        title="Test Override",
        given="A user",
        when="An action",
        sample_size=1,
        then=BehavioralExpectation(
            expected_behavior="Something happens",
            criteria=assertions.scores.proportion_gte(
                min_score=8,
                proportion=0.9,
                significance_level=0.99,  # This should override the constructor value
            ),
        ),
    )

    # Initialize SigmaEval with a different significance level
    sigma_eval = SigmaEval(
        judge_model="test/model",
        significance_level=0.05,  # This should be overridden
    )
    await sigma_eval.evaluate(scenario, AsyncMock())

    # Assert that the evaluator was called with the correct, overridden significance level
    mock_evaluator_class.assert_called_once()
    _, kwargs = mock_evaluator_class.call_args
    assert kwargs["significance_level"] == 0.99
