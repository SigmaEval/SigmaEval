"""Unit tests for the SigmaEval framework."""

from unittest.mock import patch, Mock, AsyncMock
import os

# Suppress excessive logging from LiteLLM during tests
os.environ["LITELLM_LOG"] = "ERROR"

import pytest

from sigmaeval import (
    SigmaEval,
    ScenarioTest,
    Expectation,
    assertions,
    metrics,
)
from sigmaeval.core.models import ConversationRecord, ConversationTurn


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


@pytest.mark.asyncio
@patch("sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
@patch("sigmaeval.core.framework.ProportionEvaluator")
async def test_assertion_significance_level_overrides_constructor(
    mock_evaluator_class,
    mock_generate_rubric,
    mock_collect_conversations,
    mock_judge_conversations,
) -> None:
    """
    Tests that the significance_level provided in an assertion overrides the
    default value provided in the SigmaEval constructor.
    """
    # Mock setup
    mock_evaluator_class.return_value.evaluate.return_value = {"passed": True}
    mock_generate_rubric.return_value = "Test Rubric"
    mock_collect_conversations.return_value = [ConversationRecord()]
    mock_judge_conversations.return_value = ([10.0], ["reason"])

    # Define a scenario with a specific significance level in the assertion
    scenario = (
        ScenarioTest("Test Override")
        .given("A user")
        .when("An action")
        .sample_size(1)
        .expect_behavior(
            "Something happens",
            criteria=assertions.scores.proportion_gte(
                min_score=8,
                proportion=0.9,
                significance_level=0.99,  # This should override the constructor value
            ),
        )
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


@pytest.mark.asyncio
@patch("sigmaeval.core.framework.ProportionEvaluator")
@patch("sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
async def test_mocked_multiple_expectations_all_pass(
    mock_generate_rubric,
    mock_collect_conversations,
    mock_judge_conversations,
    mock_evaluator_class,
    caplog,
) -> None:
    """
    Tests that a scenario with multiple expectations passes if and only if
    all individual expectations pass.
    """
    # 1. Setup
    mock_evaluator_class.return_value.evaluate.return_value = {"passed": True}
    mock_generate_rubric.return_value = "Test Rubric"
    mock_collect_conversations.return_value = [ConversationRecord()]
    mock_judge_conversations.return_value = ([8.0, 9.0], ["reason", "reason"])

    # Mock the evaluator's result directly to control pass/fail status
    mock_evaluator_instance = Mock()
    mock_evaluator_instance.evaluate.return_value = {"passed": True}
    mock_evaluator_class.return_value = mock_evaluator_instance

    # 2. Define Scenario with two expectations
    scenario = (
        ScenarioTest("Multi-Expectation Test (All Pass)")
        .given("A user")
        .when("An action")
        .sample_size(2)
        .expect_behavior(
            "Criteria 1",
            criteria=assertions.scores.proportion_gte(min_score=7, proportion=0.8),
            label="Expectation 1",
        )
        .expect_behavior(
            "Criteria 2",
            criteria=assertions.scores.proportion_gte(min_score=8, proportion=0.8),
            label="Expectation 2",
        )
    )

    # 3. Run evaluation
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)
    results = await sigma_eval.evaluate(scenario, AsyncMock())

    # 4. Assertions
    assert results.passed is True
    assert len(results.expectation_results) == 2
    assert results.expectation_results[0].passed is True
    assert results.expectation_results[1].passed is True
    assert mock_collect_conversations.call_count == 1
    assert mock_generate_rubric.call_count == 2
    assert mock_evaluator_instance.evaluate.call_count == 2
    assert "Starting statistical analysis for 'Multi-Expectation Test (All Pass)' (Expectation: Expectation 1)" in caplog.text
    assert "Starting statistical analysis for 'Multi-Expectation Test (All Pass)' (Expectation: Expectation 2)" in caplog.text


@pytest.mark.asyncio
@patch("sigmaeval.core.framework.ProportionEvaluator")
@patch("sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
async def test_mocked_multiple_expectations_one_fails(
    mock_generate_rubric,
    mock_collect_conversations,
    mock_judge_conversations,
    mock_evaluator_class,
    caplog,
) -> None:
    """
    Tests that a scenario with multiple expectations fails if any one of the
    individual expectations fails.
    """
    # 1. Setup
    mock_evaluator_class.return_value.evaluate.return_value = {"passed": True}
    mock_generate_rubric.return_value = "Test Rubric"
    mock_collect_conversations.return_value = [ConversationRecord()]
    mock_judge_conversations.return_value = ([8.0, 9.0], ["reason", "reason"])

    # Mock the evaluator's result directly to control pass/fail status
    mock_evaluator_instance = Mock()
    mock_evaluator_instance.evaluate.side_effect = [
        {"passed": True},
        {"passed": False},
    ]
    mock_evaluator_class.return_value = mock_evaluator_instance

    # 2. Define Scenario with two expectations
    scenario = (
        ScenarioTest("Multi-Expectation Test (One Fail)")
        .given("A user")
        .when("An action")
        .sample_size(2)
        .expect_behavior(
            "Criteria 1",
            criteria=assertions.scores.proportion_gte(min_score=7, proportion=0.8),
            label="Expectation 1",
        )
        .expect_behavior(
            "Criteria 2",
            criteria=assertions.scores.proportion_gte(min_score=8, proportion=0.8),
            label="Expectation 2",
        )
    )

    # 3. Run evaluation
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)
    results = await sigma_eval.evaluate(scenario, AsyncMock())

    # 4. Assertions
    assert results.passed is False
    assert len(results.expectation_results) == 2
    assert results.expectation_results[0].passed is True
    assert results.expectation_results[1].passed is False
    assert mock_collect_conversations.call_count == 1
    assert mock_generate_rubric.call_count == 2
    assert mock_evaluator_instance.evaluate.call_count == 2
    assert "Starting statistical analysis for 'Multi-Expectation Test (One Fail)' (Expectation: Expectation 1)" in caplog.text
    assert "Starting statistical analysis for 'Multi-Expectation Test (One Fail)' (Expectation: Expectation 2)" in caplog.text


@pytest.mark.asyncio
@patch("sigmaeval.core.framework.ProportionEvaluator")
@patch("sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
async def test_mocked_multiple_assertions_all_pass(
    mock_generate_rubric,
    mock_collect_conversations,
    mock_judge_conversations,
    mock_evaluator_class,
    caplog,
) -> None:
    """
    Tests that a scenario with multiple assertions passes if and only if
    all individual assertions pass.
    """
    # 1. Setup
    mock_evaluator_class.return_value.evaluate.return_value = {"passed": True}
    mock_generate_rubric.return_value = "Test Rubric"
    mock_collect_conversations.return_value = [ConversationRecord()]
    mock_judge_conversations.return_value = ([8.0, 9.0], ["reason", "reason"])

    # Mock the evaluator's result directly to control pass/fail status
    mock_evaluator_instance = Mock()
    mock_evaluator_instance.evaluate.return_value = {"passed": True}
    mock_evaluator_class.return_value = mock_evaluator_instance

    # 2. Define Scenario with two assertions
    scenario = (
        ScenarioTest("Multi-Assertion Test (All Pass)")
        .given("A user")
        .when("An action")
        .sample_size(2)
        .expect_behavior(
            "Criteria 1",
            criteria=[
                assertions.scores.proportion_gte(min_score=7, proportion=0.8),
                assertions.scores.proportion_gte(min_score=8, proportion=0.8),
            ],
            label="Expectation with multiple assertions",
        )
    )

    # 3. Run evaluation
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)
    results = await sigma_eval.evaluate(scenario, AsyncMock())

    # 4. Assertions
    assert results.passed is True
    assert len(results.expectation_results) == 1
    assert len(results.expectation_results[0].assertion_results) == 2
    assert results.expectation_results[0].passed is True
    assert mock_collect_conversations.call_count == 1
    assert mock_generate_rubric.call_count == 1
    assert mock_evaluator_instance.evaluate.call_count == 2
    assert "Starting statistical analysis for 'Multi-Assertion Test (All Pass)' (Expectation: Expectation with multiple assertions)" in caplog.text


@pytest.mark.asyncio
@patch("sigmaeval.core.framework.ProportionEvaluator")
@patch("sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
async def test_scenario_with_only_behavioral_expectation(
    mock_generate_rubric,
    mock_collect_conversations,
    mock_judge_conversations,
    mock_evaluator_class,
) -> None:
    """
    Tests that a scenario with only a BehavioralExpectation runs correctly.
    """
    # 1. Setup
    mock_evaluator_class.return_value.evaluate.return_value = {"passed": True}
    mock_generate_rubric.return_value = "Test Rubric"
    mock_collect_conversations.return_value = [ConversationRecord(), ConversationRecord()]
    mock_judge_conversations.return_value = ([8.0, 9.0], ["reason", "reason"])

    # 2. Define Scenario
    scenario = (
        ScenarioTest("Behavioral-Only Test")
        .given("A user")
        .when("An action")
        .sample_size(2)
        .expect_behavior(
            "Criteria 1",
            criteria=assertions.scores.proportion_gte(min_score=7, proportion=0.8),
            label="Behavioral Check",
        )
    )

    # 3. Run evaluation
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)
    results = await sigma_eval.evaluate(scenario, AsyncMock())

    # 4. Assertions
    assert results.passed is True
    assert len(results.expectation_results) == 1
    assert results.expectation_results[0].passed is True
    assert len(results.expectation_results[0].scores) == 2
    assert len(results.expectation_results[0].reasoning) == 2
    assert mock_collect_conversations.call_count == 1
    assert mock_generate_rubric.call_count == 1
    assert mock_judge_conversations.call_count == 1
    assert mock_evaluator_class.return_value.evaluate.call_count == 1


@pytest.mark.asyncio
@patch("sigmaeval.core.framework.ProportionEvaluator")
@patch("sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
async def test_scenario_with_only_metric_expectation(
    mock_generate_rubric,
    mock_collect_conversations,
    mock_judge_conversations,
    mock_evaluator_class,
) -> None:
    """
    Tests that a scenario with only a MetricExpectation runs correctly.
    """
    # 1. Setup
    mock_evaluator_class.return_value.evaluate.return_value = {"passed": True}
    
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    mock_collect_conversations.return_value = [
        ConversationRecord(turns=[
            ConversationTurn(role="user", content="hello", request_timestamp=now, response_timestamp=now + timedelta(seconds=0.1)),
            ConversationTurn(role="assistant", content="hi", request_timestamp=now + timedelta(seconds=0.2), response_timestamp=now + timedelta(seconds=0.5)),
        ]),
        ConversationRecord(turns=[
            ConversationTurn(role="user", content="bye", request_timestamp=now, response_timestamp=now + timedelta(seconds=0.1)),
            ConversationTurn(role="assistant", content="goodbye", request_timestamp=now + timedelta(seconds=0.2), response_timestamp=now + timedelta(seconds=0.4)),
        ]),
    ]

    # 2. Define Scenario
    scenario = (
        ScenarioTest("Metric-Only Test")
        .given("A user")
        .when("An action")
        .sample_size(2)
        .expect_metric(
            metrics.per_turn.response_latency,
            criteria=assertions.metrics.proportion_lt(threshold=1.0, proportion=0.9),
            label="Metric Check",
        )
    )

    # 3. Run evaluation
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)
    results = await sigma_eval.evaluate(scenario, AsyncMock())

    # 4. Assertions
    assert results.passed is True
    assert len(results.expectation_results) == 1
    assert results.expectation_results[0].passed is True
    assert len(results.expectation_results[0].scores) > 0
    assert len(results.expectation_results[0].reasoning) == 0
    assert mock_collect_conversations.call_count == 1
    assert mock_generate_rubric.call_count == 0  # Should not be called
    assert mock_judge_conversations.call_count == 0    # Should not be called
    assert mock_evaluator_class.return_value.evaluate.call_count == 1
