import pytest
from unittest.mock import AsyncMock, patch
from sigmaeval import (
    SigmaEval,
    ScenarioTest,
    Expectation,
    assertions,
    metrics,
    AppResponse,
)
from sigmaeval.core.models import ConversationRecord, ConversationTurn
from datetime import datetime, timedelta

@pytest.fixture
def metric_scenario():
    """Fixture for a ScenarioTest with a MetricExpectation."""
    return (
        ScenarioTest("Test Metric Scenario")
        .given("A test user")
        .when("The user does something")
        .sample(10)
        .expect_metric(
            metrics.per_turn.response_latency,
            criteria=assertions.metrics.proportion_lt(threshold=1.0, proportion=0.9),
        )
    )

@pytest.mark.asyncio
async def test_metric_evaluation_proportion_lt(metric_scenario):
    """
    Tests a metric evaluation with a proportion_lt assertion.
    """
    metric_scenario.sample_size = 50
    metric_scenario.then[0].criteria = assertions.metrics.proportion_lt(
        threshold=1.0, proportion=0.9, significance_level=0.05
    )
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    # Mock conversation data to control metric values.
    # With n=50, 49/50 successes (98%) is enough to be confident the true
    # proportion is > 90%.
    conversations = []
    for i in range(50):
        t1 = datetime.now()
        t2 = t1 + timedelta(seconds=0.5 if i < 49 else 2.0)  # 49 pass, 1 fail
        turn = ConversationTurn(
            role="assistant", content="...", request_timestamp=t1, response_timestamp=t2
        )
        user_turn = ConversationTurn(
            role="user", content="...", request_timestamp=t1, response_timestamp=t1
        )
        conversations.append(ConversationRecord(turns=[user_turn, turn]))

    with patch(
        "sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock
    ) as mock_collect:
        mock_collect.return_value = conversations

        results = await sigma_eval.evaluate(
            metric_scenario, AsyncMock(return_value=AppResponse(response="", state={}))
        )

        assert results.passed is True
        details = results.expectation_results[0].assertion_results[0].details
        assert "p_value" in details
        assert details["observed_proportion"] == 0.98


@pytest.mark.asyncio
async def test_metric_evaluation_total_assistant_response_time(metric_scenario):
    """
    Tests a metric evaluation with a total_assistant_response_time metric.
    """
    metric_scenario.then[
        0
    ].metric_definition = metrics.per_conversation.total_assistant_response_time
    metric_scenario.then[0].criteria = assertions.metrics.median_lt(threshold=10.0)
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    # Mock conversation data to control metric values
    conversations = []
    for _ in range(10):
        t1 = datetime.now()
        t2 = t1 + timedelta(seconds=0.5)
        t3 = t2 + timedelta(seconds=1.0)
        t4 = t3 + timedelta(seconds=1.5)
        
        user_turn_1 = ConversationTurn(
            role="user", content="...", request_timestamp=t1, response_timestamp=t1
        )
        assistant_turn_1 = ConversationTurn(
            role="assistant", content="...", request_timestamp=t1, response_timestamp=t2
        )
        user_turn_2 = ConversationTurn(
            role="user", content="...", request_timestamp=t3, response_timestamp=t3
        )
        assistant_turn_2 = ConversationTurn(
            role="assistant", content="...", request_timestamp=t3, response_timestamp=t4
        )
        conversations.append(
            ConversationRecord(
                turns=[
                    user_turn_1,
                    assistant_turn_1,
                    user_turn_2,
                    assistant_turn_2,
                ]
            )
        )

    with patch(
        "sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock
    ) as mock_collect:
        mock_collect.return_value = conversations

        results = await sigma_eval.evaluate(
            metric_scenario, AsyncMock(return_value=AppResponse(response="", state={}))
        )

        assert results.passed is True
        details = results.expectation_results[0].assertion_results[0].details
        assert "p_value" in details
        # 0.5s (turn 1) + 1.5s (turn 2) = 2.0s
        assert details["observed_median"] == 2.0


@pytest.mark.asyncio
async def test_metric_evaluation_response_length_chars(metric_scenario):
    """
    Tests a metric evaluation with a response_length_chars metric (per_turn).
    """
    metric_scenario.then[0].metric_definition = metrics.per_turn.response_length_chars
    metric_scenario.then[0].criteria = assertions.metrics.proportion_lt(
        threshold=15, proportion=0.95
    )
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    # With an n of 60 (30 conversations * 2 turns), observing 100% success
    # is enough to be statistically confident the true proportion is > 95%.
    conversations = []
    for _ in range(30):
        assistant_turn_1 = ConversationTurn(
            role="assistant",
            content="Hello",  # 5 chars
            request_timestamp=datetime.now(),
            response_timestamp=datetime.now(),
        )
        assistant_turn_2 = ConversationTurn(
            role="assistant",
            content="Hello again",  # 11 chars
            request_timestamp=datetime.now(),
            response_timestamp=datetime.now(),
        )
        conversations.append(
            ConversationRecord(turns=[assistant_turn_1, assistant_turn_2])
        )

    with patch(
        "sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock
    ) as mock_collect:
        mock_collect.return_value = conversations

        # The metric calculator returns a list of values. We can check the length
        # of this list to confirm the number of observations.
        metric_values = metric_scenario.then[0].metric_definition.calculator(conversations[0])
        assert len(metric_values) * len(conversations) == 60

        results = await sigma_eval.evaluate(
            metric_scenario, AsyncMock(return_value=AppResponse(response="", state={}))
        )

        assert results.passed is True
        details = results.expectation_results[0].assertion_results[0].details
        assert details["observed_proportion"] == 1.0


@pytest.mark.asyncio
async def test_metric_evaluation_total_assistant_response_chars(metric_scenario):
    """
    Tests a metric evaluation with a total_assistant_response_chars metric.
    """
    metric_scenario.then[
        0
    ].metric_definition = metrics.per_conversation.total_assistant_response_chars
    metric_scenario.then[0].criteria = assertions.metrics.median_lt(threshold=20.0)
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    conversations = []
    for _ in range(10):
        assistant_turn_1 = ConversationTurn(
            role="assistant",
            content="Hello",  # 5 chars
            request_timestamp=datetime.now(),
            response_timestamp=datetime.now(),
        )
        assistant_turn_2 = ConversationTurn(
            role="assistant",
            content="Hello again",  # 11 chars
            request_timestamp=datetime.now(),
            response_timestamp=datetime.now(),
        )
        conversations.append(
            ConversationRecord(turns=[assistant_turn_1, assistant_turn_2])
        )

    with patch(
        "sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock
    ) as mock_collect:
        mock_collect.return_value = conversations

        results = await sigma_eval.evaluate(
            metric_scenario, AsyncMock(return_value=AppResponse(response="", state={}))
        )

        assert results.passed is True
        details = results.expectation_results[0].assertion_results[0].details
        assert details["observed_median"] == 16.0  # 5 + 11


@pytest.mark.asyncio
async def test_metric_evaluation_median_lt(metric_scenario):
    """
    Tests a metric evaluation with a median_lt assertion.
    """
    metric_scenario.then[0].criteria = assertions.metrics.median_lt(threshold=1.0)
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    # Mock conversation data to control metric values
    conversations = []
    for i in range(10):
        t1 = datetime.now()
        t2 = t1 + timedelta(seconds=0.5)
        turn = ConversationTurn(
            role="assistant", content="...", request_timestamp=t1, response_timestamp=t2
        )
        user_turn = ConversationTurn(
            role="user", content="...", request_timestamp=t1, response_timestamp=t1
        )
        conversations.append(ConversationRecord(turns=[user_turn, turn]))

    with patch(
        "sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock
    ) as mock_collect:
        mock_collect.return_value = conversations

        results = await sigma_eval.evaluate(
            metric_scenario, AsyncMock(return_value=AppResponse(response="", state={}))
        )

        assert results.passed is True
        details = results.expectation_results[0].assertion_results[0].details
        assert "p_value" in details
        assert details["observed_median"] == 0.5
