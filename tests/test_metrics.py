import pytest
from unittest.mock import AsyncMock, patch
from sigmaeval import (
    SigmaEval,
    ScenarioTest,
    MetricExpectation,
    assertions,
    metrics,
    AppResponse,
    ConversationRecord,
    ConversationTurn,
)
from datetime import datetime, timedelta

@pytest.fixture
def metric_scenario():
    """Fixture for a ScenarioTest with a MetricExpectation."""
    return ScenarioTest(
        title="Test Metric Scenario",
        given="A test user",
        when="The user does something",
        sample_size=10,
        then=MetricExpectation(
            metric=metrics.per_turn.response_latency,
            criteria=assertions.metrics.proportion_lt(
                threshold=1.0, proportion=0.9
            ),
        ),
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
        assert "p_value" in results.results
        assert results.results["observed_proportion"] == 0.98


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
        assert "p_value" in results.results
        assert results.results["observed_median"] == 0.5
