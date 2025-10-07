"""
Tests for the data collection module.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from sigmaeval.core.data_collection import _run_single_interaction
from sigmaeval.core.models import (
    ScenarioTest,
    Expectation,
    AppResponse,
)
from sigmaeval import assertions


# A simple scenario for testing data collection
test_scenario = (
    ScenarioTest("Test Timestamp Recording")
    .given("A user wants to check timestamps")
    .when("An interaction occurs")
    .sample_size(1)
    .expect_behavior(
        "Timestamps are recorded for each turn",
        criteria=assertions.scores.proportion_gte(min_score=1, proportion=1.0),
    )
)


# A mock app handler that simulates some async work
async def mock_app_handler(message: str, state: dict) -> AppResponse:
    """A mock app handler that returns a simple response."""
    await asyncio.sleep(0.01)
    return AppResponse(
        response=f"App response to: {message}",
        state={"history": [message]}
    )


@pytest.mark.asyncio
async def test_timestamps_are_recorded_for_each_turn():
    """
    Verify that request and response timestamps are recorded for both
    the user simulator turn and the app handler turn.
    """
    # Mock the call to the external LLM to avoid actual API calls
    with patch(
        "sigmaeval.core.data_collection._litellm_acompletion",
        new_callable=AsyncMock
    ) as mock_litellm:
        # The mock needs to return a structure that mimics the real LiteLLM response
        mock_litellm.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"message": "Simulated user message", "continue": false}'
                    )
                )
            ]
        )

        # Run a single interaction to generate a conversation record
        conversation_record = await _run_single_interaction(
            scenario=test_scenario,
            app_handler=mock_app_handler,
            user_simulator_model="mock-model",
        )

    # --- Assertions ---
    assert len(conversation_record.turns) == 2, "Should have one user and one assistant turn"

    # 1. Check the user (simulator) turn
    user_turn = conversation_record.turns[0]
    assert user_turn.role == "user"
    assert user_turn.content == "Simulated user message"

    assert isinstance(user_turn.request_timestamp, datetime)
    assert isinstance(user_turn.response_timestamp, datetime)
    assert user_turn.response_timestamp >= user_turn.request_timestamp

    # 2. Check the assistant (app_handler) turn
    assistant_turn = conversation_record.turns[1]
    assert assistant_turn.role == "assistant"
    assert assistant_turn.content == "App response to: Simulated user message"

    assert isinstance(assistant_turn.request_timestamp, datetime)
    assert isinstance(assistant_turn.response_timestamp, datetime)
    assert assistant_turn.response_timestamp >= assistant_turn.request_timestamp


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._litellm_acompletion", new_callable=AsyncMock)
async def test_simulate_user_turn_success(mock_litellm):
    """Tests _simulate_user_turn successful response parsing."""
    from sigmaeval.core.data_collection import _simulate_user_turn

    mock_litellm.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"message": "Test message", "continue": true}'
                )
            )
        ]
    )
    message, should_continue, _, _ = await _simulate_user_turn(
        scenario=test_scenario,
        conversation_history=[],
        model="mock-model",
    )
    assert message == "Test message"
    assert should_continue is True


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._litellm_acompletion", new_callable=AsyncMock)
async def test_simulate_user_turn_json_error(mock_litellm):
    """Tests that _simulate_user_turn handles JSON parsing errors."""
    from sigmaeval.core.data_collection import _simulate_user_turn
    from sigmaeval.core.exceptions import LLMCommunicationError

    mock_litellm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="not json"))]
    )
    with pytest.raises(LLMCommunicationError):
        await _simulate_user_turn(
            scenario=test_scenario,
            conversation_history=[],
            model="mock-model",
        )


@pytest.mark.asyncio
async def test_simulate_user_turn_max_turns():
    """Tests that _simulate_user_turn stops at max_turns."""
    from sigmaeval.core.data_collection import _simulate_user_turn

    message, should_continue, _, _ = await _simulate_user_turn(
        scenario=test_scenario,
        conversation_history=[{"role": "user", "content": "..."}] * 10,
        model="mock-model",
        max_turns=10,
    )
    assert message.startswith("[Conversation ended")
    assert should_continue is False


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._litellm_acompletion", new_callable=AsyncMock)
async def test_judge_interaction_success(mock_litellm):
    """Tests _judge_interaction successful response parsing."""
    from sigmaeval.core.data_collection import _judge_interaction
    from sigmaeval.core.models import ConversationRecord

    mock_litellm.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"score": 8.5, "reasoning": "Good job"}'
                )
            )
        ]
    )
    score, reasoning = await _judge_interaction(
        scenario=test_scenario,
        expectation=test_scenario.then[0],
        conversation=ConversationRecord(),
        rubric="Test rubric",
        judge_model="mock-judge",
    )
    assert score == 8.5
    assert reasoning == "Good job"


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._litellm_acompletion", new_callable=AsyncMock)
async def test_judge_interaction_json_error(mock_litellm):
    """Tests that _judge_interaction handles JSON parsing errors."""
    from sigmaeval.core.data_collection import _judge_interaction
    from sigmaeval.core.models import ConversationRecord
    from sigmaeval.core.exceptions import LLMCommunicationError

    mock_litellm.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="not json"))]
    )
    with pytest.raises(LLMCommunicationError):
        await _judge_interaction(
            scenario=test_scenario,
            expectation=test_scenario.then[0],
            conversation=ConversationRecord(),
            rubric="Test rubric",
            judge_model="mock-judge",
        )


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._litellm_acompletion", new_callable=AsyncMock)
@pytest.mark.parametrize("input_score, expected_score", [(0, 1.0), (11, 10.0), (5, 5.0)])
async def test_judge_interaction_score_clamping(mock_litellm, input_score, expected_score):
    """Tests that scores from the judge are clamped to the 1-10 range."""
    from sigmaeval.core.data_collection import _judge_interaction
    from sigmaeval.core.models import ConversationRecord

    mock_litellm.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content=f'{{"score": {input_score}, "reasoning": "..."}}'
                )
            )
        ]
    )
    score, _ = await _judge_interaction(
        scenario=test_scenario,
        expectation=test_scenario.then[0],
        conversation=ConversationRecord(),
        rubric="Test rubric",
        judge_model="mock-judge",
    )
    assert score == expected_score


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._run_single_interaction", new_callable=AsyncMock)
async def test_collect_conversations(mock_run_single):
    """Tests that _collect_conversations calls _run_single_interaction the correct number of times."""
    from sigmaeval.core.data_collection import _collect_conversations
    from sigmaeval.core.models import ConversationRecord

    mock_run_single.return_value = ConversationRecord()
    sample_size = 5
    await _collect_conversations(
        scenario=test_scenario,
        app_handler=mock_app_handler,
        user_simulator_model="mock-model",
        sample_size=sample_size,
    )
    assert mock_run_single.call_count == sample_size


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._judge_interaction", new_callable=AsyncMock)
async def test_judge_conversations(mock_judge):
    """Tests that _judge_conversations calls _judge_interaction for each conversation."""
    from sigmaeval.core.data_collection import _judge_conversations
    from sigmaeval.core.models import ConversationRecord

    mock_judge.return_value = (8.0, "Good")
    conversations = [ConversationRecord() for _ in range(3)]
    await _judge_conversations(
        scenario=test_scenario,
        expectation=test_scenario.then[0],
        conversations=conversations,
        rubric="Test rubric",
        judge_model="mock-judge",
    )
    assert mock_judge.call_count == len(conversations)
