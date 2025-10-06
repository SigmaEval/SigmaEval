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
    BehavioralExpectation,
    AppResponse,
)
from sigmaeval import assertions


# A simple scenario for testing data collection
test_scenario = ScenarioTest(
    title="Test Timestamp Recording",
    given="A user wants to check timestamps",
    when="An interaction occurs",
    sample_size=1,
    then=BehavioralExpectation(
        expected_behavior="Timestamps are recorded for each turn",
        criteria=assertions.scores.proportion_gte(min_score=1, proportion=1.0),
    ),
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
