"""Unit tests for the SigmaEval framework."""

import asyncio
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, patch

import pytest

from sigmaeval import AppResponse, ScenarioTest, SigmaEval, assertions


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._simulate_user_turn")
async def test_app_handler_receives_correct_history(mock_simulate_user_turn: AsyncMock):
    """
    Tests that the app_handler receives the correct conversation history.
    """
    # Arrange
    mock_simulate_user_turn.side_effect = [
        ("Hello", True, None, None),  # Turn 1
        ("Another message", False, None, None),  # Turn 2
    ]

    # This mock will run asserts for us
    async def mock_app_handler(
        messages: List[Dict[str, str]], state: Any
    ) -> AppResponse:
        # On first turn, history should just be the user's message
        if len(messages) == 1:
            assert messages == [{"role": "user", "content": "Hello"}]
        # On second turn, it should have the full history
        elif len(messages) == 3:
            assert messages == [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "Another message"},
            ]
        return AppResponse(response="Hi there!", state={})

    scenario = (
        ScenarioTest("Test History")
        .given("A user")
        .when("An action")
        .sample_size(1)
        .expect_behavior(
            "Something happens",
            criteria=assertions.scores.proportion_gte(min_score=1, proportion=0),
        )
    )
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    # Act
    # We patch the expensive parts and test the interaction loop
    with patch(
        "sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock
    ) as mock_judge, patch(
        "sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock
    ) as mock_rubric:
        mock_judge.return_value = ([10.0], ["reason"])
        mock_rubric.return_value = "Test Rubric"
        await sigma_eval.evaluate(scenario, mock_app_handler)


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._simulate_user_turn")
async def test_app_handler_state_is_passed_between_turns(
    mock_simulate_user_turn: AsyncMock,
):
    """
    Tests that the state object is correctly passed from one turn to the next.
    """
    # Arrange
    mock_simulate_user_turn.side_effect = [
        ("First message", True, None, None),
        ("Second message", False, None, None),
    ]
    app_handler_call_count = 0

    async def stateful_app_handler(
        messages: List[Dict[str, str]], state: Any
    ) -> Tuple[str, Any]:
        nonlocal app_handler_call_count
        app_handler_call_count += 1
        current_turn = state.get("turn", 0)

        if current_turn == 0:
            assert state == {}  # Starts empty
        else:
            assert state == {"turn": 1}  # Receives state from previous turn

        new_state = {"turn": current_turn + 1}
        return f"Response {current_turn}", new_state

    scenario = (
        ScenarioTest("Test State")
        .given("A user")
        .when("An action")
        .sample_size(1)
        .expect_behavior(
            "Something happens",
            criteria=assertions.scores.proportion_gte(min_score=1, proportion=0),
        )
    )
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    with patch(
        "sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock
    ) as mock_judge, patch(
        "sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock
    ) as mock_rubric:
        mock_judge.return_value = ([10.0], ["reason"])
        mock_rubric.return_value = "Test Rubric"
        await sigma_eval.evaluate(scenario, stateful_app_handler)

    assert app_handler_call_count == 2


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._simulate_user_turn")
async def test_app_handler_return_types(mock_simulate_user_turn: AsyncMock):
    """
    Tests that the framework correctly handles all valid app_handler return types.
    """
    # Arrange
    from datetime import datetime, timezone

    mock_simulate_user_turn.return_value = (
        "Hello",
        False,
        datetime.now(timezone.utc),
        datetime.now(timezone.utc),
    )
    scenario = (
        ScenarioTest("Test Return Types")
        .given("A user")
        .when("An action")
        .sample_size(1)
        .expect_behavior(
            "Something happens",
            criteria=assertions.scores.proportion_gte(min_score=1, proportion=0),
        )
    )
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    # Define handlers for each return type
    async def string_handler(messages, state):
        return "Stateless response"

    async def tuple_handler(messages, state):
        return "Stateful response", {"key": "value"}

    async def app_response_handler(messages, state):
        return AppResponse(response="Explicit response", state={"key": "value"})

    # Act & Assert
    with patch(
        "sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock
    ) as mock_judge, patch(
        "sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock
    ) as mock_rubric:
        mock_judge.return_value = ([10.0], ["reason"])
        mock_rubric.return_value = "Test Rubric"

        # Test string return
        results_str = await sigma_eval.evaluate(scenario, string_handler)
        assert (
            results_str.conversations[0].turns[0].app_response == "Stateless response"
        )

        # Test tuple return
        results_tuple = await sigma_eval.evaluate(scenario, tuple_handler)
        assert (
            results_tuple.conversations[0].turns[0].app_response == "Stateful response"
        )

        # Test AppResponse return
        results_app_resp = await sigma_eval.evaluate(scenario, app_response_handler)
        assert (
            results_app_resp.conversations[0].turns[0].app_response
            == "Explicit response"
        )


@pytest.mark.asyncio
@patch("sigmaeval.core.data_collection._simulate_user_turn")
async def test_app_handler_invalid_return_type_raises_error(
    mock_simulate_user_turn: AsyncMock,
):
    """
    Tests that a TypeError is raised if the app_handler returns an unsupported type.
    """
    # Arrange
    mock_simulate_user_turn.return_value = ("Hello", False, None, None)
    scenario = (
        ScenarioTest("Test Invalid Return")
        .given("A user")
        .when("An action")
        .sample_size(1)
        .expect_behavior(
            "Something happens",
            criteria=assertions.scores.proportion_gte(min_score=1, proportion=0),
        )
    )
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    async def invalid_handler(messages, state):
        return 123  # Invalid return type

    # Act & Assert
    with patch(
        "sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock
    ) as mock_judge, patch(
        "sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock
    ), pytest.raises(
        TypeError, match="app_handler returned an unsupported type"
    ):
        mock_judge.return_value = ([10.0], ["reason"])
        await sigma_eval.evaluate(scenario, invalid_handler)
