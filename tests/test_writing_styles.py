"""
Tests for the user simulation writing style variation feature.
"""

import pytest
from unittest.mock import AsyncMock, patch

from sigmaeval.core.models import (
    AppResponse,
    ScenarioTest,
    Expectation,
    WritingStyleConfig,
    WritingStyleAxes,
    ConversationRecord,
    ConversationTurn,
)
from sigmaeval import SigmaEval, assertions
from sigmaeval.core.writing_styles import _generate_writing_style


def test_generate_writing_style_with_custom_axes():
    """
    Tests that _generate_writing_style correctly uses custom axes.
    """
    custom_axes = WritingStyleAxes(
        proficiency=["a"], tone=["b"], verbosity=["c"], formality=["d"]
    )
    style = _generate_writing_style(axes=custom_axes)
    assert style["Proficiency"] == "a"
    assert style["Tone"] == "b"
    assert style["Verbosity"] == "c"
    assert style["Formality"] == "d"


def test_generate_writing_style_with_default_axes():
    """
    Tests that _generate_writing_style runs with default axes when none are provided.
    """
    style = _generate_writing_style()
    assert isinstance(style, dict)
    assert "Proficiency" in style
    assert "Tone" in style
    assert "Verbosity" in style
    assert "Formality" in style


@pytest.fixture
def mock_app_handler():
    """Fixture for a mock async app_handler."""
    return AsyncMock(return_value=AppResponse(response="Test response", state={}))


@pytest.fixture
def basic_scenario():
    """Fixture for a basic ScenarioTest scenario."""
    return (
        ScenarioTest("Test Scenario")
        .given("A test user")
        .when("The user does something")
        .sample_size(2)
        .expect_behavior(
            "The app should respond appropriately",
            criteria=assertions.scores.proportion_gte(
                min_score=6, proportion=0.8, significance_level=0.05
            ),
        )
    )


@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_writing_styles_enabled_by_default(
    mock_collect_conversations,
    mock_judge_conversations,
    mock_generate_rubric,
    basic_scenario,
    mock_app_handler,
):
    """
    Verify that writing style variations are enabled by default.
    """
    mock_generate_rubric.return_value = "Test Rubric"

    # Mock the return value to include valid ConversationRecord objects
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    mock_conversations = [
        ConversationRecord(
            turns=[
                ConversationTurn(
                    role="user",
                    content="Test",
                    request_timestamp=now,
                    response_timestamp=now,
                )
            ]
        ),
        ConversationRecord(
            turns=[
                ConversationTurn(
                    role="user",
                    content="Test",
                    request_timestamp=now,
                    response_timestamp=now,
                )
            ]
        ),
    ]
    mock_collect_conversations.return_value = mock_conversations
    mock_judge_conversations.return_value = ([8.0, 9.0], ["reason", "reason"])

    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)
    await sigma_eval.evaluate(basic_scenario, mock_app_handler)

    mock_collect_conversations.assert_called_once()
    call_args, call_kwargs = mock_collect_conversations.call_args
    writing_style_config = call_kwargs.get("writing_style_config")

    assert isinstance(writing_style_config, WritingStyleConfig)
    assert writing_style_config.enabled is True


@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._judge_conversations", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._collect_conversations", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_writing_styles_can_be_disabled(
    mock_collect_conversations,
    mock_judge_conversations,
    mock_generate_rubric,
    basic_scenario,
    mock_app_handler,
):
    """
    Verify that writing style variations can be disabled.
    """
    mock_generate_rubric.return_value = "Test Rubric"

    # Mock the return value to include valid ConversationRecord objects
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    mock_conversations = [
        ConversationRecord(
            turns=[
                ConversationTurn(
                    role="user",
                    content="Test",
                    request_timestamp=now,
                    response_timestamp=now,
                )
            ]
        ),
        ConversationRecord(
            turns=[
                ConversationTurn(
                    role="user",
                    content="Test",
                    request_timestamp=now,
                    response_timestamp=now,
                )
            ]
        ),
    ]
    mock_collect_conversations.return_value = mock_conversations
    mock_judge_conversations.return_value = ([8.0, 9.0], ["reason", "reason"])

    config = WritingStyleConfig(enabled=False)
    sigma_eval = SigmaEval(
        judge_model="test/model",
        writing_style_config=config,
        significance_level=0.05,
    )
    await sigma_eval.evaluate(basic_scenario, mock_app_handler)

    mock_collect_conversations.assert_called_once()
    call_args, call_kwargs = mock_collect_conversations.call_args
    writing_style_config = call_kwargs.get("writing_style_config")

    assert isinstance(writing_style_config, WritingStyleConfig)
    assert writing_style_config.enabled is False


@patch("sigmaeval.core.data_collection._generate_writing_style")
@patch("sigmaeval.core.data_collection._litellm_acompletion", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_custom_writing_style_axes_are_used(
    mock_generate_rubric,
    mock_acompletion,
    mock_generate_style,
    basic_scenario,
    mock_app_handler,
):
    """
    Verify that custom writing style axes are correctly passed to the generator.
    """
    # Mock return values
    mock_generate_rubric.return_value = "Test Rubric"
    # User sim response, then judge response
    mock_acompletion.side_effect = [
        # User sim (sample 1)
        AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content='{"message": "Hi", "continue": false}')
                )
            ]
        ),
        # User sim (sample 2)
        AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(
                        content='{"message": "Hi again", "continue": false}'
                    )
                )
            ]
        ),
        # Judge (sample 1)
        AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content='{"score": 8, "reasoning": "Good"}')
                )
            ]
        ),
        # Judge (sample 2)
        AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content='{"score": 9, "reasoning": "Great"}')
                )
            ]
        ),
    ]
    mock_generate_style.return_value = {
        "Proficiency": "Generated style proficiency",
        "Tone": "Generated style tone",
    }

    custom_axes = WritingStyleAxes(
        proficiency=["perfect"], tone=["happy"], verbosity=["short"], formality=["casual"]
    )
    config = WritingStyleConfig(axes=custom_axes)

    sigma_eval = SigmaEval(
        judge_model="test/model",
        writing_style_config=config,
        significance_level=0.05,
    )
    await sigma_eval.evaluate(basic_scenario, mock_app_handler)

    assert mock_generate_style.call_count == basic_scenario.num_samples
    # All calls should have used the same custom axes
    for call in mock_generate_style.call_args_list:
        _, call_kwargs = call
        assert call_kwargs.get("axes") == custom_axes


@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
@patch("sigmaeval.core.data_collection._generate_writing_style")
@patch("sigmaeval.core.data_collection._litellm_acompletion", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_writing_style_is_in_prompt(
    mock_acompletion,
    mock_generate_style,
    mock_generate_rubric,
    basic_scenario,
    mock_app_handler,
):
    """
    Verify that the generated writing style is included in the user simulator prompt.
    """
    mock_generate_rubric.return_value = "Test Rubric"
    mock_acompletion.side_effect = [
        # User sim 1
        AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content='{"message": "Stop", "continue": false}')
                )
            ]
        ),
        # User sim 2
        AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content='{"message": "Stop", "continue": false}')
                )
            ]
        ),
        # Judge 1
        AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content='{"score": 8, "reasoning": "Good"}')
                )
            ]
        ),
        # Judge 2
        AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(content='{"score": 9, "reasoning": "Great"}')
                )
            ]
        ),
    ]

    # Use a very specific, unique value to make it easy to find in the prompt
    style_dict = {
        "Proficiency": "MyUniqueProficiency",
        "Tone": "MyUniqueTone",
        "Verbosity": "MyUniqueVerbosity",
        "Formality": "MyUniqueFormality",
    }
    mock_generate_style.return_value = style_dict

    config = WritingStyleConfig()  # Use default axes, but mock the output
    sigma_eval = SigmaEval(
        judge_model="test/model",
        writing_style_config=config,
        significance_level=0.05,
    )

    await sigma_eval.evaluate(basic_scenario, mock_app_handler)

    # The user simulator is the first call for each sample
    user_sim_calls = [
        c
        for c in mock_acompletion.call_args_list
        if "You are simulating a user" in c.kwargs["messages"][0]["content"]
    ]
    assert len(user_sim_calls) == 2

    for call in user_sim_calls:
        prompt = call.kwargs["messages"][1]["content"]
        assert "- Adopt the following writing style" in prompt
        assert "    - Proficiency: MyUniqueProficiency" in prompt
        assert "    - Tone: MyUniqueTone" in prompt
        assert "    - Verbosity: MyUniqueVerbosity" in prompt
        assert "    - Formality: MyUniqueFormality" in prompt
