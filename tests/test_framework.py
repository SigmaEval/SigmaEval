"""Minimal tests for the SigmaEval framework."""

from typing import Any, Dict
import os
from dotenv import load_dotenv

import pytest

from sigmaeval import (
    SigmaEval,
    BehavioralTest,
    Expectation,
    AppResponse,
    SuccessRateEvaluator,
)
from tests.example_multi_turn_chat_app import LiteLLMChatApp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize(
    "valid_model",
    [
        "openai/gpt-4o",
        "anthropic/claude-3-opus",
        "local/test-model",
    ],
)
def test_sigmaeval_init_stores_model(valid_model: str) -> None:
    se = SigmaEval(model=valid_model)
    assert se.model == valid_model


@pytest.mark.parametrize("invalid_model", ["", "   "])
def test_sigmaeval_init_invalid_model_raises(invalid_model: str) -> None:
    with pytest.raises(ValueError):
        SigmaEval(model=invalid_model)


def test_sigmaeval_init_non_string_raises() -> None:
    with pytest.raises(ValueError):
        SigmaEval(model=None)  # type: ignore[arg-type]


@pytest.mark.integration
@pytest.mark.anyio
async def test_e2e_evaluation_with_example_app() -> None:
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
                sample_size=2,  # Keep sample size low for a quick test
            ),
        ),
    )

    # 3. Create the app handler to bridge SigmaEval and the target app
    chat_app = LiteLLMChatApp(model=app_model)

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
    sigma_eval = SigmaEval(model=eval_model)
    results = await sigma_eval.evaluate(scenario, app_handler)

    # 5. Assert the results to ensure the evaluation ran correctly
    assert isinstance(results, dict)
    assert results["test_config"]["title"] == scenario.title
    assert "rubric" in results and len(results["rubric"]) > 0
    assert "scores" in results and len(results["scores"]) == 2
    assert "conversations" in results and len(results["conversations"]) == 2

    # Check that a multi-turn conversation was recorded
    first_conversation = results["conversations"][0]
    assert len(first_conversation.turns) > 1
    assert first_conversation.turns[0]["role"] == "user"
    assert first_conversation.turns[1]["role"] == "assistant"


