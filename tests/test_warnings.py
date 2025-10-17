"""Tests for basic framework functionality."""

import warnings
import pytest
from sigmaeval import SigmaEval, ScenarioTest, assertions
import os
from dotenv import load_dotenv


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_pydantic_warnings_emitted_during_evaluation():
    """
    Verifies that a basic SigmaEval run does not emit any Pydantic warnings.
    This is an integration test because the warnings are only triggered by
    real LLM calls via LiteLLM.
    """
    load_dotenv()
    eval_model = os.getenv("TEST_EVAL_MODEL")
    if not eval_model:
        pytest.skip("TEST_EVAL_MODEL env var must be set to run this test.")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Capture all warnings

        sigma_eval = SigmaEval(
            judge_model=eval_model,
            sample_size=1,
            significance_level=0.05,
        )

        scenario = (
            ScenarioTest("Minimal Warning Test")
            .given("A user")
            .when("The user says hi")
            .expect_behavior(
                "The bot says hi back.",
                criteria=assertions.scores.proportion_gte(min_score=1, proportion=0.1),
            )
        )

        async def app_handler(messages, state):
            return "hello"

        await sigma_eval.evaluate(scenario, app_handler)

        # Filter for the specific Pydantic UserWarning we want to suppress
        pydantic_warnings = [
            warn
            for warn in w
            if issubclass(warn.category, UserWarning) and "pydantic" in warn.filename
        ]

        # Assert that no such warnings were captured
        assert (
            len(pydantic_warnings) == 0
        ), f"Pydantic warnings were raised: {[str(warn.message) for warn in pydantic_warnings]}"
