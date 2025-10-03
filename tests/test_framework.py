"""Minimal tests for the SigmaEval framework."""

from typing import Any, Dict

import pytest

from sigmaeval import SigmaEval, BehavioralTest, Expectation, AppResponse


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


@pytest.mark.anyio
async def test_evaluate_returns_dict() -> None:
    """Test that evaluate returns a dict structure (uses actual API calls with small sample)."""
    from sigmaeval import SuccessRateEvaluator
    
    se = SigmaEval(model="openai/gpt-4o")

    scenario = BehavioralTest(
        title="dummy",
        given="given",
        when="when",
        then=Expectation(
            expected_behavior="do x", 
            evaluator=SuccessRateEvaluator(
                significance_level=0.05,
                min_proportion=0.5,
                sample_size=2  # Small sample for faster testing
            )
        ),
    )

    async def app_handler(message: str, state: Dict[str, Any]) -> AppResponse:
        return AppResponse(response="ok", state=state or {})

    result = await se.evaluate(scenario, app_handler)
    assert isinstance(result, dict)
    assert "scores" in result
    assert "rubric" in result
    assert "model" in result


