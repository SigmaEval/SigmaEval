import pytest
from unittest.mock import AsyncMock, patch

from sigmaeval import (
    SigmaEval,
    ScenarioTest,
    BehavioralExpectation,
    assertions,
    AppResponse,
)
from sigmaeval.assertions import ProportionGTE, MedianGTE


@pytest.fixture
def basic_scenario():
    """Fixture for a basic ScenarioTest."""
    return ScenarioTest(
        title="Test Scenario",
        given="A test user",
        when="The user does something",
        sample_size=2,
        then=BehavioralExpectation(
            expected_behavior="The app should respond appropriately",
            criteria="DUMMY",  # Will be replaced in each test
        ),
    )


@pytest.fixture
def mock_app_handler():
    """Fixture for a mock async app_handler that does nothing."""
    return AsyncMock(return_value=AppResponse(response="Test response", state={}))


def test_proportion_gte_returns_correct_dataclass():
    """
    Tests that assertions.scores.proportion_gte creates the correct dataclass.
    """
    crit = assertions.scores.proportion_gte(
        min_score=8, proportion=0.9, significance_level=0.01
    )
    assert isinstance(crit, ProportionGTE)
    assert crit.min_score == 8
    assert crit.proportion == 0.9
    assert crit.significance_level == 0.01


def test_median_gte_returns_correct_dataclass():
    """
    Tests that assertions.scores.median_gte creates the correct dataclass.
    """
    crit = assertions.scores.median_gte(threshold=7.5, significance_level=0.05)
    assert isinstance(crit, MedianGTE)
    assert crit.threshold == 7.5
    assert crit.significance_level == 0.05


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_criteria, error_match",
    [
        (
            assertions.scores.proportion_gte(min_score=8, proportion=-0.1),
            "min_proportion must be between 0 and 1",
        ),
        (
            assertions.scores.proportion_gte(min_score=8, proportion=1.1),
            "min_proportion must be between 0 and 1",
        ),
        (
            assertions.scores.proportion_gte(min_score=0, proportion=0.9),
            "min_rating must be between 1 and 10",
        ),
        (
            assertions.scores.proportion_gte(min_score=11, proportion=0.9),
            "min_rating must be between 1 and 10",
        ),
        (
            assertions.scores.proportion_gte(
                min_score=8, proportion=0.9, significance_level=-0.1
            ),
            "significance_level must be between 0 and 1",
        ),
        (
            assertions.scores.proportion_gte(
                min_score=8, proportion=0.9, significance_level=1.1
            ),
            "significance_level must be between 0 and 1",
        ),
    ],
)
@patch("sigmaeval.core.framework.collect_evaluation_data", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
async def test_proportion_gte_invalid_params_raise_in_evaluate(
    mock_generate_rubric,
    mock_collect_data,
    invalid_criteria,
    error_match,
    basic_scenario,
    mock_app_handler,
):
    """
    Tests that using an assertion with invalid parameters raises a ValueError
    during the evaluation process.
    """
    mock_generate_rubric.return_value = "Mocked rubric"
    mock_collect_data.return_value = ([10.0], ["reason"], [])  # Need non-empty scores

    basic_scenario.then.criteria = invalid_criteria
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    with pytest.raises(ValueError, match=error_match):
        await sigma_eval.evaluate(basic_scenario, mock_app_handler)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_criteria, error_match",
    [
        (
            assertions.scores.median_gte(threshold=0.5),
            "min_median_rating must be between 1 and 10",
        ),
        (
            assertions.scores.median_gte(threshold=11.5),
            "min_median_rating must be between 1 and 10",
        ),
        (
            assertions.scores.median_gte(threshold=8.0, significance_level=-0.05),
            "significance_level must be between 0 and 1",
        ),
        (
            assertions.scores.median_gte(threshold=8.0, significance_level=1.05),
            "significance_level must be between 0 and 1",
        ),
    ],
)
@patch("sigmaeval.core.framework.collect_evaluation_data", new_callable=AsyncMock)
@patch("sigmaeval.core.framework._generate_rubric", new_callable=AsyncMock)
async def test_median_gte_invalid_params_raise_in_evaluate(
    mock_generate_rubric,
    mock_collect_data,
    invalid_criteria,
    error_match,
    basic_scenario,
    mock_app_handler,
):
    """
    Tests that using a median_gte assertion with invalid parameters raises a
    ValueError during the evaluation process.
    """
    mock_generate_rubric.return_value = "Mocked rubric"
    mock_collect_data.return_value = ([10.0], ["reason"], [])  # Need non-empty scores

    basic_scenario.then.criteria = invalid_criteria
    sigma_eval = SigmaEval(judge_model="test/model", significance_level=0.05)

    with pytest.raises(ValueError, match=error_match):
        await sigma_eval.evaluate(basic_scenario, mock_app_handler)
