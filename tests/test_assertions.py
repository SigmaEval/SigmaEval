import pytest
from unittest.mock import AsyncMock

from sigmaeval import (
    ScenarioTest,
    BehavioralExpectation,
    assertions,
    AppResponse,
)
from sigmaeval.assertions import ProportionAssertion, MedianAssertion


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
            criteria=assertions.scores.proportion_gte(
                min_score=8, proportion=0.9
            ),
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
    assert isinstance(crit, ProportionAssertion)
    assert crit.threshold == 8
    assert crit.proportion == 0.9
    assert crit.comparison == "gte"
    assert crit.significance_level == 0.01


def test_median_gte_returns_correct_dataclass():
    """
    Tests that assertions.scores.median_gte creates the correct dataclass.
    """
    crit = assertions.scores.median_gte(threshold=7.5, significance_level=0.05)
    assert isinstance(crit, MedianAssertion)
    assert crit.threshold == 7.5
    assert crit.comparison == "gte"
    assert crit.significance_level == 0.05


@pytest.mark.parametrize(
    "invalid_params, error_match",
    [
        ({"min_score": 8, "proportion": -0.1}, "proportion must be between 0 and 1"),
        ({"min_score": 8, "proportion": 1.1}, "proportion must be between 0 and 1"),
        ({"min_score": 0, "proportion": 0.9}, "min_score must be between 1 and 10"),
        ({"min_score": 11, "proportion": 0.9}, "min_score must be between 1 and 10"),
        (
            {"min_score": 8, "proportion": 0.9, "significance_level": -0.1},
            "significance_level must be between 0 and 1",
        ),
        (
            {"min_score": 8, "proportion": 0.9, "significance_level": 1.1},
            "significance_level must be between 0 and 1",
        ),
    ],
)
def test_proportion_gte_invalid_params_raise(invalid_params, error_match):
    """
    Tests that creating a proportion_gte assertion with invalid parameters
    raises a ValueError.
    """
    with pytest.raises(ValueError, match=error_match):
        assertions.scores.proportion_gte(**invalid_params)


@pytest.mark.parametrize(
    "invalid_params, error_match",
    [
        ({"threshold": 0.5}, "threshold must be between 1 and 10"),
        ({"threshold": 11.5}, "threshold must be between 1 and 10"),
        (
            {"threshold": 8.0, "significance_level": -0.05},
            "significance_level must be between 0 and 1",
        ),
        (
            {"threshold": 8.0, "significance_level": 1.05},
            "significance_level must be between 0 and 1",
        ),
    ],
)
def test_median_gte_invalid_params_raise(invalid_params, error_match):
    """
    Tests that creating a median_gte assertion with invalid parameters raises
    a ValueError.
    """
    with pytest.raises(ValueError, match=error_match):
        assertions.scores.median_gte(**invalid_params)
