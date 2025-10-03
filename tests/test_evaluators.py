import pytest
from pydantic import ValidationError
from sigmaeval.evaluators import (
    SuccessRateEvaluator,
    RatingMeanEvaluator,
    RatingProportionEvaluator,
)


def test_success_rate_evaluator_requires_significance_level():
    with pytest.raises(ValidationError) as excinfo:
        SuccessRateEvaluator(min_proportion=0.9, sample_size=30)
    assert "significance_level" in str(excinfo.value)


def test_success_rate_evaluator_requires_sample_size():
    with pytest.raises(ValidationError) as excinfo:
        SuccessRateEvaluator(significance_level=0.05, min_proportion=0.9)
    assert "sample_size" in str(excinfo.value)


def test_rating_mean_evaluator_requires_significance_level():
    with pytest.raises(ValidationError) as excinfo:
        RatingMeanEvaluator(min_mean_rating=7.0, sample_size=50)
    assert "significance_level" in str(excinfo.value)


def test_rating_mean_evaluator_requires_sample_size():
    with pytest.raises(ValidationError) as excinfo:
        RatingMeanEvaluator(significance_level=0.05, min_mean_rating=7.0)
    assert "sample_size" in str(excinfo.value)


def test_rating_proportion_evaluator_requires_significance_level():
    with pytest.raises(ValidationError) as excinfo:
        RatingProportionEvaluator(min_rating=8, min_proportion=0.75, sample_size=50)
    assert "significance_level" in str(excinfo.value)


def test_rating_proportion_evaluator_requires_sample_size():
    with pytest.raises(ValidationError) as excinfo:
        RatingProportionEvaluator(
            significance_level=0.05, min_rating=8, min_proportion=0.75
        )
    assert "sample_size" in str(excinfo.value)


@pytest.mark.parametrize(
    "evaluator, kwargs",
    [
        (SuccessRateEvaluator, {"significance_level": 1.5, "min_proportion": 0.9, "sample_size": 30}),
        (SuccessRateEvaluator, {"significance_level": -0.5, "min_proportion": 0.9, "sample_size": 30}),
        (SuccessRateEvaluator, {"significance_level": 0.05, "min_proportion": 1.1, "sample_size": 30}),
        (SuccessRateEvaluator, {"significance_level": 0.05, "min_proportion": -0.1, "sample_size": 30}),
        (SuccessRateEvaluator, {"significance_level": 0.05, "min_proportion": 0.9, "sample_size": 0}),
        (SuccessRateEvaluator, {"significance_level": 0.05, "min_proportion": 0.9, "sample_size": -10}),
        (RatingMeanEvaluator, {"significance_level": 1.5, "min_mean_rating": 7.0, "sample_size": 50}),
        (RatingMeanEvaluator, {"significance_level": -0.5, "min_mean_rating": 7.0, "sample_size": 50}),
        (RatingMeanEvaluator, {"significance_level": 0.05, "min_mean_rating": 0.0, "sample_size": 50}),
        (RatingMeanEvaluator, {"significance_level": 0.05, "min_mean_rating": 11.0, "sample_size": 50}),
        (RatingMeanEvaluator, {"significance_level": 0.05, "min_mean_rating": 7.0, "sample_size": 0}),
        (RatingMeanEvaluator, {"significance_level": 0.05, "min_mean_rating": 7.0, "sample_size": -50}),
        (RatingProportionEvaluator, {"significance_level": 1.5, "min_rating": 8, "min_proportion": 0.75, "sample_size": 50}),
        (RatingProportionEvaluator, {"significance_level": -0.5, "min_rating": 8, "min_proportion": 0.75, "sample_size": 50}),
        (RatingProportionEvaluator, {"significance_level": 0.05, "min_rating": 0, "min_proportion": 0.75, "sample_size": 50}),
        (RatingProportionEvaluator, {"significance_level": 0.05, "min_rating": 11, "min_proportion": 0.75, "sample_size": 50}),
        (RatingProportionEvaluator, {"significance_level": 0.05, "min_rating": 8, "min_proportion": 1.1, "sample_size": 50}),
        (RatingProportionEvaluator, {"significance_level": 0.05, "min_rating": 8, "min_proportion": -0.1, "sample_size": 50}),
        (RatingProportionEvaluator, {"significance_level": 0.05, "min_rating": 8, "min_proportion": 0.75, "sample_size": 0}),
        (RatingProportionEvaluator, {"significance_level": 0.05, "min_rating": 8, "min_proportion": 0.75, "sample_size": -50}),
    ],
)
def test_evaluator_value_validation(evaluator, kwargs):
    with pytest.raises(ValidationError):
        evaluator(**kwargs)
