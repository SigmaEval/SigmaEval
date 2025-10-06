import pytest
import numpy as np
from sigmaeval._evaluators import (
    MedianEvaluator,
    ProportionEvaluator,
)


@pytest.mark.parametrize(
    "scores, threshold, proportion, expected_pass, description",
    [
        (
            [9] * 35 + [7] * 5,
            8,
            0.75,
            True,
            "35/40 successes (87.5%) at 75% -> clear pass",
        ),
        (
            [9] * 34 + [7] * 6,
            8,
            0.75,
            False,
            "34/40 successes (85%) at 75% -> borderline fail",
        ),
        ([8] * 10 + [7] * 30, 8, 0.75, False, "10/40 successes (25%) -> clear fail"),
        ([10] * 50, 9, 0.99, False, "50/50 successes (100%) at 99% -> borderline fail"),
        ([10] * 50, 9, 0.95, False, "50/50 successes (100%) at 95% -> borderline fail"),
        ([10], 8, 0.9, False, "1/1 success (rating > 8) at 90% -> not enough evidence"),
        ([8], 8, 0.5, False, "1/1 success (rating > 8) at 50% -> not enough evidence"),
        ([10, 10], 1, 0.1, True, "2/2 success (rating > 1) at 10% -> clear pass"),
        (
            [10, 10],
            9,
            0.9,
            False,
            "2/2 successes (rating > 9) at 90% -> not enough evidence",
        ),
        ([10, 8], 9, 0.9, False, "1/2 successes (rating > 9) at 90% -> fail"),
        (
            [10] * 29 + [5] * 1,
            6,
            0.9,
            False,
            "29/30 successes (96.7%) at 90% -> Not enough to pass",
        ),
        ([10] * 30, 6, 0.9, True, "30/30 successes (100%) at 90% -> Clear pass"),
        (
            [10] * 28 + [5] * 2,
            6,
            0.9,
            False,
            "28/30 successes at 90% -> borderline fail",
        ),
        ([10] * 5 + [5] * 25, 6, 0.9, False, "5/30 successes -> clear fail"),
        (
            [6] * 15 + [5] * 15,
            6,
            0.5,
            False,
            "15/30 successes at 50% -> borderline fail",
        ),
        ([6] * 20 + [5] * 10, 6, 0.5, True, "20/30 successes at 50% -> clear pass"),
        ([10], 6, 0.9, False, "1/1 success at 90% -> not enough evidence"),
        ([10], 6, 0.5, False, "1/1 success at 50% -> not enough evidence"),
        ([5], 6, 0.5, False, "0/1 successes at 50% -> fail"),
        ([10] * 2, 6, 0.9, False, "2/2 successes at 90% -> not enough evidence"),
        ([10, 5], 6, 0.9, False, "1/2 successes at 90% -> fail"),
    ],
)
def test_proportion_evaluator_gte(
    scores, threshold, proportion, expected_pass, description
):
    """Tests the ProportionEvaluator with gte comparison."""
    evaluator = ProportionEvaluator(
        significance_level=0.05,
        threshold=threshold,
        proportion=proportion,
        comparison="gte",
    )
    results = evaluator.evaluate(scores)
    assert results["passed"] is expected_pass, f"Failed on: {description}"
    if expected_pass:
        assert results["p_value"] < 0.05, f"p-value should be < 0.05 for: {description}"
    else:
        assert results["p_value"] >= 0.05, f"p-value should be >= 0.05 for: {description}"


@pytest.mark.parametrize(
    "scores, threshold, expected_pass, description",
    [
        ([8, 9, 8, 10, 9, 9, 10, 8], 7.0, True, "Scores clearly above threshold"),
        ([1, 2, 1, 3, 2, 2, 1, 3], 4.0, False, "Scores clearly below threshold"),
        ([7, 8, 9, 8, 7, 9, 8, 8, 9, 7], 8.0, False, "Scores centered on threshold"),
        ([9] * 20, 8.0, True, "All scores are identical and above the minimum"),
        ([9] * 20, 9.0, False, "All scores are identical and equal to the minimum"),
        ([8, 9, 8, 10, 9], 7.0, True, "Small sample size, should pass"),
        ([8, 9, 7, 10], 8.5, False, "Small sample size, median is at threshold"),
        ([8], 7.0, True, "Single sample, should pass"),
        ([8], 8.0, False, "Single sample, should fail (equal to threshold)"),
    ],
)
def test_median_evaluator_gte(scores, threshold, expected_pass, description):
    """Tests the MedianEvaluator with gte comparison."""
    evaluator = MedianEvaluator(
        significance_level=0.05,
        threshold=threshold,
        comparison="gte",
        bootstrap_resamples=1000,
    )
    results = evaluator.evaluate(scores)
    assert results["passed"] is expected_pass, f"Failed on: {description}"
