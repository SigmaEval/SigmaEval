"""
Statistical evaluators for SigmaEval framework.
"""

import logging
from pydantic import BaseModel
from typing import List, Literal
import numpy as np
from scipy.stats import binomtest

logger = logging.getLogger("sigmaeval")


class ProportionEvaluator(BaseModel):
    """
    Tests if the proportion of values meeting a threshold satisfies a statistical
    comparison.
    """

    significance_level: float
    threshold: float
    proportion: float
    comparison: Literal["gte", "lte"]

    def evaluate(self, values: List[float], label: str | None = None) -> dict:
        """
        Evaluate if the proportion of values meeting the threshold is sufficient.
        """
        if self.comparison == "gte":
            # Success is defined as a value GREATER THAN OR EQUAL TO the threshold
            successes = sum(1 for v in values if v >= self.threshold)
        else:  # lte
            # Success is defined as a value LESS THAN the threshold
            successes = sum(1 for v in values if v < self.threshold)

        sample_size = len(values)

        # In both 'gte' and 'lte' scenarios, we are testing if the proportion of
        # "successes" (defined differently for each case) is statistically
        # GREATER than the specified minimum proportion.
        result = binomtest(k=successes, n=sample_size, p=self.proportion, alternative="greater")
        p_value = result.pvalue
        passed = p_value < self.significance_level

        results = {
            "passed": bool(passed),
            "p_value": float(p_value),
            "observed_proportion": successes / sample_size if sample_size > 0 else 0,
        }

        if label:
            results = {f"{label}: {k.replace('_', ' ').title()}": v for k, v in results.items()}
            results["passed"] = bool(passed)

        return results


class MedianEvaluator(BaseModel):
    """
    Performs a one-sided bootstrap hypothesis test for the median of a set of
    values.
    """

    significance_level: float
    threshold: float
    comparison: Literal["gte", "lte"]
    bootstrap_resamples: int = 10000

    def evaluate(self, values: List[float], label: str | None = None) -> dict:
        """
        Evaluate if the median of the values is statistically less than the threshold.
        """
        sample_size = len(values)
        if sample_size == 0:
            return {"passed": False, "error": "Cannot evaluate empty list of values."}

        bootstrap_medians = np.array(
            [
                np.median(np.random.choice(values, size=sample_size, replace=True))
                for _ in range(self.bootstrap_resamples)
            ]
        )

        if self.comparison == "gte":
            # H0: median <= threshold
            # H1: median > threshold
            p_value = np.mean(bootstrap_medians <= self.threshold)
        else:  # lte
            # H0: median >= threshold
            # H1: median < threshold
            p_value = np.mean(bootstrap_medians >= self.threshold)

        passed = p_value < self.significance_level

        results = {
            "passed": bool(passed),
            "p_value": float(p_value),
            "observed_median": float(np.median(values)),
            "observed_mean": float(np.mean(values)),
        }

        if label:
            results = {f"{label}: {k.replace('_', ' ').title()}": v for k, v in results.items()}
            results["passed"] = bool(passed)

        return results
