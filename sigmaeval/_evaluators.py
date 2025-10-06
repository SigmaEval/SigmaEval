"""
Statistical evaluators for SigmaEval framework.
"""
import logging
from pydantic import BaseModel, Field, field_validator
from typing import List
import numpy as np
from scipy.stats import binomtest

logger = logging.getLogger("sigmaeval")


class MetricProportionEvaluator(BaseModel):
    """
    Tests if the proportion of measured metric values is below a threshold.
    """
    significance_level: float
    threshold: float
    proportion: float

    def evaluate(self, values: List[float], label: str | None = None) -> dict:
        """
        Evaluate if the proportion of values meeting the threshold is sufficient.
        """
        # Success is defined as a value LESS THAN the threshold
        successes = sum(1 for v in values if v < self.threshold)
        sample_size = len(values)
        
        # H0: p <= proportion (the true proportion of successes is less than or equal to the desired proportion)
        # H1: p > proportion (the true proportion of successes is greater than the desired proportion)
        result = binomtest(
            k=successes, n=sample_size, p=self.proportion, alternative="greater"
        )
        p_value = result.pvalue
        passed = p_value < self.significance_level

        results = {
            "passed": bool(passed),
            "p_value": float(p_value),
            "observed_proportion": successes / sample_size,
        }

        if label:
            results = {
                f"{label}: {k.replace('_', ' ').title()}": v for k, v in results.items()
            }
            results["passed"] = passed
        
        return results


class MetricMedianEvaluator(BaseModel):
    """
    Performs a one-sided bootstrap hypothesis test for the median of a metric.

    This is a non-parametric test that is robust to the underlying data
    distribution and is not sensitive to outliers. It is recommended for
    metrics like latency or turn count that may be skewed.

    Attributes:
        significance_level: Significance level for hypothesis test (e.g., 0.05)
        threshold: The value that the median should be less than.
        bootstrap_resamples: Number of bootstrap resamples to perform (default: 10000).
    """

    significance_level: float
    threshold: float
    bootstrap_resamples: int = 10000

    def evaluate(self, values: List[float], label: str | None = None) -> dict:
        """
        Evaluate if the median of the values is statistically less than the threshold.
        """
        sample_size = len(values)
        if sample_size == 0:
            return {"passed": False, "error": "Cannot evaluate empty list of values."}

        # H0: median >= threshold
        # H1: median < threshold

        bootstrap_medians = np.array(
            [
                np.median(np.random.choice(values, size=sample_size, replace=True))
                for _ in range(self.bootstrap_resamples)
            ]
        )

        # The p-value is the proportion of bootstrap medians that are more
        # extreme than the null hypothesis (i.e., >= threshold).
        p_value = np.mean(bootstrap_medians >= self.threshold)

        passed = p_value < self.significance_level

        results = {
            "passed": bool(passed),
            "p_value": float(p_value),
            "observed_median": float(np.median(values)),
            "observed_mean": float(np.mean(values)),
        }

        if label:
            results = {
                f"{label}: {k.replace('_', ' ').title()}": v for k, v in results.items()
            }
            results["passed"] = passed

        return results


class RatingMedianEvaluator(BaseModel):
    """
    Performs a one-sided bootstrap hypothesis test for the median rating.

    This method is non-parametric and does not assume the data is normally
    distributed or symmetric, making it robust for skewed data like 1-10 scores.
    It resamples the collected scores thousands of times to build an empirical
    distribution of the median, then calculates a confidence interval to
    determine if the true median is statistically greater than a baseline.

    Useful for subjective qualities like helpfulness or tone.

    Attributes:
        significance_level: Significance level for hypothesis test (e.g., 0.05)
        min_median_rating: Minimum acceptable median rating (e.g., 7.0)
        bootstrap_resamples: Number of bootstrap resamples to perform (default: 10000).
    """

    significance_level: float = Field(
        ...,
        description="The probability of incorrectly rejecting the null hypothesis (a 'false positive'). Common values are 0.05 (5%) or 0.01 (1%). A value of 0.05 means you accept a 5% chance of concluding the system's median rating exceeds the minimum when it actually doesn't.",
    )
    min_median_rating: float = Field(..., description="Minimum median rating threshold")
    bootstrap_resamples: int = Field(
        10000, description="Number of bootstrap resamples to perform"
    )

    @field_validator("significance_level")
    def validate_significance_level(cls, v):
        if not (0 < v < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return v

    @field_validator("min_median_rating")
    def validate_min_median_rating(cls, v):
        if not (1 <= v <= 10):
            raise ValueError("min_median_rating must be between 1 and 10")
        return v

    @field_validator("bootstrap_resamples")
    def validate_bootstrap_resamples(cls, v):
        if v <= 0:
            raise ValueError("bootstrap_resamples must be a positive integer")
        return v

    def evaluate(self, scores: List[float], label: str | None = None) -> dict:
        """
        Evaluate if the median rating meets the minimum threshold using bootstrapping.

        Args:
            scores: List of scores (1-10) from Judge LLM

        Returns:
            Dictionary with evaluation results including pass/fail and statistics
        """
        log_prefix = f"RatingMedianEvaluator ('{label}')" if label else "RatingMedianEvaluator"
        logger.debug(f"{log_prefix} received scores: {scores}")
        sample_size = len(scores)

        # H0: median <= min_median_rating
        # H1: median > min_median_rating

        # Generate bootstrap samples and calculate their medians
        bootstrap_medians = np.array(
            [
                np.median(np.random.choice(scores, size=sample_size, replace=True))
                for _ in range(self.bootstrap_resamples)
            ]
        )

        # The p-value in a bootstrap test is the proportion of bootstrap statistics
        # that are more extreme than what was observed under the null hypothesis.
        # Here, we check how many bootstrap medians are less than or equal to the
        # min_median_rating.
        p_value = np.mean(bootstrap_medians <= self.min_median_rating)

        # To pass, the p-value must be less than the significance level.
        passed = p_value < self.significance_level

        # Also calculate the lower bound of the confidence interval for reporting
        confidence_interval_lower_bound = np.percentile(
            bootstrap_medians, self.significance_level * 100
        )

        results = {
            "passed": bool(passed),
            "p_value": float(p_value),
            "confidence_interval_lower_bound": float(confidence_interval_lower_bound),
            "significance_level": self.significance_level,
            "min_median_rating": self.min_median_rating,
            "observed_median": float(np.median(scores)),
            "observed_mean": float(np.mean(scores)),
            "sample_size": sample_size,
            "bootstrap_resamples": self.bootstrap_resamples,
        }
        logger.info(
            f"{log_prefix} results: passed={bool(passed)}, p_value={p_value:.4f}, observed_median={np.median(scores):.2f}, significance_level={self.significance_level}"
        )
        logger.debug(f"Full {log_prefix} results: {results}")
        if label:
            results = {
                f"{label}: {k.replace('_', ' ').title()}": v for k, v in results.items()
            }
            # Also include the raw "passed" key for the framework's final status check
            results["passed"] = passed

        logger.info(
            f"{log_prefix} results: passed={bool(passed)}, p_value={p_value:.4f}, observed_median={np.median(scores):.2f}, significance_level={self.significance_level}"
        )
        return results


class RatingProportionEvaluator(BaseModel):
    """
    Tests if proportion of ratings at or above threshold exceeds minimum.
    
    Similar to SuccessRateEvaluator but with configurable rating threshold.
    
    Attributes:
        significance_level: Significance level for hypothesis test (e.g., 0.05)
        min_rating: Minimum acceptable rating on 1-10 scale (e.g., 8)
        min_proportion: Minimum proportion of responses meeting min_rating (e.g., 0.75)
    """
    significance_level: float = Field(
        ...,
        description="The probability of incorrectly rejecting the null hypothesis (a 'false positive'). Common values are 0.05 (5%) or 0.01 (1%). A value of 0.05 means you accept a 5% chance of concluding the system meets the minimum proportion when it actually doesn't."
    )
    min_rating: int = Field(..., description="Minimum acceptable rating (1-10)")
    min_proportion: float = Field(..., description="Minimum proportion at or above min_rating")

    @field_validator("significance_level")
    def validate_significance_level(cls, v):
        if not (0 < v < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return v

    @field_validator("min_rating")
    def validate_min_rating(cls, v):
        if not (1 <= v <= 10):
            raise ValueError("min_rating must be between 1 and 10")
        return v

    @field_validator("min_proportion")
    def validate_min_proportion(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("min_proportion must be between 0 and 1")
        return v

    def evaluate(self, scores: List[float], label: str | None = None) -> dict:
        """
        Evaluate if the proportion of ratings meeting threshold is sufficient.
        
        Args:
            scores: List of scores (1-10) from Judge LLM
            
        Returns:
            Dictionary with evaluation results including pass/fail and statistics
        """
        log_prefix = f"RatingProportionEvaluator ('{label}')" if label else "RatingProportionEvaluator"
        logger.debug(f"{log_prefix} received scores: {scores}")
        sample_size = len(scores)

        successes = sum(1 for score in scores if score >= self.min_rating)

        # H0: p <= min_proportion
        # H1: p > min_proportion
        # Using exact binomial test for accuracy with all sample sizes.
        # Note: With very small sample sizes (e.g., n<5), it can be
        # statistically impossible to achieve a p-value < 0.05, even with
        # a 100% success rate. This is expected behavior, as there isn't
        # enough evidence to reject the null hypothesis with confidence.
        result = binomtest(
            k=successes, n=sample_size, p=self.min_proportion, alternative="greater"
        )
        p_value = result.pvalue

        passed = p_value < self.significance_level

        results = {
            "passed": bool(passed),
            "p_value": float(p_value),
            "significance_level": self.significance_level,
            "min_rating": self.min_rating,
            "min_proportion": self.min_proportion,
            "observed_proportion": successes / sample_size,
            "successes": successes,
            "sample_size": sample_size,
        }
        logger.info(
            f"{log_prefix} results: passed={bool(passed)}, p_value={p_value:.4f}, observed_proportion={successes / sample_size:.2f}, significance_level={self.significance_level}"
        )
        logger.debug(f"Full {log_prefix} results: {results}")
        if label:
            results = {
                f"{label}: {k.replace('_', ' ').title()}": v for k, v in results.items()
            }
            # Also include the raw "passed" key for the framework's final status check
            results["passed"] = passed

        logger.info(
            f"{log_prefix} results: passed={bool(passed)}, p_value={p_value:.4f}, observed_proportion={successes / sample_size:.2f}, significance_level={self.significance_level}"
        )
        return results

