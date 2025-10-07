from typing import Optional, Literal
from pydantic import BaseModel, Field


class Assertion(BaseModel):
    """Base class for all assertion criteria."""

    pass


class ScoreAssertion(Assertion):
    """Base class for assertions on scores."""

    pass


class MetricAssertion(Assertion):
    """Base class for assertions on metrics."""

    pass


class ProportionAssertion(Assertion):
    """
    Asserts that the proportion of outcomes meeting a threshold satisfies a
    statistical comparison.

    This assertion is used for hypothesis testing on proportions. It checks if
    the true proportion of "successful" outcomes in the entire population is
    likely to be above or below a certain value based on the collected sample.

    A "successful" outcome is defined as a score or metric value that meets a
    given threshold (e.g., a score >= 7, or latency < 1.5s).

    Attributes:
        threshold: The value that a score or metric must meet to be considered
            a "success".
        proportion: The hypothesized proportion of successes to test against.
        comparison: The type of comparison to perform (``gte`` for greater than
            or equal to, ``lte`` for less than or equal to).
        significance_level: The significance level (alpha) for the hypothesis
            test. Overrides the default if provided.
    """

    threshold: float = Field(
        ..., description="The threshold for an outcome to be counted as a 'success'."
    )
    proportion: float = Field(..., description="The proportion of successes to compare against.")
    comparison: Literal["gte", "lte"] = Field(
        ...,
        description="The type of comparison to perform (greater than or equal to, or less than or equal to).",
    )
    significance_level: Optional[float] = Field(
        None, description="The significance level (alpha) for the hypothesis test."
    )


class MedianAssertion(Assertion):
    """
    Asserts that the median of a set of values satisfies a statistical comparison
    with a given threshold.

    This assertion is used for hypothesis testing on the median. It is
    particularly useful for non-normally distributed data, such as scores or
    latency metrics, as the median is robust to outliers. The test is performed
    using a bootstrap hypothesis test.

    Attributes:
        threshold: The threshold to compare the median against.
        comparison: The type of comparison to perform (``gte`` for greater than
            or equal to, ``lte`` for less than or equal to).
        significance_level: The significance level (alpha) for the hypothesis
            test. Overrides the default if provided.
    """

    threshold: float = Field(..., description="The threshold to compare the median against.")
    comparison: Literal["gte", "lte"] = Field(
        ...,
        description="The type of comparison to perform (greater than or equal to, or less than or equal to).",
    )
    significance_level: Optional[float] = Field(
        None, description="The significance level (alpha) for the hypothesis test."
    )


class ScoreProportionAssertion(ProportionAssertion, ScoreAssertion):
    pass


class ScoreMedianAssertion(MedianAssertion, ScoreAssertion):
    pass


class MetricProportionAssertion(ProportionAssertion, MetricAssertion):
    pass


class MetricMedianAssertion(MedianAssertion, MetricAssertion):
    pass


class Scores:
    def proportion_gte(
        self,
        min_score: int,
        proportion: float,
        significance_level: Optional[float] = None,
    ) -> ScoreProportionAssertion:
        """
        Asserts that the proportion of scores >= min_score is statistically >= proportion.

        This criterion performs a one-sided hypothesis test to determine if the
        true proportion of high-quality outcomes is greater than a specified
        minimum. A score at or above ``min_score`` is considered a "high-quality"
        outcome. The test passes if there is statistical evidence that the
        system's performance exceeds the ``proportion``.

        Args:
            min_score: The minimum score (inclusive) to be considered a success.
                Must be between 1 and 10.
            proportion: The minimum proportion of successful outcomes. Must be
                between 0 and 1.
            significance_level: The significance level (alpha) for the test.
                Overrides the default if provided.

        Example:
            .. code-block:: python

                # Asserts that it's statistically likely that at least 90% of
                # responses will have a score of 7 or higher.
                assertions.scores.proportion_gte(min_score=7, proportion=0.90)
        """
        if not (1 <= min_score <= 10):
            raise ValueError("min_score must be between 1 and 10")
        if not (0 <= proportion <= 1):
            raise ValueError("proportion must be between 0 and 1")
        if significance_level is not None and not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return ScoreProportionAssertion(
            threshold=min_score,
            proportion=proportion,
            comparison="gte",
            significance_level=significance_level,
        )

    def median_gte(
        self, threshold: float, significance_level: Optional[float] = None
    ) -> ScoreMedianAssertion:
        """
        Asserts that the median score is statistically >= threshold.

        This criterion performs a one-sided bootstrap hypothesis test to
        determine if the true median score is statistically higher than the
        specified ``threshold``. By testing the median, it ensures that at
        least 50% of responses meet a certain quality bar.

        Args:
            threshold: The median score threshold to test against. Must be
                between 1 and 10.
            significance_level: The significance level (alpha) for the test.
                Overrides the default if provided.

        Example:
            .. code-block:: python

                # Asserts that it's statistically likely that the median score
                # is greater than or equal to 8.
                assertions.scores.median_gte(threshold=8)
        """
        if not (1 <= threshold <= 10):
            raise ValueError("threshold must be between 1 and 10")
        if significance_level is not None and not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return ScoreMedianAssertion(
            threshold=threshold,
            comparison="gte",
            significance_level=significance_level,
        )


class Metrics:
    def proportion_lt(
        self,
        threshold: float,
        proportion: float,
        significance_level: Optional[float] = None,
    ) -> MetricProportionAssertion:
        """
        Asserts that the proportion of metric values < threshold is statistically >= proportion.

        This criterion performs a one-sided hypothesis test to determine if the
        true proportion of metric values below a certain ``threshold`` is
        statistically significant. This is useful for "lower is better"
        metrics like latency.

        Args:
            threshold: The threshold that the metric value should be less than.
            proportion: The minimum proportion of values that must be below the
                threshold. Must be between 0 and 1.
            significance_level: The significance level (alpha) for the test.
                Overrides the default if provided.

        Example:
            .. code-block:: python

                # Asserts that it's statistically likely that at least 95% of
                # responses have a latency of less than 1.5 seconds.
                assertions.metrics.proportion_lt(threshold=1.5, proportion=0.95)
        """
        if not (0 <= proportion <= 1):
            raise ValueError("proportion must be between 0 and 1")
        if significance_level is not None and not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return MetricProportionAssertion(
            threshold=threshold,
            proportion=proportion,
            comparison="lte",
            significance_level=significance_level,
        )

    def median_lt(
        self, threshold: float, significance_level: Optional[float] = None
    ) -> MetricMedianAssertion:
        """
        Asserts that the median metric value is statistically < threshold.

        This criterion performs a one-sided bootstrap hypothesis test to
        determine if the true median of a metric is statistically lower than a
        specified ``threshold``. This is ideal for skewed, "lower is better"
        metrics like latency or turn count.

        Args:
            threshold: The median threshold to test against.
            significance_level: The significance level (alpha) for the test.
                Overrides the default if provided.

        Example:
            .. code-block:: python

                # Asserts that it's statistically likely that the median
                # number of turns in a conversation is less than 4.
                assertions.metrics.median_lt(threshold=4.0)
        """
        if significance_level is not None and not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return MetricMedianAssertion(
            threshold=threshold,
            comparison="lte",
            significance_level=significance_level,
        )


class Assertions:
    def __init__(self):
        self.scores = Scores()
        self.metrics = Metrics()


assertions = Assertions()
