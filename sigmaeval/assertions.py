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
    """

    threshold: float = Field(..., description="The threshold for an outcome to be counted as a 'success'.")
    proportion: float = Field(
        ..., description="The proportion of successes to compare against."
    )
    comparison: Literal["gte", "lte"] = Field(
        ..., description="The type of comparison to perform (greater than or equal to, or less than or equal to)."
    )
    significance_level: Optional[float] = Field(
        None, description="The significance level (alpha) for the hypothesis test."
    )


class MedianAssertion(Assertion):
    """
    Asserts that the median of a set of values satisfies a statistical comparison
    with a given threshold.
    """

    threshold: float = Field(..., description="The threshold to compare the median against.")
    comparison: Literal["gte", "lte"] = Field(
        ..., description="The type of comparison to perform (greater than or equal to, or less than or equal to)."
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
