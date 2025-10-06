from typing import Optional, Literal
from pydantic import BaseModel, Field


class Assertion(BaseModel):
    """Base class for all assertion criteria."""

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


class Scores:
    def proportion_gte(
        self,
        min_score: int,
        proportion: float,
        significance_level: Optional[float] = None,
    ) -> ProportionAssertion:
        if not (1 <= min_score <= 10):
            raise ValueError("min_score must be between 1 and 10")
        if not (0 <= proportion <= 1):
            raise ValueError("proportion must be between 0 and 1")
        if significance_level is not None and not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return ProportionAssertion(
            threshold=min_score,
            proportion=proportion,
            comparison="gte",
            significance_level=significance_level,
        )

    def median_gte(
        self, threshold: float, significance_level: Optional[float] = None
    ) -> MedianAssertion:
        if not (1 <= threshold <= 10):
            raise ValueError("threshold must be between 1 and 10")
        if significance_level is not None and not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return MedianAssertion(
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
    ) -> ProportionAssertion:
        if not (0 <= proportion <= 1):
            raise ValueError("proportion must be between 0 and 1")
        if significance_level is not None and not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return ProportionAssertion(
            threshold=threshold,
            proportion=proportion,
            comparison="lte",
            significance_level=significance_level,
        )

    def median_lt(
        self, threshold: float, significance_level: Optional[float] = None
    ) -> MedianAssertion:
        if significance_level is not None and not (0 < significance_level < 1):
            raise ValueError("significance_level must be between 0 and 1")
        return MedianAssertion(
            threshold=threshold,
            comparison="lte",
            significance_level=significance_level,
        )


class Assertions:
    def __init__(self):
        self.scores = Scores()
        self.metrics = Metrics()


assertions = Assertions()
