from typing import Optional
from pydantic import BaseModel


class Assertion(BaseModel):
    """Base class for all assertion criteria."""
    pass


class ProportionGTE(Assertion):
    min_score: int
    proportion: float
    significance_level: Optional[float] = None


class MedianGTE(Assertion):
    threshold: float
    significance_level: Optional[float] = None


class ProportionLT(Assertion):
    threshold: float
    proportion: float
    significance_level: Optional[float] = None


class MedianLT(Assertion):
    threshold: float
    significance_level: Optional[float] = None


class Scores:
    def proportion_gte(
        self,
        min_score: int,
        proportion: float,
        significance_level: Optional[float] = None,
    ) -> ProportionGTE:
        return ProportionGTE(
            min_score=min_score,
            proportion=proportion,
            significance_level=significance_level,
        )

    def median_gte(
        self, threshold: float, significance_level: Optional[float] = None
    ) -> MedianGTE:
        return MedianGTE(threshold=threshold, significance_level=significance_level)


class Metrics:
    def proportion_lt(
        self,
        threshold: float,
        proportion: float,
        significance_level: Optional[float] = None,
    ) -> ProportionLT:
        return ProportionLT(
            threshold=threshold,
            proportion=proportion,
            significance_level=significance_level,
        )

    def median_lt(
        self, threshold: float, significance_level: Optional[float] = None
    ) -> MedianLT:
        return MedianLT(threshold=threshold, significance_level=significance_level)


class Assertions:
    def __init__(self):
        self.scores = Scores()
        self.metrics = Metrics()


assertions = Assertions()
