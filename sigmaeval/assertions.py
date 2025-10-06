from dataclasses import dataclass
from typing import Optional


@dataclass
class ProportionGTE:
    min_score: int
    proportion: float
    significance_level: Optional[float] = None


@dataclass
class MedianGTE:
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


class Assertions:
    def __init__(self):
        self.scores = Scores()


assertions = Assertions()
