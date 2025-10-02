"""
SigmaEval - A Python library for evaluating Generative AI agents and apps.
"""

from sigmaeval.core import SigmaEval, BehavioralTest, Expectation, AppResponse
from sigmaeval.evaluators import (
    SuccessRateEvaluator,
    RatingMeanEvaluator,
    RatingProportionEvaluator,
)

__version__ = "0.1.0"
__author__ = "Itura AI"
__license__ = "Apache-2.0"

__all__ = [
    "SigmaEval",
    "BehavioralTest",
    "Expectation",
    "AppResponse",
    "SuccessRateEvaluator",
    "RatingMeanEvaluator",
    "RatingProportionEvaluator",
]

