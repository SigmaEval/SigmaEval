"""
SigmaEval - A Python library for evaluating Generative AI agents and apps.
"""

from sigmaeval.core import SigmaEval, BehavioralTestCase, Expectation
from sigmaeval.evaluators import (
    BinaryEvaluator,
    RatingMeanEvaluator,
    RatingProportionEvaluator,
)

__version__ = "0.1.0"
__author__ = "Itura AI"
__license__ = "Apache-2.0"

__all__ = [
    "SigmaEval",
    "BehavioralTestCase",
    "Expectation",
    "BinaryEvaluator",
    "RatingMeanEvaluator",
    "RatingProportionEvaluator",
]

