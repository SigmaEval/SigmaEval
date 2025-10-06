"""
SigmaEval: A Python library for evaluating Generative AI agents and apps.
"""

from sigmaeval.core import (
    SigmaEval,
    ScenarioTest,
    BehavioralExpectation,
    AppResponse,
    ConversationRecord,
    EvaluationResult,
    RetryConfig,
    WritingStyleConfig,
    WritingStyleAxes,
)
from .assertions import assertions

__version__ = "0.1.0"
__author__ = "Itura AI"
__license__ = "Apache-2.0"

__all__ = [
    "SigmaEval",
    "ScenarioTest",
    "BehavioralExpectation",
    "AppResponse",
    "ConversationRecord",
    "EvaluationResult",
    "RetryConfig",
    "WritingStyleConfig",
    "WritingStyleAxes",
    "assertions",
]

