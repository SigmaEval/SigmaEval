"""
The sigmaeval package.
"""

from .core.framework import SigmaEval
from .core.models import (
    AppResponse,
    BehavioralExpectation,
    ScenarioTestResult,
    ScenarioTest,
    RetryConfig,
    WritingStyleConfig,
    WritingStyleAxes,
    MetricExpectation,
    Turn,
    Conversation,
)
from .assertions import assertions
from .metrics import metrics

__version__ = "0.1.0"
__author__ = "Itura AI"
__license__ = "Apache-2.0"

__all__ = [
    "SigmaEval",
    "AppResponse",
    "ScenarioTest",
    "BehavioralExpectation",
    "ScenarioTestResult",
    "assertions",
    "RetryConfig",
    "WritingStyleConfig",
    "WritingStyleAxes",
    "MetricExpectation",
    "metrics",
    "Turn",
    "Conversation",
]

