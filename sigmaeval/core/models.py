"""
Data models for the SigmaEval core package.
"""

from typing import Any, Dict
from pydantic import BaseModel, Field


class AppResponse(BaseModel):
    """
    The response from the application under test for a single turn.

    Attributes:
        response: The string response from the app.
        state: An updated state object to be passed to the next turn.
    """
    response: str
    state: Dict[str, Any]


class Expectation(BaseModel):
    """
    Defines the expected outcome and evaluation method for a behavioral test case.
    
    Attributes:
        expected_behavior: Description of the expected behavior (passed to Judge LLM)
        evaluator: Statistical evaluator to assess the results
    """
    expected_behavior: str = Field(..., description="Expected behavior description")
    evaluator: Any = Field(..., description="Evaluator instance for statistical analysis")


class BehavioralTest(BaseModel):
    """
    Defines a behavioral test scenario using Given-When-Then BDD format.
    
    Attributes:
        title: Short descriptive title for the test case
        given: Prerequisite state and context for User Simulator LLM
        when: Specific goal or action the User Simulator LLM will attempt
        then: Expected outcome with evaluator for statistical analysis
    """
    title: str = Field(..., description="Test case title")
    given: str = Field(..., description="Context and prerequisites")
    when: str = Field(..., description="User goal or action")
    then: Expectation = Field(..., description="Expected outcome and evaluator")


