"""
Core classes for SigmaEval framework.
"""

from typing import Callable, Awaitable, Any, Dict
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


class SigmaEval:
    """
    Main evaluation framework for AI applications.
    
    Combines inferential statistics, AI-driven user simulation, and LLM-as-a-Judge
    evaluation within a BDD framework.
    """
    
    def __init__(self):
        """Initialize SigmaEval framework."""
        pass
    
    async def evaluate(
        self, 
        scenario: BehavioralTest, 
        app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]]
    ) -> dict[str, Any]:
        """
        Run evaluation for a behavioral test case.
        
        Args:
            scenario: The behavioral test case to evaluate
            app_handler: Async callback that takes a message and state dict, and returns an AppResponse
            
        Returns:
            Dictionary containing evaluation results
        """
        # TODO: Implement evaluation logic
        # Phase 1: Test Setup
        #   1. Parse BehavioralTest
        #   2. Generate rubric from expected_behavior
        
        # Phase 2: Data Collection (repeated sample_size times)
        #   3. Simulate user with User Simulator LLM
        #   4. Record interaction via app_callback
        #   5. Judge expected behavior with Judge LLM using rubric
        
        # Phase 3: Statistical Analysis
        #   6. Pass scores to evaluator for statistical testing
        
        return {}

