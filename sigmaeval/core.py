"""
Core classes for SigmaEval framework.
"""

from typing import Callable, Awaitable, Any
from pydantic import BaseModel, Field


class Then(BaseModel):
    """
    Defines the expected outcome and evaluation method for a behavioral test case.
    
    Attributes:
        outcome: Description of the expected outcome (passed to Judge LLM)
        evaluator: Statistical evaluator to assess the results
    """
    outcome: str = Field(..., description="Expected outcome description")
    evaluator: Any = Field(..., description="Evaluator instance for statistical analysis")


class BehavioralTestCase(BaseModel):
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
    then: Then = Field(..., description="Expected outcome and evaluator")


class SigmaEval:
    """
    Main evaluation framework for AI applications.
    
    Combines inferential statistics, AI-driven user simulation, and LLM-as-a-Judge
    evaluation within a BDD framework.
    """
    
    def __init__(self):
        """Initialize SigmaEval framework."""
        pass
    
    async def run(
        self, 
        scenario: BehavioralTestCase, 
        app_callback: Callable[[str], Awaitable[str]]
    ) -> dict[str, Any]:
        """
        Run evaluation for a behavioral test case.
        
        Args:
            scenario: The behavioral test case to evaluate
            app_callback: Async callback function that takes a message and returns app's response
            
        Returns:
            Dictionary containing evaluation results
        """
        # TODO: Implement evaluation logic
        # Phase 1: Test Setup
        #   1. Parse BehavioralTestCase
        #   2. Generate rubric from outcome
        
        # Phase 2: Data Collection (repeated num_of_samples times)
        #   3. Simulate user with User Simulator LLM
        #   4. Record interaction via app_callback
        #   5. Judge outcome with Judge LLM using rubric
        
        # Phase 3: Statistical Analysis
        #   6. Pass scores to evaluator for statistical testing
        
        return {}

