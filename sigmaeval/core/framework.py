"""
Framework orchestration logic for SigmaEval.
"""

from typing import Callable, Awaitable, Any, Dict

from .models import AppResponse, BehavioralTest
from .rubric_generator import _parse_behavioral_test, _generate_rubric


class SigmaEval:
    """
    Main evaluation framework for AI applications.
    
    Combines inferential statistics, AI-driven user simulation, and LLM-as-a-Judge
    evaluation within a BDD framework.
    """
    
    def __init__(self, model: str):
        """
        Initialize SigmaEval framework.

        Args:
            model: Fully-qualified model identifier used for LLM-as-a-Judge, e.g.,
                "openai/gpt-4o". The application under test may use any model; this
                parameter configures the judge model.

        Note:
            SigmaEval uses LiteLLM as the unified interface for the LLM-as-a-Judge.
            For a complete list of supported providers, refer to the LiteLLM documentation:
            https://docs.litellm.ai/docs/providers
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string, e.g., 'openai/gpt-4o'.\nFor a complete list of supported providers, refer to the LiteLLM documentation: https://docs.litellm.ai/docs/providers")

        self.model: str = model
    
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
        # Phase 1: Test Setup
        # 1. Parse BehavioralTest
        parsed_test = _parse_behavioral_test(scenario)
        
        # 2. Generate rubric from expected_behavior
        rubric = await _generate_rubric(scenario, self.model)
        
        # TODO: Phase 2: Data Collection (repeated sample_size times)
        #   3. Simulate user with User Simulator LLM
        #   4. Record interaction via app_callback
        #   5. Judge expected behavior with Judge LLM using rubric
        
        # TODO: Phase 3: Statistical Analysis
        #   6. Pass scores to evaluator for statistical testing
        
        return {
            "model": self.model,
            "test_config": parsed_test,
            "rubric": rubric,
        }
