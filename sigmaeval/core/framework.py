"""
Framework orchestration logic for SigmaEval.
"""

from typing import Callable, Awaitable, Any, Dict

from .models import AppResponse, BehavioralTest
from .rubric_generator import _parse_behavioral_test, _generate_rubric
from .data_collection import collect_evaluation_data


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
        app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
        concurrency: int = 10,
        max_turns: int = 10
    ) -> dict[str, Any]:
        """
        Run evaluation for a behavioral test case.
        
        Args:
            scenario: The behavioral test case to evaluate
            app_handler: Async callback that takes a user message and state dict, and returns 
                an AppResponse. The state dict is managed by your application - SigmaEval 
                passes back the state from your previous AppResponse without modification. 
                On the first turn, state will be an empty dict. Use it to track conversation 
                history, user context, or any other stateful information your app needs.
            concurrency: Number of evaluations to run concurrently (default: 10)
            max_turns: Maximum conversation turns per interaction (default: 10)
            
        Returns:
            Dictionary containing evaluation results
        """
        # Phase 1: Test Setup
        # 1. Parse BehavioralTest
        parsed_test = _parse_behavioral_test(scenario)
        
        # 2. Generate rubric from expected_behavior
        rubric = await _generate_rubric(scenario, self.model)
        
        # Phase 2: Data Collection (repeated sample_size times)
        #   3. Simulate user with User Simulator LLM
        #   4. Initiate and record interaction with system under test via app_handler
        #   5. Judge expected behavior with Judge LLM using rubric
        sample_size = parsed_test["sample_size"]
        
        scores, reasoning_list, conversations = await collect_evaluation_data(
            scenario=scenario,
            app_handler=app_handler,
            rubric=rubric,
            model=self.model,
            sample_size=sample_size,
            concurrency=concurrency,
            max_turns=max_turns
        )
        
        # TODO: Phase 3: Statistical Analysis
        #   6. Pass scores to evaluator for statistical testing
        
        return {
            "model": self.model,
            "test_config": parsed_test,
            "rubric": rubric,
            "scores": scores,
            "reasoning": reasoning_list,
            "conversations": conversations,
            "num_conversations": len(conversations),
        }
