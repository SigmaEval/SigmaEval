"""
Framework orchestration logic for SigmaEval.
"""

import logging
import asyncio
from typing import Callable, Awaitable, Any, Dict, List

from .models import AppResponse, BehavioralTest, EvaluationResult, WritingStyleConfig
from .rubric_generator import _parse_behavioral_test, _generate_rubric
from .data_collection import collect_evaluation_data
from .models import RetryConfig


class SigmaEval:
    """
    Main evaluation framework for AI applications.
    
    Combines inferential statistics, AI-driven user simulation, and LLM-as-a-Judge
    evaluation within a BDD framework.
    """
    
    def __init__(
        self,
        judge_model: str,
        user_simulator_model: str | None = None,
        log_level: int = logging.INFO,
        retry_config: RetryConfig | None = None,
        writing_style_config: WritingStyleConfig | None = None,
    ):
        """
        Initialize SigmaEval framework.

        Args:
            judge_model: Fully-qualified model identifier used for the Judge LLM
                and rubric generation, e.g., "openai/gpt-4o". The application under
                test may use any model; this parameter configures the judge model.
            user_simulator_model: Optional model identifier for the User Simulator
                LLM. If None, the `judge_model` will be used for all roles.
            log_level: The logging level for the 'sigmaeval' logger.
            retry_config: Optional configuration for Tenacity-based retries on
                LiteLLM calls. If None, default settings are used (enabled=True, 
                max_attempts=5, backoff_multiplier=0.5, max_backoff_seconds=30.0).
            writing_style_config: Optional configuration for user simulator writing
                style variations. To disable, pass `WritingStyleConfig(enabled=False)`.
                See `WritingStyleAxes` for default style definitions.

        Note:
            SigmaEval uses LiteLLM as the unified interface for the LLM-as-a-Judge.
            For a complete list of supported providers, refer to the LiteLLM documentation:
            https://docs.litellm.ai/docs/providers
            
            Tenacity-based retries are applied to rubric generation, user simulation,
            and judge calls. Retries can be disabled via the RetryConfig object.
        """
        if not isinstance(judge_model, str) or not judge_model.strip():
            raise ValueError("judge_model must be a non-empty string, e.g., 'openai/gpt-4o'.\nFor a complete list of supported providers, refer to the LiteLLM documentation: https://docs.litellm.ai/docs/providers")

        self.judge_model: str = judge_model
        self.user_simulator_model: str = user_simulator_model or judge_model
        self.logger = logging.getLogger("sigmaeval")
        self.retry_config = retry_config or RetryConfig()
        self.writing_style_config = writing_style_config or WritingStyleConfig()
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.setLevel(log_level)
    
    async def _evaluate_single(
        self,
        scenario: BehavioralTest,
        app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
        concurrency: int = 10,
    ) -> EvaluationResult:
        """
        Run evaluation for a single behavioral test case.
        
        Args:
            scenario: The behavioral test case to evaluate
            app_handler: Async callback that takes a user message and state dict, and returns 
                an AppResponse. The state dict is managed by your application - SigmaEval 
                passes back the state from your previous AppResponse without modification. 
                On the first turn, state will be an empty dict. Use it to track conversation 
                history, user context, or any other stateful information your app needs.
            concurrency: Number of evaluations to run concurrently (default: 10)
            
        Returns:
            EvaluationResult: A data class containing the evaluation results.
        
        Raises:
            LLMCommunicationError: If any LLM call (rubric generation, user simulation,
                or judging) fails or returns an invalid/malformed response.
        """
        self.logger.info(f"--- Starting evaluation for BehavioralTest: {scenario.title} ---")
        
        # Phase 1: Test Setup
        # 1. Parse BehavioralTest
        self.logger.debug(f"Parsing BehavioralTest: {scenario.title}")
        parsed_test = _parse_behavioral_test(scenario)
        
        # 2. Generate rubric from expected_behavior
        self.logger.debug("Generating rubric...")
        rubric = await _generate_rubric(scenario, self.judge_model, self.retry_config)
        self.logger.debug(f"Generated rubric: {rubric}")
        
        # Phase 2: Data Collection (repeated sample_size times)
        #   3. Simulate user with User Simulator LLM
        #   4. Initiate and record interaction with system under test via app_handler
        #   5. Judge expected behavior with Judge LLM using rubric
        sample_size = parsed_test["sample_size"]
        self.logger.info(f"Collecting {sample_size} samples for '{scenario.title}'...")
        
        scores, reasoning_list, conversations = await collect_evaluation_data(
            scenario=scenario,
            app_handler=app_handler,
            rubric=rubric,
            judge_model=self.judge_model,
            user_simulator_model=self.user_simulator_model,
            retry_config=self.retry_config,
            sample_size=sample_size,
            concurrency=concurrency,
            max_turns=scenario.max_turns,
            writing_style_config=self.writing_style_config,
        )
        
        # Phase 3: Statistical Analysis
        #   6. Pass scores to evaluator for statistical testing
        self.logger.debug(f"Collected scores for '{scenario.title}': {scores}")
        self.logger.info(f"Starting statistical analysis for '{scenario.title}'...")
        evaluator = scenario.then.evaluator
        results = evaluator.evaluate(scores)
        
        self.logger.info(f"--- Evaluation complete for: {scenario.title} ---")
        
        return EvaluationResult(
            judge_model=self.judge_model,
            user_simulator_model=self.user_simulator_model,
            test_config=parsed_test,
            retry_config=self.retry_config,
            rubric=rubric,
            scores=scores,
            reasoning=reasoning_list,
            conversations=conversations,
            num_conversations=len(conversations),
            results=results,
        )

    async def evaluate(
        self, 
        scenarios: BehavioralTest | List[BehavioralTest], 
        app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
        concurrency: int = 10,
    ) -> EvaluationResult | List[EvaluationResult]:
        """
        Run evaluation for one or more behavioral test cases. When a list of tests
        is provided, they are run concurrently.
        
        Args:
            scenarios: A single behavioral test case or a list of test cases to evaluate.
            app_handler: Async callback that takes a user message and state dict, and returns 
                an AppResponse. The state dict is managed by your application - SigmaEval 
                passes back the state from your previous AppResponse without modification. 
                On the first turn, state will be an empty dict. Use it to track conversation 
                history, user context, or any other stateful information your app needs.
            concurrency: Number of evaluations to run concurrently for each test case
                (default: 10). This refers to the concurrency of the sample collection
                within a single test, not the concurrency of running multiple tests in parallel.
            
        Returns:
            - If a single `BehavioralTest` is provided, returns a single `EvaluationResult`.
            - If a list of `BehavioralTest` is provided, returns a list of `EvaluationResult`.
        
        Raises:
            LLMCommunicationError: If any LLM call (rubric generation, user simulation,
                or judging) fails or returns an invalid/malformed response.
        """
        is_single_item = False
        if isinstance(scenarios, BehavioralTest):
            scenarios = [scenarios]
            is_single_item = True
        else:
            self.logger.info(f"--- Starting evaluation for test suite with {len(scenarios)} scenarios ---")

        tasks = [
            self._evaluate_single(scenario, app_handler, concurrency)
            for scenario in scenarios
        ]
        all_results = await asyncio.gather(*tasks)

        if not is_single_item:
            self.logger.info("--- Test suite evaluation complete ---")

        if is_single_item:
            return all_results[0]
        
        return all_results
