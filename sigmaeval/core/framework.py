"""
Framework orchestration logic for SigmaEval.
"""

import logging
import asyncio
from typing import Callable, Awaitable, Any, Dict, List

from .models import (
    AppResponse,
    ScenarioTest,
    ScenarioTestResult,
    ExpectationResult,
    AssertionResult,
    WritingStyleConfig,
    BehavioralExpectation,
    MetricExpectation,
)
from .rubric_generator import _generate_rubric
from .data_collection import _collect_conversations, _judge_conversations
from .models import RetryConfig
from .utils import _convert_conversation_records

from ..assertions import MedianAssertion, ProportionAssertion
from .._evaluators import (
    MedianEvaluator,
    ProportionEvaluator,
)


class SigmaEval:
    """
    Main evaluation framework for AI applications.
    
    Combines inferential statistics, AI-driven user simulation, and LLM-as-a-Judge
    evaluation within a BDD framework.
    """
    
    def __init__(
        self,
        judge_model: str,
        significance_level: float,
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
            significance_level: The significance level (alpha) for statistical
                tests, representing the probability of detecting an effect that is
                not actually present (a "false positive"). A lower value means a
                stricter test. The standard value is 0.05. This can be overridden
                on a per-assertion basis.
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
        self.significance_level = significance_level
        self.writing_style_config = writing_style_config or WritingStyleConfig()
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.setLevel(log_level)
    
    async def _evaluate_single(
        self,
        scenario: ScenarioTest,
        app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
        concurrency: int = 10,
    ) -> ScenarioTestResult:
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
        self.logger.info(f"--- Starting evaluation for ScenarioTest: {scenario.title} ---")

        # Phase 2 (first half): Data Collection via Simulation
        # This is done only once per ScenarioTest, regardless of how many
        # expectations are in the `then` clause.
        self.logger.info(f"Simulating {scenario.sample_size} conversations for '{scenario.title}'...")
        conversations = await _collect_conversations(
            scenario=scenario,
            app_handler=app_handler,
            user_simulator_model=self.user_simulator_model,
            sample_size=scenario.sample_size,
            concurrency=concurrency,
            max_turns=scenario.max_turns,
            retry_config=self.retry_config,
            writing_style_config=self.writing_style_config,
        )

        expectation_results = []
        all_rubrics = []
        
        # A ScenarioTest can have multiple `then` clauses (BehavioralExpectations)
        # Each one is evaluated independently against the same set of conversations.
        for expectation in scenario.then:
            assertion_results = []
            scores = []
            reasoning = []
            
            if isinstance(expectation, BehavioralExpectation):
                # Phase 1: Test Setup
                # Generate a rubric for this specific expectation
                self.logger.debug(f"Generating rubric for expectation: {expectation.label or expectation.expected_behavior[:50]}")
                rubric = await _generate_rubric(
                    scenario=scenario, 
                    expectation=expectation, 
                    model=self.judge_model, 
                    retry_config=self.retry_config
                )
                self.logger.debug(f"Generated rubric: {rubric}")
                all_rubrics.append(rubric)
                
                # Phase 2 (second half): Judging
                # The collected conversations are now judged against the new rubric.
                scores, reasoning = await _judge_conversations(
                    scenario=scenario,
                    expectation=expectation,
                    conversations=conversations,
                    rubric=rubric,
                    judge_model=self.judge_model,
                    concurrency=concurrency,
                    retry_config=self.retry_config,
                )
                
                # Phase 3: Statistical Analysis
                self.logger.debug(f"Collected scores for '{scenario.title}': {scores}")
                
                log_msg = f"Starting statistical analysis for '{scenario.title}'"
                if expectation.label:
                    log_msg += f" (Expectation: {expectation.label})"
                self.logger.info(log_msg)

                criteria_list = expectation.criteria if isinstance(expectation.criteria, list) else [expectation.criteria]
                for criteria in criteria_list:
                    evaluator = None
                    significance_level = criteria.significance_level or self.significance_level
                    if isinstance(criteria, ProportionAssertion):
                        evaluator = ProportionEvaluator(
                            significance_level=significance_level,
                            threshold=criteria.threshold,
                            proportion=criteria.proportion,
                            comparison=criteria.comparison,
                        )
                    elif isinstance(criteria, MedianAssertion):
                        evaluator = MedianEvaluator(
                            significance_level=significance_level,
                            threshold=criteria.threshold,
                            comparison=criteria.comparison,
                        )
                    else:
                        raise TypeError(f"Unsupported criteria type: {type(criteria)}")

                    eval_result_dict = evaluator.evaluate(scores, label=expectation.label)
                    
                    about_str = "Unknown assertion"
                    if isinstance(criteria, ProportionAssertion):
                        about_str = f"proportion of scores {criteria.comparison} {criteria.proportion} (threshold: {criteria.threshold})"
                    elif isinstance(criteria, MedianAssertion):
                        about_str = f"median score {criteria.comparison} {criteria.threshold}"

                    assertion_results.append(
                        AssertionResult(
                            about=about_str,
                            passed=eval_result_dict["passed"],
                            p_value=eval_result_dict.get("p_value"),
                            details=eval_result_dict,
                        )
                    )

            elif isinstance(expectation, MetricExpectation):
                metric = expectation.metric
                # Calculate metric values for all conversations
                all_metric_values = []
                for conv in conversations:
                    all_metric_values.extend(metric(conv))
                
                criteria_list = expectation.criteria if isinstance(expectation.criteria, list) else [expectation.criteria]
                for criteria in criteria_list:
                    evaluator = None
                    significance_level = criteria.significance_level or self.significance_level
                    if isinstance(criteria, ProportionAssertion):
                        evaluator = ProportionEvaluator(
                            significance_level=significance_level,
                            threshold=criteria.threshold,
                            proportion=criteria.proportion,
                            comparison=criteria.comparison,
                        )
                    elif isinstance(criteria, MedianAssertion):
                        evaluator = MedianEvaluator(
                            significance_level=significance_level,
                            threshold=criteria.threshold,
                            comparison=criteria.comparison,
                        )
                    else:
                        raise TypeError(f"Unsupported criteria type for MetricExpectation: {type(criteria)}")
                    
                    eval_result_dict = evaluator.evaluate(all_metric_values, label=expectation.label)
                    
                    about_str = "Unknown assertion"
                    if isinstance(criteria, ProportionAssertion):
                        about_str = f"{metric.name} {criteria.comparison} {criteria.proportion} (threshold: {criteria.threshold})"
                    elif isinstance(criteria, MedianAssertion):
                        about_str = f"median {metric.name} {criteria.comparison} {criteria.threshold}"

                    assertion_results.append(
                        AssertionResult(
                            about=about_str,
                            passed=eval_result_dict["passed"],
                            p_value=eval_result_dict.get("p_value"),
                            details=eval_result_dict,
                        )
                    )

            expectation_results.append(
                ExpectationResult(
                    about=expectation.label
                    or expectation.expected_behavior[:50]
                    if isinstance(expectation, BehavioralExpectation)
                    else expectation.metric.name,
                    assertion_results=assertion_results,
                    scores=scores if isinstance(expectation, BehavioralExpectation) else all_metric_values,
                    reasoning=reasoning if isinstance(expectation, BehavioralExpectation) else [],
                )
            )

        self.logger.info(f"--- Evaluation complete for: {scenario.title} ---")
        
        return ScenarioTestResult(
            title=scenario.title,
            expectation_results=expectation_results,
            conversations=_convert_conversation_records(conversations),
            significance_level=self.significance_level,
            judge_model=self.judge_model,
            user_simulator_model=self.user_simulator_model,
            retry_config=self.retry_config,
            rubric="\\n\\n---\\n\\n".join(all_rubrics) if all_rubrics else None,
        )

    async def evaluate(
        self, 
        scenarios: ScenarioTest | List[ScenarioTest], 
        app_handler: Callable[[str, Dict[str, Any]], Awaitable[AppResponse]],
        concurrency: int = 10,
    ) -> ScenarioTestResult | List[ScenarioTestResult]:
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
            - If a single `ScenarioTest` is provided, returns a single `EvaluationResult`.
            - If a list of `ScenarioTest` is provided, returns a list of `EvaluationResult`.
        
        Raises:
            LLMCommunicationError: If any LLM call (rubric generation, user simulation,
                or judging) fails or returns an invalid/malformed response.
        """
        is_single_item = False
        if isinstance(scenarios, ScenarioTest):
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
