"""
Framework orchestration logic for SigmaEval.
"""

import logging
import asyncio
from typing import Callable, Awaitable, Any, Dict, List, Tuple, Union
import warnings

from .models import (
    AppResponse,
    ScenarioTest,
    ScenarioTestResult,
    ExpectationResult,
    AssertionResult,
    WritingStyleConfig,
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


# Define the flexible app_handler signature and its possible return types
AppHandler = Callable[
    [List[Dict[str, str]], Any],
    Awaitable[Union[AppResponse, str, Tuple[str, Any]]],
]


class SigmaEval:
    """
    The main evaluation framework for AI applications.

    SigmaEval combines inferential statistics, AI-driven user simulation, and
    LLM-as-a-Judge evaluation within a Behavior-Driven Development (BDD)
    framework. This approach allows you to move beyond simple pass/fail tests
    and gain statistical confidence in your AI's performance.

    The evaluation process for a single :class:`~sigmaeval.ScenarioTest` unfolds
    in three main phases:
    1.  **Test Setup**: A detailed scoring rubric is generated based on the
        test's expected behavior.
    2.  **Data Collection**: An LLM simulates user interactions with your app to
        collect conversation data. An LLM Judge then scores these
        conversations against the rubric.
    3.  **Statistical Analysis**: The collected scores are statistically
        analyzed to determine if the application's performance meets your
        pre-defined quality bar.

    .. seealso:: :class:`~sigmaeval.ScenarioTest` for details on how to define test scenarios.

    Example:
        .. code-block:: python

            from sigmaeval import SigmaEval, ScenarioTest, AppResponse, assertions
            import asyncio

            # Define a test scenario
            scenario = (
                ScenarioTest("Bot explains its capabilities")
                .given("A new user")
                .when("The user asks about the bot's capabilities")
                .expect_behavior(
                    "Bot lists its main functions.",
                    criteria=assertions.scores.proportion_gte(min_score=7, proportion=0.90)
                )
            )

            # Define the callback to connect SigmaEval to your app
            async def app_handler(messages, state):
                # Your app logic here
                # messages is a list of dicts, e.g., [{"role": "user", "content": "Hello"}]
                new_user_message = messages[-1]["content"]
                return AppResponse(response=f"You said: {new_user_message}", state={})

            # Initialize SigmaEval and run the evaluation
            async def main():
                sigma_eval = SigmaEval(
                    judge_model="openai/gpt-4o",
                    significance_level=0.05,
                    sample_size=30
                )
                results = await sigma_eval.evaluate(scenario, app_handler)
                print(results)

            if __name__ == "__main__":
                asyncio.run(main())
    """

    def __init__(
        self,
        judge_model: str,
        significance_level: float | None = None,
        sample_size: int | None = None,
        user_simulator_model: str | None = None,
        log_level: int = logging.INFO,
        retry_config: RetryConfig | None = None,
        writing_style_config: WritingStyleConfig | None = None,
    ):
        """
        Initializes the SigmaEval framework.

        Args:
            judge_model: The fully-qualified model identifier for the Judge LLM
                and rubric generator (e.g., "openai/gpt-4o", "anthropic/claude-3-opus").
                This model is responsible for generating scoring rubrics and
                evaluating conversations. The application under test can use
                any model, as it is decoupled from the judge.
            significance_level: The default significance level (alpha) for all
                statistical tests. This value represents the probability of a
                Type I error (false positive). It can be overridden on a
                per-assertion basis. If not provided here, it *must* be provided
                in every assertion.
            sample_size: The default number of conversations to simulate for each
                :class:`~sigmaeval.ScenarioTest`. A larger sample size provides more
                statistical power. This can be overridden on a per-scenario basis.
            user_simulator_model: The model identifier for the User Simulator
                LLM. If ``None``, the ``judge_model`` will be used for all roles.
            log_level: The logging level for the 'sigmaeval' logger. Use
                ``logging.DEBUG`` for detailed output, including prompts and
                LLM reasoning.
            retry_config: Configuration for retrying failed LLM calls. If
                ``None``, default settings are used.
            writing_style_config: Configuration for the user simulator's
                writing style variations. Enabled by default. To disable,
                pass ``WritingStyleConfig(enabled=False)``.

        Raises:
            ValueError: If ``judge_model`` is not a valid, non-empty string.

        Note:
            SigmaEval uses the `LiteLLM`_ library to interface with various
            LLM providers. Ensure the necessary API keys are set as environment
            variables for your chosen models.

            .. _LiteLLM: https://github.com/BerriAI/litellm
        """
        # Suppress Pydantic serializer warnings that are not actionable for the end-user.
        # These warnings are noisy and are caused by internal LiteLLM/Pydantic interactions.
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")

        if not isinstance(judge_model, str) or not judge_model.strip():
            raise ValueError(
                "judge_model must be a non-empty string, e.g., 'openai/gpt-4o'.\nFor a complete list of supported providers, refer to the LiteLLM documentation: https://docs.litellm.ai/docs/providers"
            )

        self.judge_model: str = judge_model
        self.user_simulator_model: str = user_simulator_model or judge_model
        self.logger = logging.getLogger("sigmaeval")
        self.retry_config = retry_config or RetryConfig()
        self.significance_level = significance_level
        self.sample_size = sample_size
        self.writing_style_config = writing_style_config or WritingStyleConfig()

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(log_level)

    async def _evaluate_single(
        self,
        scenario: ScenarioTest,
        app_handler: AppHandler,
        concurrency: int = 10,
    ) -> ScenarioTestResult:
        """
        Run evaluation for a single behavioral test case.

        Args:
            scenario: The behavioral test case to evaluate
            app_handler: Async callback that takes a list of messages and a state
                object, and returns an AppResponse. The state object is managed
                by your application. On the first turn, state will be an empty
                dict. Use the state to track conversation history, user context,
                or any other stateful information your app needs.
            concurrency: Number of evaluations to run concurrently (default: 10)

        Returns:
            EvaluationResult: A data class containing the evaluation results.

        Raises:
            LLMCommunicationError: If any LLM call (rubric generation, user simulation,
                or judging) fails or returns an invalid/malformed response.
        """
        # Finalize the build and trigger validation
        scenario._finalize_build()

        self.logger.info(f"--- Starting evaluation for ScenarioTest: {scenario.title} ---")

        # Validate significance_level before expensive operations
        if self.significance_level is None:
            self.logger.debug("No global significance_level set. Verifying on each assertion.")
            for expectation in scenario.then:
                # Get a descriptive name for the expectation for the error message
                about_str = "Unknown"
                if expectation.expected_behavior:
                    about_str = expectation.label or f"'{expectation.expected_behavior[:50]}...'"
                elif expectation.metric_definition:
                    about_str = expectation.label or expectation.metric_definition.name

                for criteria in expectation.criteria:
                    if criteria.significance_level is None:
                        raise ValueError(
                            f"Expectation '{about_str}' is missing a significance_level. "
                            "A significance_level must be provided either in the SigmaEval "
                            "constructor or in every assertion when no default is set."
                        )
                    else:
                        self.logger.debug(
                            f"Expectation '{about_str}' uses assertion-specific "
                            f"significance_level: {criteria.significance_level}"
                        )
        else:
            self.logger.debug(
                f"Global significance_level set to {self.significance_level}. "
                "This will be used as a fallback for assertions."
            )

        # Validate and determine sample_size
        sample_size = scenario.num_samples or self.sample_size
        if sample_size is None or sample_size <= 0:
            raise ValueError(
                f"ScenarioTest '{scenario.title}' is missing a sample_size. "
                "A sample_size must be provided either in the SigmaEval "
                "constructor or in the ScenarioTest."
            )
        self.logger.debug(f"Using sample_size: {sample_size} for '{scenario.title}'")

        # Phase 2 (first half): Data Collection via Simulation
        # This is done only once per ScenarioTest, regardless of how many
        # expectations are in the `then` clause.
        self.logger.info(f"Simulating {sample_size} conversations for '{scenario.title}'...")
        conversations = await _collect_conversations(
            scenario=scenario,
            app_handler=app_handler,
            user_simulator_model=self.user_simulator_model,
            sample_size=sample_size,
            concurrency=concurrency,
            max_turns=scenario.max_turns_value,
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
            about_str = "Unknown expectation"

            if expectation.expected_behavior is not None:
                about_str = expectation.label or expectation.expected_behavior[:50]
                # Phase 1: Test Setup
                # Generate a rubric for this specific expectation
                self.logger.debug(f"Generating rubric for expectation: {about_str}")
                rubric = await _generate_rubric(
                    scenario=scenario,
                    expectation=expectation,
                    model=self.judge_model,
                    retry_config=self.retry_config,
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

                criteria_list = (
                    expectation.criteria
                    if isinstance(expectation.criteria, list)
                    else [expectation.criteria]
                )
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

                    assertion_about_str = "Unknown assertion"
                    if isinstance(criteria, ProportionAssertion):
                        assertion_about_str = f"proportion of scores {criteria.comparison} {criteria.proportion} (threshold: {criteria.threshold})"
                    elif isinstance(criteria, MedianAssertion):
                        assertion_about_str = (
                            f"median score {criteria.comparison} {criteria.threshold}"
                        )

                    assertion_results.append(
                        AssertionResult(
                            about=assertion_about_str,
                            passed=eval_result_dict["passed"],
                            p_value=eval_result_dict.get("p_value"),
                            details=eval_result_dict,
                        )
                    )

            elif expectation.metric_definition is not None:
                about_str = expectation.label or expectation.metric_definition.name
                metric = expectation.metric_definition
                # Calculate metric values for all conversations
                all_metric_values = []
                for conv in conversations:
                    all_metric_values.extend(metric(conv))

                criteria_list = (
                    expectation.criteria
                    if isinstance(expectation.criteria, list)
                    else [expectation.criteria]
                )
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
                        raise TypeError(
                            f"Unsupported criteria type for MetricExpectation: {type(criteria)}"
                        )

                    eval_result_dict = evaluator.evaluate(
                        all_metric_values, label=expectation.label
                    )

                    assertion_about_str = "Unknown assertion"
                    if isinstance(criteria, ProportionAssertion):
                        assertion_about_str = f"proportion of {metric.name}s {criteria.comparison} {criteria.proportion} (threshold: {criteria.threshold})"
                    elif isinstance(criteria, MedianAssertion):
                        assertion_about_str = (
                            f"median {metric.name} {criteria.comparison} {criteria.threshold}"
                        )

                    assertion_results.append(
                        AssertionResult(
                            about=assertion_about_str,
                            passed=eval_result_dict["passed"],
                            p_value=eval_result_dict.get("p_value"),
                            details=eval_result_dict,
                        )
                    )

            expectation_results.append(
                ExpectationResult(
                    about=about_str,
                    assertion_results=assertion_results,
                    scores=(
                        scores if expectation.expected_behavior is not None else all_metric_values
                    ),
                    reasoning=reasoning if expectation.expected_behavior is not None else [],
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
        app_handler: AppHandler,
        concurrency: int = 10,
    ) -> ScenarioTestResult | List[ScenarioTestResult]:
        """
        Runs an evaluation for one or more test scenarios.

        When a list of :class:`~sigmaeval.ScenarioTest` objects is provided,
        they are run concurrently.

        Args:
            scenarios: A single :class:`~sigmaeval.ScenarioTest` or a list of
                scenarios to evaluate.
            app_handler: An async callback that connects SigmaEval to your
                application. It receives a list of messages
                (e.g., ``[{"role": "user", "content": "..."}]``) and a state
                object (``Any``), and can return a ``str``, a ``tuple`` of
                ``(str, Any)``, or an :class:`~sigmaeval.AppResponse`. The
                state object is managed by your application; SigmaEval passes
                it back unmodified on subsequent turns. On the first turn of
                a conversation, the state will be an empty dictionary.
            concurrency: The number of simulated conversations to run in
                parallel for each test scenario. This controls the concurrency
                *within* a single test, not the number of tests run in parallel.

        Returns:
            - A single :class:`~sigmaeval.ScenarioTestResult` if one scenario was provided.
            - A ``list`` of :class:`~sigmaeval.ScenarioTestResult` objects if a list
              of scenarios was provided.

        Raises:
            LLMCommunicationError: If any LLM call (rubric generation, user
                simulation, or judging) fails after all retries.
            ValueError: If a ``significance_level`` or ``sample_size`` is not
                properly configured at either the framework or scenario level.
        """
        is_single_item = False
        if isinstance(scenarios, ScenarioTest):
            scenarios = [scenarios]
            is_single_item = True
        else:
            self.logger.info(
                f"--- Starting evaluation for test suite with {len(scenarios)} scenarios ---"
            )

        tasks = [
            self._evaluate_single(scenario, app_handler, concurrency) for scenario in scenarios
        ]
        all_results = await asyncio.gather(*tasks)

        if not is_single_item:
            self.logger.info("--- Test suite evaluation complete ---")

        if is_single_item:
            return all_results[0]

        return all_results
