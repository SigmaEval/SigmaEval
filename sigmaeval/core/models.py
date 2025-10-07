"""
Data models for the SigmaEval core package.
"""

import numpy as np
from typing import Any, Dict, List, Union, Optional, Callable
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError, PrivateAttr
from datetime import datetime

from ..assertions import Assertion, ScoreAssertion, MetricAssertion


class MetricDefinition(BaseModel):
    name: str
    scope: str  # "per_turn" or "per_conversation"
    # The calculator function will take a conversation and return a list of values
    calculator: Callable[["ConversationRecord"], List[float]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __call__(self, conversation: "ConversationRecord") -> List[float]:
        return self.calculator(conversation)


class ConversationTurn(BaseModel):
    """A single turn in a conversation, with timestamps."""
    role: str
    content: str
    request_timestamp: datetime
    response_timestamp: datetime


class WritingStyleAxes(BaseModel):
    """
    Defines the axes for writing style variations.
    """
    proficiency: List[str] = Field(
        default=[
            "Third-grade level.",
            "Fifth-grade level.",
            "Middle-school level.",
            "Frequent grammar and spelling errors.",
            "Clear with some minor errors.",
            "High-school level.",
            "Good grammar and vocabulary.",
            "University-graduate level.",
            "Flawless grammar and sophisticated vocabulary.",
        ]
    )
    tone: List[str] = Field(
        default=[
            "Enthusiastic and very friendly.",
            "Polite and friendly.",
            "Curious and inquisitive.",
            "Formal and professional.",
            "Direct and neutral.",
            "Skeptical and questioning.",
            "Slightly confused.",
            "Impatient and slightly frustrated.",
            "Annoyed and critical.",
        ]
    )
    verbosity: List[str] = Field(
        default=[
            "Very terse (1-5 words).",
            "Terse (5-10 words).",
            "Concise (10-20 words).",
            "Moderately detailed (20-40 words).",
            "Detailed (40-80 words).",
            "Verbose (80-120 words).",
            "Very verbose (120-180 words).",
            "Extremely verbose (180-250 words).",
            "Rambling and overly detailed (250+ words).",
        ]
    )
    formality: List[str] = Field(
        default=[
            "Extremely formal, almost academic.",
            "Formal and professional.",
            "Slightly formal.",
            "Neutral.",
            "Slightly informal.",
            "Casual and conversational.",
            "Very casual, uses slang and abbreviations.",
            "Uses internet slang and emojis.",
            "Extremely informal, uses memespeak or textspeak.",
        ]
    )


class WritingStyleConfig(BaseModel):
    """
    Configuration for user simulator writing style variations.
    """
    enabled: bool = True
    axes: WritingStyleAxes = Field(default_factory=WritingStyleAxes)


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
    Defines the expected outcome and evaluation method for a scenario test case.
    
    Use the factory methods to create instances:
    - Expectation.behavior() for LLM-judged behavioral expectations
    - Expectation.metric() for objective metric-based expectations
    
    Attributes:
        expected_behavior: Description of the expected behavior (passed to Judge LLM)
        metric_definition: The metric to be measured (e.g., response_latency).
        criteria: Statistical criteria to assess the results
        label: An optional short name for the expectation, which will be displayed in logs and the evaluation results summary.
    """
    expected_behavior: Optional[str] = Field(None, description="Expected behavior description")
    metric_definition: Optional[MetricDefinition] = Field(None, description="The metric to be measured.")
    criteria: List[Assertion] = Field(..., description="Criteria for statistical analysis")
    label: Optional[str] = Field(None, description="Optional short name for the expectation, which will be displayed in logs and the evaluation results summary.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @model_validator(mode="before")
    @classmethod
    def check_behavior_or_metric(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if (
                data.get("expected_behavior") is not None
                and data.get("metric_definition") is not None
            ):
                raise ValueError(
                    "An Expectation cannot have both 'expected_behavior' and 'metric_definition' defined."
                )
            if (
                data.get("expected_behavior") is None
                and data.get("metric_definition") is None
            ):
                raise ValueError(
                    "An Expectation must have either 'expected_behavior' or 'metric_definition' defined."
                )
        return data

    @field_validator("criteria")
    def validate_criteria(cls, v):
        if not v:
            raise ValueError("'criteria' cannot be an empty list")
        return v
    
    @classmethod
    def behavior(
        cls,
        expected_behavior: str,
        criteria: Union[ScoreAssertion, List[ScoreAssertion]],
        label: Optional[str] = None,
    ) -> "Expectation":
        """
        Creates a behavioral expectation, which is evaluated by an LLM judge.
        
        Args:
            expected_behavior: A description of the desired behavior.
            criteria: A single or list of statistical assertions to run on the judge's scores.
            label: An optional short name for this expectation.
        """
        criteria_list = criteria if isinstance(criteria, list) else [criteria]
        return cls(
            expected_behavior=expected_behavior, criteria=criteria_list, label=label
        )

    @classmethod
    def metric(
        cls,
        metric: MetricDefinition,
        criteria: Union[MetricAssertion, List[MetricAssertion]],
        label: Optional[str] = None,
    ) -> "Expectation":
        """
        Creates a metric-based expectation, which is evaluated on objective data.
        
        Args:
            metric: The metric to measure.
            criteria: A single or list of statistical assertions to run on the metric data.
            label: An optional short name for this expectation.
        """
        criteria_list = criteria if isinstance(criteria, list) else [criteria]
        return cls(metric_definition=metric, criteria=criteria_list, label=label)


class ScenarioTest(BaseModel):
    """
    Defines a test case for a specific behavior of an AI application.
    
    This class uses a fluent builder API for improved readability and discoverability.
    
    Example:
        scenario = (
            ScenarioTest("Bot explains its capabilities")
            .given("A new user who has not interacted with the bot before")
            .when("The user asks a general question about the bot's capabilities")
            .sample_size(30)
            .expect_behavior(
                "Bot lists its main functions: tracking orders, initiating returns, etc.",
                criteria=assertions.scores.proportion_gte(min_score=6, proportion=0.90)
            )
            .expect_metric(
                metrics.per_turn.response_latency,
                criteria=assertions.metrics.proportion_lt(threshold=1.0, proportion=0.90)
            )
        )
    """

    title: str
    given_context: str = Field(default="", serialization_alias="given", validation_alias="given")
    when_action: str = Field(default="", serialization_alias="when", validation_alias="when")
    then: List[Expectation] = Field(default_factory=list)
    num_samples: int | None = Field(default=None, serialization_alias="sample_size", validation_alias="sample_size")
    max_turns_value: int = Field(default=10, serialization_alias="max_turns", validation_alias="max_turns")
    
    # Use Pydantic private attributes
    _building: bool = PrivateAttr(default=True)

    model_config = ConfigDict(validate_assignment=True, populate_by_name=True)

    def __init__(self, title: str, **kwargs):
        """
        Initialize a ScenarioTest with a title.
        
        Args:
            title: The title/name of the test scenario
        """
        if not title or not title.strip():
            raise ValueError("title must not be empty")
        super().__init__(title=title, **kwargs)
        # Private attribute is automatically set to True by default
    
    def given(self, context: str) -> "ScenarioTest":
        """
        Set the 'Given' context for the test scenario.
        
        Args:
            context: The prerequisite state and context for the user simulator
            
        Returns:
            Self for method chaining
        """
        if not context or not context.strip():
            raise ValueError("'given' context must not be empty")
        self.given_context = context
        return self
    
    def when(self, action: str) -> "ScenarioTest":
        """
        Set the 'When' action/goal for the test scenario.
        
        Args:
            action: The specific goal or action the user simulator will try to achieve
            
        Returns:
            Self for method chaining
        """
        if not action or not action.strip():
            raise ValueError("'when' action must not be empty")
        self.when_action = action
        return self
    
    def sample_size(self, size: int) -> "ScenarioTest":
        """
        Set the sample size for the test scenario.
        
        Args:
            size: The number of conversations to simulate
            
        Returns:
            Self for method chaining
        """
        if size <= 0:
            raise ValueError("sample_size must be a positive integer")
        self.num_samples = size
        return self
    
    def max_turns(self, turns: int) -> "ScenarioTest":
        """
        Set the maximum number of turns per conversation.
        
        Args:
            turns: The maximum number of turns allowed in each conversation
            
        Returns:
            Self for method chaining
        """
        if turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        self.max_turns_value = turns
        return self
    
    def expect_behavior(
        self,
        expected_behavior: str,
        criteria: Union[ScoreAssertion, List[ScoreAssertion]],
        label: Optional[str] = None,
    ) -> "ScenarioTest":
        """
        Add a behavioral expectation to be evaluated by an LLM judge.
        
        Args:
            expected_behavior: A description of the desired behavior
            criteria: A single or list of statistical assertions to run on the judge's scores
            label: An optional short name for this expectation
            
        Returns:
            Self for method chaining
        """
        criteria_list = criteria if isinstance(criteria, list) else [criteria]
        expectation = Expectation(
            expected_behavior=expected_behavior,
            criteria=criteria_list,
            label=label
        )
        # Get the current then list and append to it
        self.__dict__["then"].append(expectation)
        return self
    
    def expect_metric(
        self,
        metric: MetricDefinition,
        criteria: Union[MetricAssertion, List[MetricAssertion]],
        label: Optional[str] = None,
    ) -> "ScenarioTest":
        """
        Add a metric-based expectation to be evaluated on objective data.
        
        Args:
            metric: The metric to measure
            criteria: A single or list of statistical assertions to run on the metric data
            label: An optional short name for this expectation
            
        Returns:
            Self for method chaining
        """
        criteria_list = criteria if isinstance(criteria, list) else [criteria]
        expectation = Expectation(
            metric_definition=metric,
            criteria=criteria_list,
            label=label
        )
        # Get the current then list and append to it
        self.__dict__["then"].append(expectation)
        return self
    
    def _finalize_build(self) -> None:
        """
        Internal method to finalize the build and trigger validation.
        Called by the framework before using the ScenarioTest.
        """
        self._building = False
        # Now trigger validation by re-validating the model
        self.model_validate(self)
    
    @model_validator(mode="after")
    def validate_complete(self) -> "ScenarioTest":
        """
        Validate that all required fields are set before the test can be executed.
        This validation is skipped during builder pattern construction.
        """
        # Skip validation if we're still in builder mode
        if self._building:
            return self
        
        errors = []
        
        # Access the fields
        given_value = self.given_context
        when_value = self.when_action
        sample_size_value = self.num_samples
        then_value = self.then
        
        if not given_value or not given_value.strip():
            errors.append("'given' context must be set using .given()")
        
        if not when_value or not when_value.strip():
            errors.append("'when' action must be set using .when()")
        
        # The check for a missing sample_size is done in SigmaEval to allow for a
        # global default.
        if sample_size_value is not None and sample_size_value <= 0:
            errors.append("sample_size must be a positive integer")
        
        if not then_value:
            errors.append("at least one expectation must be added using .expect_behavior() or .expect_metric()")
        
        if errors:
            raise ValueError(
                "ScenarioTest is incomplete. Missing required configuration:\n  - "
                + "\n  - ".join(errors)
            )
        
        return self


class RetryConfig(BaseModel):
    """
    Configuration for Tenacity retry behavior used for LiteLLM calls.

    Set enabled=False or max_attempts<=1 to disable retries.
    """

    enabled: bool = True
    max_attempts: int = 5
    backoff_multiplier: float = 0.5
    max_backoff_seconds: float = 30.0


class ConversationRecord(BaseModel):
    """
    Record of a single conversation between user simulator and app.
    
    This class stores the turn-by-turn interaction between the simulated user
    and the application under test.
    
    Attributes:
        turns: List of conversation turns.
        writing_style: The writing style used for this conversation, if any.
    """
    turns: list[ConversationTurn] = Field(default_factory=list)
    writing_style: Dict[str, str] | None = None

    def add_user_message(
        self,
        message: str,
        request_timestamp: datetime,
        response_timestamp: datetime
    ):
        """Add a user message to the conversation."""
        self.turns.append(ConversationTurn(
            role="user",
            content=message,
            request_timestamp=request_timestamp,
            response_timestamp=response_timestamp,
        ))
    
    def add_assistant_message(
        self,
        message: str,
        request_timestamp: datetime,
        response_timestamp: datetime
    ):
        """Add an assistant message to the conversation."""
        self.turns.append(ConversationTurn(
            role="assistant",
            content=message,
            request_timestamp=request_timestamp,
            response_timestamp=response_timestamp,
        ))
    
    def to_formatted_string(self) -> str:
        """
        Format the conversation as a human-readable string.
        
        Returns:
            A string with each turn formatted as "User: ..." or "Assistant: ..."
        """
        lines = []
        for turn in self.turns:
            if turn.role == "user":
                lines.append(f"User: {turn.content}")
            else:
                lines.append(f"Assistant: {turn.content}")
        return "\n\n".join(lines)

    def to_detailed_string(self) -> str:
        """
        Format the conversation as a detailed, human-readable string with timestamps
        and turn durations.
        """
        lines = []
        for turn in self.turns:
            duration = (turn.response_timestamp - turn.request_timestamp).total_seconds()
            lines.append(
                f"[{turn.request_timestamp.isoformat()}]"
                f"({duration:.2f}s) {turn.role.capitalize()}: {turn.content}"
            )
        return "\n".join(lines)


class Turn(BaseModel):
    """Represents a single turn in a conversation."""

    user_message: str
    app_response: str
    latency: float
    details: Dict[str, Any] = Field(default_factory=dict)


class Conversation(BaseModel):
    """Represents a full conversation from a single simulation run."""

    turns: List[Turn]
    details: Dict[str, Any] = Field(default_factory=dict)


class AssertionResult(BaseModel):
    """The result of a single assertion check."""

    about: str
    passed: bool
    p_value: Optional[float] = None
    details: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        p_value_str = f", p-value: {self.p_value:.4f}" if self.p_value is not None else ""
        return f"[{status}] {self.about}{p_value_str}"


class ExpectationResult(BaseModel):
    """The result of a single Expectation, which may contain multiple assertions."""

    about: str
    assertion_results: List[AssertionResult]
    scores: List[float] = Field(default_factory=list)
    reasoning: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only if ALL assertions for this expectation passed."""
        return all(r.passed for r in self.assertion_results)

    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        # The 'about' for the Expectation gives high-level context
        if len(self.assertion_results) == 1:
            # If there's only one assertion, condense the output to a single line
            res = self.assertion_results[0]
            res_status = "✅ PASSED" if res.passed else "❌ FAILED"
            p_value_str = (
                f", p-value: {res.p_value:.4f}" if res.p_value is not None else ""
            )
            return f"[{res_status}] {self.about}{p_value_str}"
        else:
            # For multiple assertions, use a detailed breakdown
            title_line = f"Expectation: '{self.about}' -> {status}"
            # Each assertion result is then listed
            results_breakdown = []
            for res in self.assertion_results:
                assertion_status = "✅" if res.passed else "❌"
                p_value_str = (
                    f", p-value: {res.p_value:.4f}" if res.p_value is not None else ""
                )
                results_breakdown.append(
                    f"    - [{assertion_status}] {res.about}{p_value_str}"
                )

            breakdown_str = "\n".join(results_breakdown)
            return f"{title_line}\n{breakdown_str}"


class ScenarioTestResult(BaseModel):
    """The comprehensive result of a single ScenarioTest run."""

    title: str
    expectation_results: List["ExpectationResult"]
    conversations: List["Conversation"]
    significance_level: float | None
    judge_model: str
    user_simulator_model: str
    retry_config: "RetryConfig"
    rubric: Optional[str] = None

    @property
    def passed(self) -> bool:
        """True only if ALL expectations passed."""
        return all(r.passed for r in self.expectation_results)

    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        title_line = f"--- Result for Scenario: '{self.title}' ---"
        status_line = f"Overall Status: {status}"
        passed_count = sum(1 for r in self.expectation_results if r.passed)
        total_count = len(self.expectation_results)
        summary_line = f"Summary: {passed_count}/{total_count} expectations passed."
        results_breakdown = "\n\n".join(f"  - {r}" for r in self.expectation_results)
        return (
            f"{title_line}\n{status_line}\n{summary_line}\n\nBreakdown:\n{results_breakdown}"
        )


