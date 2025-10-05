"""
Data models for the SigmaEval core package.
"""

import numpy as np
from typing import Any, Dict, List
from pydantic import BaseModel, Field, field_validator


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
    Defines the expected outcome and evaluation method for a behavioral test case.
    
    Attributes:
        expected_behavior: Description of the expected behavior (passed to Judge LLM)
        evaluator: Statistical evaluator to assess the results
    """
    expected_behavior: str = Field(..., description="Expected behavior description")
    evaluator: Any = Field(..., description="Evaluator instance for statistical analysis")


class ScenarioTest(BaseModel):
    """
    Defines a test case for a specific behavior of an AI application.
    """

    title: str
    given: str
    when: str
    then: "Expectation"
    sample_size: int
    max_turns: int = 10

    @field_validator("sample_size")
    def validate_sample_size(cls, v):
        if v <= 0:
            raise ValueError("sample_size must be a positive integer")
        return v

    @field_validator("max_turns")
    def validate_max_turns(cls, v):
        if v <= 0:
            raise ValueError("max_turns must be a positive integer")
        return v

    @field_validator("title", "given", "when")
    def validate_non_empty_strings(cls, v):
        if not v or not v.strip():
            raise ValueError("string fields must not be empty")
        return v


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
        turns: List of conversation turns, each a dict with 'role' and 'content'
        writing_style: The writing style used for this conversation, if any.
    """
    turns: list[Dict[str, str]] = Field(default_factory=list)
    writing_style: Dict[str, str] | None = None

    def add_user_message(self, message: str):
        """Add a user message to the conversation."""
        self.turns.append({"role": "user", "content": message})
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the conversation."""
        self.turns.append({"role": "assistant", "content": message})
    
    def to_formatted_string(self) -> str:
        """
        Format the conversation as a human-readable string.
        
        Returns:
            A string with each turn formatted as "User: ..." or "Assistant: ..."
        """
        lines = []
        for turn in self.turns:
            if turn["role"] == "user":
                lines.append(f"User: {turn['content']}")
            else:
                lines.append(f"Assistant: {turn['content']}")
        return "\n\n".join(lines)


class EvaluationResult(BaseModel):
    """
    Structured data class for the results of a single `ScenarioTest` evaluation.

    This class not only stores the raw data from the evaluation but also provides
    properties and methods for easier analysis and interpretation of the results.

    Attributes:
        judge_model: The model identifier used for the judge.
        user_simulator_model: The model identifier used for the user simulator.
        test_config: The configuration of the behavioral test.
        retry_config: The retry configuration used for the evaluation.
        rubric: The rubric used by the Judge LLM to score the interaction.
        scores: A list of scores (1-10) from the Judge LLM for each interaction.
        reasoning: A list of the Judge LLM's reasoning for each score.
        conversations: A list of all conversation transcripts.
        num_conversations: The total number of conversations (i.e., the sample size).
        results: The final statistical analysis results from the evaluator.
    """
    judge_model: str
    user_simulator_model: str
    test_config: Dict[str, Any]
    retry_config: "RetryConfig"
    rubric: str
    scores: list[float]
    reasoning: list[str]
    conversations: list[ConversationRecord]
    num_conversations: int
    results: Dict[str, Any]

    @field_validator("scores")
    def scores_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("scores cannot be an empty list")
        return v

    @property
    def passed(self) -> bool:
        """Convenience property to check if the test passed."""
        return self.results.get("passed", False)

    @property
    def p_value(self) -> float | None:
        """Convenience property to get the p-value, if available."""
        return self.results.get("p_value")

    @property
    def average_score(self) -> float:
        """The average of all collected scores."""
        return np.mean(self.scores) if self.scores else 0

    @property
    def median_score(self) -> float:
        """The median of all collected scores."""
        return np.median(self.scores) if self.scores else 0

    @property
    def min_score(self) -> float:
        """The minimum score from the evaluation."""
        return min(self.scores) if self.scores else 0

    @property
    def max_score(self) -> float:
        """The maximum score from the evaluation."""
        return max(self.scores) if self.scores else 0

    @property
    def std_dev_score(self) -> float:
        """The standard deviation of the scores."""
        return np.std(self.scores) if self.scores else 0

    def get_worst_conversation(self) -> tuple[float, str, ConversationRecord]:
        """
        Finds and returns the conversation with the lowest score.

        If there are multiple conversations with the same lowest score, the first
        one encountered will be returned.

        Returns:
            A tuple containing the score, the judge's reasoning, and the
            conversation record.
        """
        if not self.scores:
            raise ValueError("Cannot get worst conversation from empty scores list.")

        min_score = min(self.scores)
        min_index = self.scores.index(min_score)
        return (
            self.scores[min_index],
            self.reasoning[min_index],
            self.conversations[min_index],
        )

    def get_best_conversation(self) -> tuple[float, str, ConversationRecord]:
        """
        Finds and returns the conversation with the highest score.

        If there are multiple conversations with the same highest score, the first
        one encountered will be returned.

        Returns:
            A tuple containing the score, the judge's reasoning, and the
            conversation record.
        """
        if not self.scores:
            raise ValueError("Cannot get best conversation from empty scores list.")

        max_score = max(self.scores)
        max_index = self.scores.index(max_score)
        return (
            self.scores[max_index],
            self.reasoning[max_index],
            self.conversations[max_index],
        )

    def __str__(self) -> str:
        """
        Provides a human-readable summary of the evaluation results.
        """
        # Title and Pass/Fail status
        title = self.test_config.get("title", "Evaluation Results")
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        header = f"--- {title}: {status} ---"

        # Key stats
        p_value_str = f"P-value: {self.p_value:.4f}" if self.p_value is not None else ""
        stats = f"""
Summary Statistics:
  - Average Score: {self.average_score:.2f}
  - Median Score:  {self.median_score:.2f}
  - Min Score:     {self.min_score:.2f}
  - Max Score:     {self.max_score:.2f}
  - Std Dev:       {self.std_dev_score:.2f}
        """

        # Evaluator-specific results
        evaluator_results = "\n".join(
            [f"  - {key.replace('_', ' ').title()}: {value}" for key, value in self.results.items()]
        )
        
        return f"""
{header}
{p_value_str}

{stats.strip()}

Full Evaluator Results:
{evaluator_results}
        """.strip()


