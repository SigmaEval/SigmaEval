"""
Data models for the SigmaEval core package.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field
from dataclasses import dataclass


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
            "Single words or very short phrases.",
            "Extremely terse, uses incomplete sentences.",
            "Terse and to-the-point.",
            "Concise and clear.",
            "Moderately detailed.",
            "Slightly verbose.",
            "Verbose and descriptive.",
            "Overly detailed and rambling.",
            "Long, multi-paragraph messages.",
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


@dataclass
class BehavioralTest:
    """
    Defines a test case for a specific behavior of an AI application.
    """
    title: str
    given: str
    when: str
    then: "Expectation"
    max_turns: int = 10


@dataclass
class RetryConfig:
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
    writing_style: str | None = None

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
    Structured data class for the results of a single `BehavioralTest` evaluation.

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


