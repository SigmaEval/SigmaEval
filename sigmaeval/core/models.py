"""
Data models for the SigmaEval core package.
"""

from typing import Any, Dict
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


class ConversationRecord(BaseModel):
    """
    Record of a single conversation between user simulator and app.
    
    This class stores the turn-by-turn interaction between the simulated user
    and the application under test.
    
    Attributes:
        turns: List of conversation turns, each a dict with 'role' and 'content'
    """
    turns: list[Dict[str, str]] = []

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
    rubric: str
    scores: list[float]
    reasoning: list[str]
    conversations: list[ConversationRecord]
    num_conversations: int
    results: Dict[str, Any]


