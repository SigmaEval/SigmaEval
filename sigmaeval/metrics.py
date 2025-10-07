"""
This module provides built-in metrics to measure objective, quantitative
aspects of an AI's performance.

Metrics are organized into namespaces based on their scope:
-   **Per-Turn Metrics**: Collected for each assistant response within a
    conversation.
-   **Per-Conversation Metrics**: Collected once for the entire conversation.

All metrics are available under the `sigmaeval.metrics` object.

Example:
    .. code-block:: python

        from sigmaeval import ScenarioTest, assertions, metrics

        scenario = (
            ScenarioTest("Bot is responsive")
            .given("A user asks a question")
            .when("The user is waiting for a reply")
            .expect_metric(
                metrics.per_turn.response_latency,
                criteria=assertions.metrics.proportion_lt(threshold=1.5, proportion=0.95)
            )
            .expect_metric(
                metrics.per_conversation.turn_count,
                criteria=assertions.metrics.median_lt(threshold=4.0)
            )
        )
"""

from .core.models import MetricDefinition, ConversationRecord
from typing import List


def _calculate_response_latency(conversation: ConversationRecord) -> List[float]:
    """Calculates the response latency for each assistant turn in a conversation."""
    latencies = []
    assistant_turns = [turn for turn in conversation.turns if turn.role == "assistant"]
    for turn in assistant_turns:
        latencies.append((turn.response_timestamp - turn.request_timestamp).total_seconds())
    return latencies


def _calculate_response_length_chars(conversation: ConversationRecord) -> List[float]:
    """Calculates the length of each assistant response in characters."""
    return [float(len(turn.content)) for turn in conversation.turns if turn.role == "assistant"]


def _calculate_turn_count(conversation: ConversationRecord) -> List[float]:
    """Calculates the total number of assistant turns in a conversation."""
    # Returns the count of assistant responses, which is a better reflection of "turns"
    return [sum(1 for turn in conversation.turns if turn.role == "assistant")]


def _calculate_total_assistant_response_time(
    conversation: ConversationRecord,
) -> List[float]:
    """Calculates the total time the assistant spent processing responses."""
    total_time = sum(
        (turn.response_timestamp - turn.request_timestamp).total_seconds()
        for turn in conversation.turns
        if turn.role == "assistant"
    )
    return [total_time]


def _calculate_total_assistant_response_chars(
    conversation: ConversationRecord,
) -> List[float]:
    """Calculates the total characters in all assistant responses."""
    total_chars = sum(len(turn.content) for turn in conversation.turns if turn.role == "assistant")
    return [float(total_chars)]


class PerTurn:
    """Metrics that are collected for each assistant response within a conversation."""

    def __init__(self):
        self.response_latency = MetricDefinition(
            name="response_latency",
            scope="per_turn",
            calculator=_calculate_response_latency,
        )
        """
        Measures the time (in seconds) between the application receiving a
        user's message and sending its response.
        
        - **Scope**: Per-Turn
        - **Use Case**: Ensuring the application feels responsive.
        """
        self.response_length_chars = MetricDefinition(
            name="response_length_chars",
            scope="per_turn",
            calculator=_calculate_response_length_chars,
        )
        """
        The number of characters in the assistant's response.
        
        - **Scope**: Per-Turn
        - **Use Case**: Enforcing conciseness in individual responses.
        """


class PerConversation:
    """Metrics that are collected once for the entire conversation."""

    def __init__(self):
        self.turn_count = MetricDefinition(
            name="turn_count",
            scope="per_conversation",
            calculator=_calculate_turn_count,
        )
        """
        The total number of assistant responses in a conversation.
        
        - **Scope**: Per-Conversation
        - **Use Case**: Measuring the efficiency of the AI.
        """
        self.total_assistant_response_time = MetricDefinition(
            name="total_assistant_response_time",
            scope="per_conversation",
            calculator=_calculate_total_assistant_response_time,
        )
        """
        The total time (in seconds) the assistant spent processing responses
        for the entire conversation.
        
        - **Scope**: Per-Conversation
        - **Use Case**: Evaluating total computational effort.
        """
        self.total_assistant_response_chars = MetricDefinition(
            name="total_assistant_response_chars",
            scope="per_conversation",
            calculator=_calculate_total_assistant_response_chars,
        )
        """
        The total number of characters in all of the assistant's responses.
        
        - **Scope**: Per-Conversation
        - **Use Case**: Measuring the overall verbosity of the assistant.
        """


class Metrics:
    def __init__(self):
        self.per_turn = PerTurn()
        self.per_conversation = PerConversation()


metrics = Metrics()
