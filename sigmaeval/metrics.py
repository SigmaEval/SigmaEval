from .core.models import MetricDefinition, ConversationRecord
from typing import List


def _calculate_response_latency(conversation: ConversationRecord) -> List[float]:
    """Calculates the response latency for each assistant turn in a conversation."""
    latencies = []
    assistant_turns = [
        turn for turn in conversation.turns if turn.role == "assistant"
    ]
    for turn in assistant_turns:
        latencies.append(
            (turn.response_timestamp - turn.request_timestamp).total_seconds()
        )
    return latencies


def _calculate_response_length_chars(conversation: ConversationRecord) -> List[float]:
    """Calculates the length of each assistant response in characters."""
    return [
        float(len(turn.content))
        for turn in conversation.turns
        if turn.role == "assistant"
    ]


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
    total_chars = sum(
        len(turn.content)
        for turn in conversation.turns
        if turn.role == "assistant"
    )
    return [float(total_chars)]


class PerTurn:
    def __init__(self):
        self.response_latency = MetricDefinition(
            name="response_latency",
            scope="per_turn",
            calculator=_calculate_response_latency,
        )
        self.response_length_chars = MetricDefinition(
            name="response_length_chars",
            scope="per_turn",
            calculator=_calculate_response_length_chars,
        )


class PerConversation:
    def __init__(self):
        self.turn_count = MetricDefinition(
            name="turn_count",
            scope="per_conversation",
            calculator=_calculate_turn_count,
        )
        self.total_assistant_response_time = MetricDefinition(
            name="total_assistant_response_time",
            scope="per_conversation",
            calculator=_calculate_total_assistant_response_time,
        )
        self.total_assistant_response_chars = MetricDefinition(
            name="total_assistant_response_chars",
            scope="per_conversation",
            calculator=_calculate_total_assistant_response_chars,
        )


class Metrics:
    def __init__(self):
        self.per_turn = PerTurn()
        self.per_conversation = PerConversation()


metrics = Metrics()
