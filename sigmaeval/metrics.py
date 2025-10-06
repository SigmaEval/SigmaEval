from .core.models import Metric, ConversationRecord
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


class PerTurn:
    def __init__(self):
        self.response_latency = Metric(
            name="response_latency",
            scope="per_turn",
            calculator=_calculate_response_latency,
        )


class PerConversation:
    def __init__(self):
        self.turn_count = Metric(
            name="turn_count",
            scope="per_conversation",
            calculator=_calculate_turn_count,
        )
        self.total_assistant_response_time = Metric(
            name="total_assistant_response_time",
            scope="per_conversation",
            calculator=_calculate_total_assistant_response_time,
        )


class Metrics:
    def __init__(self):
        self.per_turn = PerTurn()
        self.per_conversation = PerConversation()


metrics = Metrics()
