from .core.models import Metric, ConversationRecord
from typing import List


def _calculate_response_latency(conversation: ConversationRecord) -> List[float]:
    """Calculates the response latency for each turn in a conversation."""
    latencies = []
    for turn in conversation.turns:
        if turn.role == "assistant":
            # Find the preceding user turn to calculate latency
            user_turn_index = conversation.turns.index(turn) - 1
            if user_turn_index >= 0:
                user_turn = conversation.turns[user_turn_index]
                latency = (turn.response_timestamp - user_turn.response_timestamp).total_seconds()
                latencies.append(latency)
    return latencies


def _calculate_turn_count(conversation: ConversationRecord) -> List[float]:
    """Calculates the total number of turns in a conversation."""
    return [len(conversation.turns)]


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


class Metrics:
    def __init__(self):
        self.per_turn = PerTurn()
        self.per_conversation = PerConversation()


metrics = Metrics()
