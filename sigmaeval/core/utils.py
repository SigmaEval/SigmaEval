import json
import re
from typing import Any, Dict, List
from .models import ConversationRecord, Conversation, Turn


def _convert_conversation_records(
    records: List[ConversationRecord],
) -> List[Conversation]:
    """Converts data collection records to the final Conversation model."""
    conversations = []
    for record in records:
        turns = []
        # Group turns into pairs of (user, assistant)
        for i in range(0, len(record.turns), 2):
            user_turn = record.turns[i]
            assistant_turn = record.turns[i + 1] if i + 1 < len(record.turns) else None
            
            if assistant_turn:
                latency = (assistant_turn.response_timestamp - user_turn.response_timestamp).total_seconds()
                turns.append(
                    Turn(
                        user_message=user_turn.content,
                        app_response=assistant_turn.content,
                        latency=latency,
                    )
                )
        conversations.append(Conversation(turns=turns, details={"writing_style": record.writing_style}))
    return conversations


def _extract_json_from_response(content: str) -> Dict[str, Any] | None:
    """
    Extracts a JSON object from an LLM response string.

    This function tries two methods:
    1.  Looks for a JSON object enclosed in markdown code fences (```json ... ```).
    2.  If not found, uses a non-greedy regex to find the first complete JSON object.

    Args:
        content: The raw string response from the LLM.

    Returns:
        The parsed JSON object as a dictionary, or None if no valid JSON is found.
    """
    # Method 1: Look for markdown-fenced JSON
    match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass  # Fall through to the next method

    # Method 2: Non-greedy search for the first complete JSON object
    match = re.search(r"\{", content)
    if match:
        start_index = match.start()
        open_braces = 1
        for i in range(start_index + 1, len(content)):
            if content[i] == "{":
                open_braces += 1
            elif content[i] == "}":
                open_braces -= 1
            
            if open_braces == 0:
                json_str = content[start_index : i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass  # Fall through to return None
                break  # Found a balanced set of braces

    return None
