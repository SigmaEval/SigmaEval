import pytest
from sigmaeval.core.utils import _extract_json_from_response, _convert_conversation_records
from sigmaeval.core.models import ConversationRecord, ConversationTurn, Conversation, Turn
from datetime import datetime, timedelta, timezone


@pytest.mark.parametrize(
    "response_content, expected_output",
    [
        # Test case 1: Clean JSON
        ('{"key": "value", "number": 123}', {"key": "value", "number": 123}),
        # Test case 2: JSON with leading/trailing whitespace
        ('  {"key": "value"}  ', {"key": "value"}),
        # Test case 3: JSON within markdown code fence
        ('```json\n{"key": "value"}\n```', {"key": "value"}),
        # Test case 4: JSON with surrounding text
        ('Here is the JSON: {"key": "value"}!', {"key": "value"}),
        # Test case 5: JSON with markdown and surrounding text
        (
            'Some text before.\n```json\n{"key": "value"}\n```\nSome text after.',
            {"key": "value"},
        ),
        # Test case 6: Malformed JSON
        ('{"key": "value",}', None),
        # Test case 7: No JSON
        ("This is a string without any JSON.", None),
        # Test case 8: Empty string
        ("", None),
        # Test case 9: Non-greedy test
        ('{"a": 1}{"b": 2}', {"a": 1}),
        # Test case 10: Nested JSON
        (
            '{"outer": {"inner": "value"}}',
            {"outer": {"inner": "value"}},
        ),
    ],
)
def test_extract_json_from_response(response_content, expected_output):
    """
    Tests the _extract_json_from_response function with various inputs.
    """
    assert _extract_json_from_response(response_content) == expected_output


# --- Test cases for _convert_conversation_records ---

now = datetime.now(timezone.utc)

# Basic case with one turn and writing style
record1 = ConversationRecord(
    turns=[
        ConversationTurn(
            role="user",
            content="hello",
            request_timestamp=now,
            response_timestamp=now + timedelta(seconds=0.1),
        ),
        ConversationTurn(
            role="assistant",
            content="hi there",
            request_timestamp=now + timedelta(seconds=0.2),
            response_timestamp=now + timedelta(seconds=0.6),
        ),
    ],
    writing_style={"tone": "friendly"},
)
expected1 = Conversation(
    turns=[Turn(user_message="hello", app_response="hi there", latency=0.4)],
    details={"writing_style": {"tone": "friendly"}},
)

# Multi-turn case
record2 = ConversationRecord(
    turns=[
        ConversationTurn(
            role="user",
            content="q1",
            request_timestamp=now,
            response_timestamp=now + timedelta(seconds=0.1),
        ),
        ConversationTurn(
            role="assistant",
            content="a1",
            request_timestamp=now + timedelta(seconds=0.2),
            response_timestamp=now + timedelta(seconds=0.8),
        ),
        ConversationTurn(
            role="user",
            content="q2",
            request_timestamp=now + timedelta(seconds=1),
            response_timestamp=now + timedelta(seconds=1.1),
        ),
        ConversationTurn(
            role="assistant",
            content="a2",
            request_timestamp=now + timedelta(seconds=1.2),
            response_timestamp=now + timedelta(seconds=2.0),
        ),
    ]
)
expected2 = Conversation(
    turns=[
        Turn(user_message="q1", app_response="a1", latency=0.6),
        Turn(user_message="q2", app_response="a2", latency=0.8),
    ],
    details={"writing_style": None},
)

# Record with an odd number of turns (last user message is ignored)
record3 = ConversationRecord(
    turns=[
        ConversationTurn(
            role="user", content="ping", request_timestamp=now, response_timestamp=now
        ),
        ConversationTurn(
            role="assistant",
            content="pong",
            request_timestamp=now,
            response_timestamp=now + timedelta(seconds=1),
        ),
        ConversationTurn(
            role="user", content="hanging", request_timestamp=now, response_timestamp=now
        ),
    ]
)
expected3 = Conversation(
    turns=[Turn(user_message="ping", app_response="pong", latency=1.0)],
    details={"writing_style": None},
)


@pytest.mark.parametrize(
    "input_records, expected_output",
    [
        # Test case 1: Basic conversion
        ([record1], [expected1]),
        # Test case 2: Multi-turn conversion
        ([record2], [expected2]),
        # Test case 3: List with multiple records
        ([record1, record2], [expected1, expected2]),
        # Test case 4: Edge case - empty list of records
        ([], []),
        # Test case 5: Edge case - record with no turns
        ([ConversationRecord(turns=[])], [Conversation(turns=[], details={"writing_style": None})]),
        # Test case 6: Edge case - odd number of turns
        ([record3], [expected3]),
    ],
)
def test_convert_conversation_records(input_records, expected_output):
    """
    Tests the _convert_conversation_records function with various inputs.
    """
    result = _convert_conversation_records(input_records)
    # Pydantic models can be compared directly for equality
    assert result == expected_output
