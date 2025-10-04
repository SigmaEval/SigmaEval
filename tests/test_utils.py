import pytest
from sigmaeval.core.utils import _extract_json_from_response


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
