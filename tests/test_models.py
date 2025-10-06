import pytest
from pydantic import ValidationError

from sigmaeval.core.models import ScenarioTest, BehavioralExpectation
from sigmaeval.assertions import Assertion


class MockAssertion(Assertion):
    def __call__(self, scores: list[float]) -> dict:
        return {"passed": True}


def test_scenario_test_valid():
    """Tests that a valid ScenarioTest model can be created."""
    scenario = ScenarioTest(
        title="Test Scenario",
        given="A user",
        when="They do something",
        then=BehavioralExpectation(
            expected_behavior="Something happens",
            criteria=MockAssertion(),
        ),
        sample_size=10,
    )
    assert scenario.title == "Test Scenario"
    assert scenario.sample_size == 10
    assert isinstance(scenario.then[0], BehavioralExpectation)


def test_scenario_test_invalid_sample_size():
    """Tests that a non-positive sample_size raises a ValidationError."""
    with pytest.raises(ValidationError, match="sample_size must be a positive integer"):
        ScenarioTest(
            title="Test",
            given="Given",
            when="When",
            then=BehavioralExpectation(
                expected_behavior="Then", criteria=MockAssertion()
            ),
            sample_size=0,
        )


def test_scenario_test_invalid_max_turns():
    """Tests that a non-positive max_turns raises a ValidationError."""
    with pytest.raises(ValidationError, match="max_turns must be a positive integer"):
        ScenarioTest(
            title="Test",
            given="Given",
            when="When",
            then=BehavioralExpectation(
                expected_behavior="Then", criteria=MockAssertion()
            ),
            sample_size=1,
            max_turns=0,
        )


@pytest.mark.parametrize(
    "field",
    ["title", "given", "when"],
)
def test_scenario_test_empty_string_fields(field: str):
    """Tests that empty or whitespace-only string fields raise a ValidationError."""
    with pytest.raises(ValidationError, match="string fields must not be empty"):
        kwargs = {
            "title": "Test",
            "given": "Given",
            "when": "When",
            "then": BehavioralExpectation(
                expected_behavior="Then", criteria=MockAssertion()
            ),
            "sample_size": 1,
        }
        kwargs[field] = " "
        ScenarioTest(**kwargs)


def test_scenario_test_empty_then_list():
    """Tests that an empty list for the 'then' clause raises a ValidationError."""
    with pytest.raises(ValidationError, match="'then' clause cannot be an empty list"):
        ScenarioTest(
            title="Test",
            given="Given",
            when="When",
            then=[],
            sample_size=1,
        )


def test_behavioral_expectation_empty_criteria_list():
    """Tests that an empty list for 'criteria' in BehavioralExpectation raises a ValidationError."""
    with pytest.raises(ValidationError, match="'criteria' cannot be an empty list"):
        BehavioralExpectation(
            expected_behavior="Test behavior",
            criteria=[],
        )


def test_metric_expectation_empty_criteria_list():
    """Tests that an empty list for 'criteria' in MetricExpectation raises a ValidationError."""
    from sigmaeval.core.models import MetricExpectation, Metric

    with pytest.raises(ValidationError, match="'criteria' cannot be an empty list"):
        MetricExpectation(
            metric=Metric(name="test", scope="per_turn", calculator=lambda conv: [1.0]),
            criteria=[],
        )


def test_conversation_record_add_messages():
    """Tests adding user and assistant messages to a ConversationRecord."""
    from sigmaeval.core.models import ConversationRecord
    from datetime import datetime

    convo = ConversationRecord()
    ts1 = datetime.now()
    ts2 = datetime.now()
    convo.add_user_message("Hello", ts1, ts2)
    convo.add_assistant_message("Hi", ts1, ts2)

    assert len(convo.turns) == 2
    assert convo.turns[0].role == "user"
    assert convo.turns[0].content == "Hello"
    assert convo.turns[1].role == "assistant"
    assert convo.turns[1].content == "Hi"


def test_conversation_record_to_formatted_string():
    """Tests the to_formatted_string method of ConversationRecord."""
    from sigmaeval.core.models import ConversationRecord
    from datetime import datetime

    convo = ConversationRecord()
    ts = datetime.now()
    convo.add_user_message("User message", ts, ts)
    convo.add_assistant_message("Assistant message", ts, ts)

    expected = "User: User message\n\nAssistant: Assistant message"
    assert convo.to_formatted_string() == expected


def test_conversation_record_to_detailed_string():
    """Tests the to_detailed_string method of ConversationRecord."""
    from sigmaeval.core.models import ConversationRecord
    from datetime import datetime, timedelta

    convo = ConversationRecord()
    ts1 = datetime.now()
    ts2 = ts1 + timedelta(seconds=1.23)
    convo.add_user_message("User message", ts1, ts2)

    expected = f"[{ts1.isoformat()}](1.23s) User: User message"
    assert convo.to_detailed_string() == expected


def test_assertion_result_str():
    """Tests the __str__ method of AssertionResult."""
    from sigmaeval.core.models import AssertionResult

    passed_result = AssertionResult(about="Test", passed=True, p_value=0.01)
    failed_result = AssertionResult(about="Test", passed=False)

    assert str(passed_result) == "[✅ PASSED] Test, p-value: 0.0100"
    assert str(failed_result) == "[❌ FAILED] Test"


def test_expectation_result_str_single_assertion():
    """Tests the __str__ method of ExpectationResult with a single assertion."""
    from sigmaeval.core.models import ExpectationResult, AssertionResult

    # Single passed assertion
    result = ExpectationResult(
        about="Expectation",
        assertion_results=[AssertionResult(about="Assertion", passed=True, p_value=0.01)],
    )
    assert str(result) == "[✅ PASSED] Expectation, p-value: 0.0100"

    # Single failed assertion
    result.assertion_results[0].passed = False
    assert str(result) == "[❌ FAILED] Expectation, p-value: 0.0100"


def test_expectation_result_str_multiple_assertions():
    """Tests the __str__ method of ExpectationResult with multiple assertions."""
    from sigmaeval.core.models import ExpectationResult, AssertionResult

    result = ExpectationResult(
        about="Expectation",
        assertion_results=[
            AssertionResult(about="Assertion 1", passed=True, p_value=0.01),
            AssertionResult(about="Assertion 2", passed=False),
        ],
    )
    expected = (
        "Expectation: 'Expectation' -> ❌ FAILED\n"
        "    - [✅] Assertion 1, p-value: 0.0100\n"
        "    - [❌] Assertion 2"
    )
    assert str(result) == expected


def test_scenario_test_result_str():
    """Tests the __str__ method of ScenarioTestResult."""
    from sigmaeval.core.models import (
        ScenarioTestResult,
        ExpectationResult,
        AssertionResult,
        RetryConfig,
    )

    result = ScenarioTestResult(
        title="Scenario",
        expectation_results=[
            ExpectationResult(
                about="Expectation 1",
                assertion_results=[
                    AssertionResult(about="Assertion", passed=True, p_value=0.01)
                ],
            )
        ],
        conversations=[],
        significance_level=0.05,
        judge_model="test_judge",
        user_simulator_model="test_simulator",
        retry_config=RetryConfig(),
    )

    expected = (
        "--- Result for Scenario: 'Scenario' ---\n"
        "Overall Status: ✅ PASSED\n"
        "Summary: 1/1 expectations passed.\n\n"
        "Breakdown:\n"
        "  - [✅ PASSED] Expectation 1, p-value: 0.0100"
    )
    assert str(result) == expected
