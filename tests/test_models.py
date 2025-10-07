import pytest
from pydantic import ValidationError

from sigmaeval.core.models import ScenarioTest, Expectation, MetricDefinition
from sigmaeval.assertions import ScoreAssertion, MetricAssertion


class MockScoreAssertion(ScoreAssertion):
    def __call__(self, scores: list[float]) -> dict:
        return {"passed": True}


class MockMetricAssertion(MetricAssertion):
    def __call__(self, scores: list[float]) -> dict:
        return {"passed": True}


def test_scenario_test_valid():
    """Tests that a valid ScenarioTest model can be created."""
    scenario = (
        ScenarioTest("Test Scenario")
        .given("A user")
        .when("They do something")
        .sample_size(10)
        .expect_behavior(
            "Something happens",
            criteria=MockScoreAssertion(),
        )
    )
    assert scenario.title == "Test Scenario"
    assert scenario.num_samples == 10
    assert isinstance(scenario.then[0], Expectation)


def test_scenario_test_fluent_api_order_independent():
    """Tests that the fluent API builder methods can be called in any order."""
    # Call builder methods in a non-standard order
    scenario = (
        ScenarioTest("Order Independent Test")
        .sample_size(50)
        .expect_metric(
            metric=MetricDefinition(
                name="test", scope="per_turn", calculator=lambda conv: [1.0]
            ),
            criteria=MockMetricAssertion(),
        )
        .given("A user who likes non-sequential building")
        .when("They build a test out of order")
        .expect_behavior("It still works", criteria=MockScoreAssertion())
    )

    # Assert that all fields were set correctly
    assert scenario.title == "Order Independent Test"
    assert scenario.num_samples == 50
    assert scenario.given_context == "A user who likes non-sequential building"
    assert scenario.when_action == "They build a test out of order"
    assert len(scenario.then) == 2
    assert scenario.then[0].metric_definition is not None
    assert scenario.then[1].expected_behavior == "It still works"

    # Finalize the build to ensure it passes validation
    try:
        scenario._finalize_build()
    except ValidationError as e:
        pytest.fail(f"Validation failed unexpectedly for order-independent build: {e}")


def test_scenario_test_invalid_sample_size():
    """Tests that a non-positive sample_size raises a ValueError."""
    with pytest.raises(ValueError, match="sample_size must be a positive integer"):
        ScenarioTest("Test").given("Given").when("When").sample_size(0)


def test_scenario_test_invalid_max_turns():
    """Tests that a non-positive max_turns raises a ValueError."""
    with pytest.raises(ValueError, match="max_turns must be a positive integer"):
        ScenarioTest("Test").given("Given").when("When").sample_size(1).max_turns(0)


def test_scenario_test_empty_title():
    """Tests that an empty title raises a ValueError."""
    with pytest.raises(ValueError, match="title must not be empty"):
        ScenarioTest(" ")


def test_scenario_test_empty_given():
    """Tests that an empty given raises a ValueError."""
    with pytest.raises(ValueError, match="'given' context must not be empty"):
        ScenarioTest("Test").given(" ")


def test_scenario_test_empty_when():
    """Tests that an empty when raises a ValueError."""
    with pytest.raises(ValueError, match="'when' action must not be empty"):
        ScenarioTest("Test").given("Given").when(" ")


def test_scenario_test_incomplete():
    """Tests that an incomplete ScenarioTest raises a ValidationError."""
    with pytest.raises(ValidationError, match="ScenarioTest is incomplete"):
        # Try to use it without adding expectations - validation happens when finalized
        scenario = ScenarioTest("Test").given("Given").when("When").sample_size(1)
        # Force validation by calling _finalize_build
        scenario._finalize_build()


@pytest.mark.parametrize(
    "expectation_factory",
    [
        lambda: Expectation.behavior(expected_behavior="Test behavior", criteria=[]),
        lambda: Expectation.metric(
            metric=MetricDefinition(name="test", scope="per_turn", calculator=lambda conv: [1.0]),
            criteria=[],
        ),
    ],
)
def test_expectation_empty_criteria_list(expectation_factory):
    """Tests that an empty list for 'criteria' in any Expectation raises a ValidationError."""
    with pytest.raises(ValidationError, match="'criteria' cannot be an empty list"):
        expectation_factory()


def test_expectation_behavior_factory():
    """Tests that the .behavior() factory method correctly populates fields."""
    expectation = Expectation.behavior(
        expected_behavior="Test", criteria=MockScoreAssertion()
    )
    assert expectation.expected_behavior == "Test"
    assert expectation.metric_definition is None
    assert isinstance(expectation.criteria[0], MockScoreAssertion)


def test_expectation_metric_factory():
    """Tests that the .metric() factory method correctly populates fields."""
    metric = MetricDefinition(name="test", scope="per_turn", calculator=lambda conv: [1.0])
    expectation = Expectation.metric(metric=metric, criteria=MockMetricAssertion())
    assert expectation.metric_definition == metric
    assert expectation.expected_behavior is None
    assert isinstance(expectation.criteria[0], MockMetricAssertion)


@pytest.mark.parametrize(
    "invalid_params, error_match",
    [
        (
            {
                "expected_behavior": "Test",
                "metric_definition": MetricDefinition(
                    name="test", scope="per_turn", calculator=lambda conv: [1.0]
                ),
                "criteria": [MockScoreAssertion()],
            },
            "An Expectation cannot have both 'expected_behavior' and 'metric_definition' defined.",
        ),
        (
            {"criteria": [MockScoreAssertion()]},
            "An Expectation must have either 'expected_behavior' or 'metric_definition' defined.",
        ),
    ],
)
def test_expectation_invalid_manual_creation(invalid_params, error_match):
    """
    Tests that manually creating an Expectation with invalid parameters
    raises a ValidationError.
    """
    with pytest.raises(ValidationError, match=error_match):
        Expectation(**invalid_params)


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


def test_metric_model():
    """Tests the Metric model."""
    from sigmaeval.core.models import ConversationRecord

    def latency_calculator(conversation: "ConversationRecord") -> list[float]:
        return [1.0, 2.0]

    metric = MetricDefinition(
        name="test_latency", scope="per_turn", calculator=latency_calculator
    )
    convo = ConversationRecord()

    assert metric.name == "test_latency"
    assert metric.scope == "per_turn"
    assert metric(convo) == [1.0, 2.0]
