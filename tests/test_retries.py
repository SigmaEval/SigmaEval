import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sigmaeval.core.rubric_generator import _generate_rubric
from sigmaeval.core.data_collection import _run_single_interaction, _judge_conversations
from sigmaeval.core.models import RetryConfig, AppResponse
from sigmaeval.core.exceptions import LLMCommunicationError


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


@pytest.fixture
def mock_scenario():
    """Provides a mock scenario object for tests."""
    return SimpleNamespace(
        given_context="given",
        when_action="when",
        then=[SimpleNamespace(expected_behavior="expected", label=None)],
        max_turns_value=1,
    )


@pytest.mark.asyncio
async def test_retry_succeeds_after_failures(monkeypatch, mock_scenario):
    call_counter = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_counter["n"] += 1
        if call_counter["n"] <= 2:
            # Fail first two attempts
            raise RuntimeError("transient error")
        return _FakeResponse("ok rubric")

    monkeypatch.setattr(
        "sigmaeval.core.rubric_generator._litellm_acompletion", fake_acompletion
    )

    cfg = RetryConfig(
        enabled=True,
        max_attempts=5,
        backoff_multiplier=0.0,
        max_backoff_seconds=0.0,
    )
    rubric = await _generate_rubric(
        scenario=mock_scenario,
        expectation=mock_scenario.then[0],
        model="dummy",
        retry_config=cfg,
    )

    assert rubric == "ok rubric"
    # Should have tried exactly 3 times (2 failures + 1 success)
    assert call_counter["n"] == 3


@pytest.mark.asyncio
async def test_retry_on_empty_response(monkeypatch, mock_scenario):
    """Test that the retry logic is triggered on empty/invalid response content."""
    call_counter = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_counter["n"] += 1
        if call_counter["n"] <= 2:
            # Return empty content for first two attempts
            return _FakeResponse("")
        return _FakeResponse("ok rubric")

    monkeypatch.setattr(
        "sigmaeval.core.rubric_generator._litellm_acompletion", fake_acompletion
    )

    cfg = RetryConfig(
        enabled=True,
        max_attempts=5,
        backoff_multiplier=0.0,
        max_backoff_seconds=0.0,
    )
    rubric = await _generate_rubric(
        scenario=mock_scenario,
        expectation=mock_scenario.then[0],
        model="dummy",
        retry_config=cfg,
    )

    assert rubric == "ok rubric"
    assert call_counter["n"] == 3


@pytest.mark.asyncio
async def test_retry_disabled_no_retry(monkeypatch, mock_scenario):
    call_counter = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_counter["n"] += 1
        raise RuntimeError("should not retry when disabled")

    monkeypatch.setattr(
        "sigmaeval.core.rubric_generator._litellm_acompletion", fake_acompletion
    )

    cfg = RetryConfig(enabled=False, max_attempts=5)
    with pytest.raises(LLMCommunicationError):
        await _generate_rubric(
            scenario=mock_scenario,
            expectation=mock_scenario.then[0],
            model="dummy",
            retry_config=cfg,
        )

    # Only one attempt should be made when retries are disabled
    assert call_counter["n"] == 1


@pytest.mark.asyncio
async def test_retry_exhausts_attempts(monkeypatch, mock_scenario):
    call_counter = {"n": 0}

    async def always_fail(**kwargs):
        call_counter["n"] += 1
        raise RuntimeError("always failing")

    monkeypatch.setattr("sigmaeval.core.rubric_generator._litellm_acompletion", always_fail)

    cfg = RetryConfig(
        enabled=True,
        max_attempts=3,
        backoff_multiplier=0.0,
        max_backoff_seconds=0.0,
    )
    with pytest.raises(LLMCommunicationError):
        await _generate_rubric(
            scenario=mock_scenario,
            expectation=mock_scenario.then[0],
            model="dummy",
            retry_config=cfg,
        )

    # Should have attempted exactly max_attempts times
    assert call_counter["n"] == 3


@pytest.mark.asyncio
async def test_retry_succeeds_for_user_simulation(monkeypatch, mock_scenario):
    """Test that retry logic succeeds for user simulation after failures."""
    call_counter = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_counter["n"] += 1
        if call_counter["n"] <= 2:
            raise RuntimeError("transient error")
        # A valid JSON response for the user simulator
        return _FakeResponse('{"message": "Test message", "continue": false}')

    monkeypatch.setattr(
        "sigmaeval.core.data_collection._litellm_acompletion", fake_acompletion
    )

    cfg = RetryConfig(
        enabled=True,
        max_attempts=5,
        backoff_multiplier=0.0,
        max_backoff_seconds=0.0,
    )
    # This is a simplified call, focusing only on the retry aspect
    result = await _run_single_interaction(
        scenario=mock_scenario,
        app_handler=AsyncMock(return_value=AppResponse(response="ok", state={})),
        user_simulator_model="dummy",
        retry_config=cfg,
    )

    assert result is not None
    assert "Test message" in result.turns[0].content
    assert call_counter["n"] == 3


@pytest.mark.asyncio
async def test_retry_succeeds_for_judging(monkeypatch, mock_scenario):
    """Test that retry logic succeeds for judging after failures."""
    call_counter = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_counter["n"] += 1
        if call_counter["n"] <= 2:
            raise RuntimeError("transient error")
        # A valid JSON response for the judge
        return _FakeResponse('{"score": 8, "reasoning": "Good response"}')

    monkeypatch.setattr(
        "sigmaeval.core.data_collection._litellm_acompletion", fake_acompletion
    )

    cfg = RetryConfig(
        enabled=True,
        max_attempts=5,
        backoff_multiplier=0.0,
        max_backoff_seconds=0.0,
    )
    # Provide minimal mock data needed for the function to run
    mock_conversation = SimpleNamespace(turns=[])
    scores, reasoning = await _judge_conversations(
        scenario=mock_scenario,
        expectation=mock_scenario.then[0],
        conversations=[mock_conversation],
        rubric="Test Rubric",
        judge_model="dummy",
        retry_config=cfg,
    )

    assert scores == [8]
    assert reasoning == ["Good response"]
    assert call_counter["n"] == 3


