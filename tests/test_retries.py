import pytest
from types import SimpleNamespace

from sigmaeval.core.rubric_generator import _generate_rubric
from sigmaeval.core.models import RetryConfig
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
        given="given",
        when="when",
        then=SimpleNamespace(expected_behavior="expected"),
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
        scenario=mock_scenario, expectation=mock_scenario.then, model="dummy", retry_config=cfg
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
        scenario=mock_scenario, expectation=mock_scenario.then, model="dummy", retry_config=cfg
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
            scenario=mock_scenario, expectation=mock_scenario.then, model="dummy", retry_config=cfg
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
            scenario=mock_scenario, expectation=mock_scenario.then, model="dummy", retry_config=cfg
        )

    # Should have attempted exactly max_attempts times
    assert call_counter["n"] == 3


