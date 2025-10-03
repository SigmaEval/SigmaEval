import pytest

from sigmaeval.core.llm_client import _acompletion_with_retry
from sigmaeval.core.models import RetryConfig


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


async def _raise(exc: Exception):
    raise exc


@pytest.mark.asyncio
async def test_retry_succeeds_after_failures(monkeypatch):
    call_counter = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_counter["n"] += 1
        if call_counter["n"] <= 2:
            # Fail first two attempts
            raise RuntimeError("transient error")
        return _FakeResponse("ok")

    monkeypatch.setattr("sigmaeval.core.llm_client._litellm_acompletion", fake_acompletion)

    cfg = RetryConfig(enabled=True, max_attempts=5, backoff_multiplier=0.0, max_backoff_seconds=0.0)
    resp = await _acompletion_with_retry(model="dummy", messages=[], retry_config=cfg)

    assert isinstance(resp, _FakeResponse)
    assert resp.choices[0].message.content == "ok"
    # Should have tried exactly 3 times (2 failures + 1 success)
    assert call_counter["n"] == 3


@pytest.mark.asyncio
async def test_retry_disabled_no_retry(monkeypatch):
    call_counter = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_counter["n"] += 1
        raise RuntimeError("should not retry when disabled")

    monkeypatch.setattr("sigmaeval.core.llm_client._litellm_acompletion", fake_acompletion)

    cfg = RetryConfig(enabled=False, max_attempts=5)
    with pytest.raises(RuntimeError):
        await _acompletion_with_retry(model="dummy", messages=[], retry_config=cfg)

    # Only one attempt should be made when retries are disabled
    assert call_counter["n"] == 1


@pytest.mark.asyncio
async def test_retry_exhausts_attempts(monkeypatch):
    call_counter = {"n": 0}

    async def always_fail(**kwargs):
        call_counter["n"] += 1
        raise RuntimeError("always failing")

    monkeypatch.setattr("sigmaeval.core.llm_client._litellm_acompletion", always_fail)

    cfg = RetryConfig(enabled=True, max_attempts=3, backoff_multiplier=0.0, max_backoff_seconds=0.0)
    with pytest.raises(RuntimeError):
        await _acompletion_with_retry(model="dummy", messages=[], retry_config=cfg)

    # Should have attempted exactly max_attempts times
    assert call_counter["n"] == 3


