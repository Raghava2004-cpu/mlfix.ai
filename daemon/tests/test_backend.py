import pytest
from unittest.mock import AsyncMock, patch

from datetime import datetime
from mlfix.backend.client import BackendClient
from mlfix.memory.episodic import Episode, new_episode_id


def _make_ep() -> Episode:
    return Episode(
        episode_id=new_episode_id(),
        ts=datetime.utcnow().isoformat(),
        category="shape_mismatch",
        model_used="gemini-2.5-flash",
        stages_run=["triage", "specialist", "critic", "judge"],
        error="RuntimeError: shape mismatch",
        original_code="x @ y",
        fixed_code="x @ y.T",
        explanation="transpose y",
        confidence=0.9,
        critic_approved=True,
        critic_issues=[],
        executed=False,
        execution_success=None,
        judge_success=True,
        judge_reason="ok",
        tokens_in=100,
        tokens_out=50,
    )


def test_client_disabled_when_no_config():
    c = BackendClient(base_url="", api_key="", user_id="u")
    assert not c.enabled


def test_client_enabled_when_configured():
    c = BackendClient(base_url="https://x.com", api_key="k", user_id="u")
    assert c.enabled


def test_hash_code_is_short():
    c = BackendClient(base_url="https://x.com", api_key="k", user_id="u")
    h = c._hash_code("some code here")
    assert len(h) == 16


@pytest.mark.asyncio
async def test_upload_returns_false_when_disabled():
    c = BackendClient(base_url="", api_key="", user_id="u")
    ok = await c.upload_episode(_make_ep())
    assert not ok


@pytest.mark.asyncio
async def test_upload_strips_code_when_not_opted_in():
    c = BackendClient(base_url="https://fake.example.com", api_key="k", user_id="u")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.status_code = 200
        await c.upload_episode(_make_ep(), upload_code=False)
        args, kwargs = mock_post.call_args
        body = kwargs["json"]
        assert "original_code" not in body["episode"]
        assert "fixed_code" not in body["episode"]
        assert "code_hash" in body["episode"]


@pytest.mark.asyncio
async def test_upload_swallows_network_errors():
    import httpx
    c = BackendClient(base_url="https://fake.example.com", api_key="k", user_id="u")
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("boom")
        ok = await c.upload_episode(_make_ep())
        assert not ok  # graceful degradation