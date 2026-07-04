import tempfile
from datetime import datetime
from pathlib import Path

from mlfix.memory.episodic import Episode, EpisodicMemory, new_episode_id


def _make_ep(**overrides) -> Episode:
    defaults = dict(
        episode_id=new_episode_id(),
        ts=datetime.utcnow().isoformat(),
        category="shape_mismatch",
        model_used="gemini-2.5-flash",
        stages_run=["triage", "specialist"],
        error="err",
        original_code="x = 1",
        fixed_code="x = 2",
        explanation="fixed",
        confidence=0.9,
        critic_approved=True,
        critic_issues=[],
        executed=False,
        execution_success=None,
        judge_success=True,
        judge_reason="ok",
        tokens_in=10,
        tokens_out=20,
    )
    defaults.update(overrides)
    return Episode(**defaults)


def test_record_and_get():
    with tempfile.TemporaryDirectory() as td:
        m = EpisodicMemory(Path(td) / "e.db")
        ep = _make_ep()
        m.record(ep)
        got = m.get(ep.episode_id)
        assert got is not None
        assert got.category == "shape_mismatch"
        assert got.user_decision is None


def test_set_user_decision():
    with tempfile.TemporaryDirectory() as td:
        m = EpisodicMemory(Path(td) / "e.db")
        ep = _make_ep()
        m.record(ep)
        assert m.set_user_decision(ep.episode_id, "accept")
        got = m.get(ep.episode_id)
        assert got.user_decision == "accept"


def test_query_successful_filters():
    with tempfile.TemporaryDirectory() as td:
        m = EpisodicMemory(Path(td) / "e.db")

        # accepted + judged ok
        ep1 = _make_ep()
        m.record(ep1)
        m.set_user_decision(ep1.episode_id, "accept")

        # rejected
        ep2 = _make_ep()
        m.record(ep2)
        m.set_user_decision(ep2.episode_id, "reject")

        # judge failed
        ep3 = _make_ep(judge_success=False)
        m.record(ep3)
        m.set_user_decision(ep3.episode_id, "accept")

        results = m.query_successful()
        assert len(results) == 1
        assert results[0].episode_id == ep1.episode_id