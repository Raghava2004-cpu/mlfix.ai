import tempfile
from pathlib import Path

from mlfix.router.bandit import ThompsonBandit


def test_bandit_prefers_winning_arm():
    with tempfile.TemporaryDirectory() as td:
        b = ThompsonBandit(Path(td) / "b.db")

        # Feed arm A many wins, arm B many losses
        for _ in range(50):
            b.update("cat", "model_A", 1.0)
            b.update("cat", "model_B", 0.0)

        # Over many samples, A should dominate
        wins_A = sum(1 for _ in range(200) if b.select("cat", ["model_A", "model_B"]) == "model_A")
        assert wins_A > 150  # very high confidence


def test_bandit_new_arms_equal_prior():
    with tempfile.TemporaryDirectory() as td:
        b = ThompsonBandit(Path(td) / "b.db")
        # No updates: both arms Beta(1,1) -> uniform. Roughly 50/50.
        wins_A = sum(1 for _ in range(400) if b.select("cat", ["A", "B"]) == "A")
        assert 150 < wins_A < 250  # loose bound; this is a randomized test


def test_snapshot():
    with tempfile.TemporaryDirectory() as td:
        b = ThompsonBandit(Path(td) / "b.db")
        b.update("cat", "M", 1.0)
        snap = b.snapshot()
        assert len(snap) == 1
        assert snap[0].category == "cat"
        assert snap[0].alpha > 1.0