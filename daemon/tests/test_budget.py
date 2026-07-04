import tempfile
from pathlib import Path

from mlfix.memory.budget import BudgetManager


def test_budget_records_and_reads():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        b = BudgetManager(db, daily_limits={"model-x": 10})

        allowed, _ = b.can_use("model-x")
        assert allowed

        b.record("model-x", 100, 50)
        stats = b.get("model-x")
        assert stats.requests == 1
        assert stats.tokens_in == 100
        assert stats.tokens_out == 50


def test_budget_blocks_at_limit():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "test.db"
        b = BudgetManager(db, daily_limits={"model-x": 2})

        b.record("model-x", 1, 1)
        b.record("model-x", 1, 1)

        allowed, reason = b.can_use("model-x")
        assert not allowed
        assert "limit" in reason.lower()