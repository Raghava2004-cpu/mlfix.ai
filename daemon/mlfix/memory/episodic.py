"""Persist every fix attempt with its outcome. Used for both UI history and RL replay."""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

log = logging.getLogger("mlfix.episodic")


@dataclass
class Episode:
    """One end-to-end fix attempt."""
    episode_id: str
    ts: str                        # ISO timestamp
    category: str
    model_used: str
    stages_run: list[str]
    error: str
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float
    critic_approved: bool
    critic_issues: list[str]
    executed: bool
    execution_success: bool | None
    judge_success: bool
    judge_reason: str
    tokens_in: int
    tokens_out: int
    # Outcome — filled in later when user clicks Accept/Reject
    user_decision: str | None = None    # "accept" | "reject" | None (pending)
    user_decision_ts: str | None = None


class EpisodicMemory:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.db_path))) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    ts TEXT NOT NULL,
                    category TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    stages_run TEXT NOT NULL,
                    error TEXT NOT NULL,
                    original_code TEXT NOT NULL,
                    fixed_code TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    critic_approved INTEGER NOT NULL,
                    critic_issues TEXT NOT NULL,
                    executed INTEGER NOT NULL,
                    execution_success INTEGER,
                    judge_success INTEGER NOT NULL,
                    judge_reason TEXT NOT NULL,
                    tokens_in INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    user_decision TEXT,
                    user_decision_ts TEXT
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_episodes_category ON episodes(category)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_episodes_model ON episodes(model_used)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts)")
            con.commit()

    def record(self, ep: Episode) -> None:
        with self._lock, closing(sqlite3.connect(str(self.db_path))) as con:
            con.execute("""
                INSERT INTO episodes VALUES (
                    :episode_id, :ts, :category, :model_used, :stages_run, :error,
                    :original_code, :fixed_code, :explanation, :confidence,
                    :critic_approved, :critic_issues, :executed, :execution_success,
                    :judge_success, :judge_reason, :tokens_in, :tokens_out,
                    :user_decision, :user_decision_ts
                )
            """, {
                **asdict(ep),
                "stages_run": json.dumps(ep.stages_run),
                "critic_issues": json.dumps(ep.critic_issues),
                "critic_approved": int(ep.critic_approved),
                "executed": int(ep.executed),
                "execution_success": None if ep.execution_success is None else int(ep.execution_success),
                "judge_success": int(ep.judge_success),
            })
            con.commit()
        log.info("recorded episode %s (category=%s model=%s judge=%s)",
                 ep.episode_id, ep.category, ep.model_used, ep.judge_success)

    def set_user_decision(self, episode_id: str, decision: str) -> bool:
        assert decision in ("accept", "reject")
        with self._lock, closing(sqlite3.connect(str(self.db_path))) as con:
            cur = con.execute(
                "UPDATE episodes SET user_decision=?, user_decision_ts=? WHERE episode_id=?",
                (decision, datetime.utcnow().isoformat(), episode_id),
            )
            con.commit()
            updated = cur.rowcount > 0
        if updated:
            log.info("episode %s -> user_decision=%s", episode_id, decision)
        return updated

    def get(self, episode_id: str) -> Episode | None:
        with closing(sqlite3.connect(str(self.db_path))) as con:
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT * FROM episodes WHERE episode_id=?", (episode_id,)).fetchone()
        return self._row_to_episode(row) if row else None

    def query_successful(self, category: str | None = None, limit: int = 200) -> list[Episode]:
        """Successful fixes (judge=success AND user accepted) — used to build few-shot examples."""
        with closing(sqlite3.connect(str(self.db_path))) as con:
            con.row_factory = sqlite3.Row
            if category:
                rows = con.execute("""
                    SELECT * FROM episodes
                    WHERE judge_success=1 AND user_decision='accept' AND category=?
                    ORDER BY ts DESC LIMIT ?
                """, (category, limit)).fetchall()
            else:
                rows = con.execute("""
                    SELECT * FROM episodes
                    WHERE judge_success=1 AND user_decision='accept'
                    ORDER BY ts DESC LIMIT ?
                """, (limit,)).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def query_for_bandit(self, category: str, model: str) -> list[tuple[bool, bool]]:
        """
        Returns (judge_success, user_accepted) tuples for a (category, model) pair.
        Used by the bandit to compute reward stats.
        """
        with closing(sqlite3.connect(str(self.db_path))) as con:
            rows = con.execute("""
                SELECT judge_success, user_decision FROM episodes
                WHERE category=? AND model_used=?
            """, (category, model)).fetchall()
        return [(bool(j), (d == "accept")) for j, d in rows]

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        return Episode(
            episode_id=row["episode_id"],
            ts=row["ts"],
            category=row["category"],
            model_used=row["model_used"],
            stages_run=json.loads(row["stages_run"]),
            error=row["error"],
            original_code=row["original_code"],
            fixed_code=row["fixed_code"],
            explanation=row["explanation"],
            confidence=row["confidence"],
            critic_approved=bool(row["critic_approved"]),
            critic_issues=json.loads(row["critic_issues"]),
            executed=bool(row["executed"]),
            execution_success=None if row["execution_success"] is None else bool(row["execution_success"]),
            judge_success=bool(row["judge_success"]),
            judge_reason=row["judge_reason"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            user_decision=row["user_decision"],
            user_decision_ts=row["user_decision_ts"],
        )


def new_episode_id() -> str:
    return uuid.uuid4().hex[:12]