"""Simple SQLite-backed token & request counter with daily budget enforcement."""
from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import Lock

log = logging.getLogger("mlfix.budget")

# Free-tier limits (approx, as of 2026) — request-per-day caps we self-enforce.
# We stay conservative so we never actually hit Google's 429.
DEFAULT_DAILY_LIMITS = {
    "gemini-2.5-flash-lite": 900,   # actual ~1000+, keep buffer
    "gemini-2.5-flash": 200,        # more conservative
}


@dataclass
class UsageStats:
    model: str
    day: str
    requests: int
    tokens_in: int
    tokens_out: int


class BudgetManager:
    def __init__(self, db_path: Path, daily_limits: dict[str, int] | None = None) -> None:
        self.db_path = db_path
        self.daily_limits = daily_limits or DEFAULT_DAILY_LIMITS
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.db_path))) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    day TEXT NOT NULL,
                    model TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    tokens_in INTEGER NOT NULL DEFAULT 0,
                    tokens_out INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (day, model)
                )
            """)
            con.commit()

    def _today(self) -> str:
        return date.today().isoformat()

    def can_use(self, model: str) -> tuple[bool, str]:
        """Return (allowed, reason). Check before making a call."""
        limit = self.daily_limits.get(model)
        if limit is None:
            return True, "no limit configured"
        stats = self.get(model)
        if stats.requests >= limit:
            return False, f"daily limit {limit} reached for {model}"
        return True, f"{stats.requests}/{limit} used today"

    def record(self, model: str, tokens_in: int, tokens_out: int) -> None:
        with self._lock, closing(sqlite3.connect(str(self.db_path))) as con:
            con.execute("""
                INSERT INTO usage (day, model, requests, tokens_in, tokens_out)
                VALUES (?, ?, 1, ?, ?)
                ON CONFLICT(day, model) DO UPDATE SET
                    requests = requests + 1,
                    tokens_in = tokens_in + excluded.tokens_in,
                    tokens_out = tokens_out + excluded.tokens_out
            """, (self._today(), model, tokens_in, tokens_out))
            con.commit()
        log.info("recorded usage: model=%s tokens_in=%d tokens_out=%d",
                 model, tokens_in, tokens_out)

    def get(self, model: str) -> UsageStats:
        with closing(sqlite3.connect(str(self.db_path))) as con:
            row = con.execute(
                "SELECT day, model, requests, tokens_in, tokens_out FROM usage WHERE day=? AND model=?",
                (self._today(), model),
            ).fetchone()
        if row is None:
            return UsageStats(model=model, day=self._today(),
                              requests=0, tokens_in=0, tokens_out=0)
        return UsageStats(day=row[0], model=row[1], requests=row[2],
                          tokens_in=row[3], tokens_out=row[4])

    def get_all_today(self) -> list[UsageStats]:
        with closing(sqlite3.connect(str(self.db_path))) as con:
            rows = con.execute(
                "SELECT day, model, requests, tokens_in, tokens_out FROM usage WHERE day=?",
                (self._today(),),
            ).fetchall()
        return [UsageStats(day=r[0], model=r[1], requests=r[2],
                           tokens_in=r[3], tokens_out=r[4]) for r in rows]
