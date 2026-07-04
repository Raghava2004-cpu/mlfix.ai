"""Thompson sampling bandit: for each (category, model) it maintains a Beta(alpha, beta)
posterior on success rate. On each request it samples from each arm's posterior and
picks the highest sample.

'Context' here = error category. This is a per-arm contextual bandit.

We store alpha/beta in SQLite so it persists across restarts.
"""
from __future__ import annotations

import logging
import random
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

log = logging.getLogger("mlfix.bandit")


@dataclass
class ArmStats:
    category: str
    model: str
    alpha: float  # successes + 1
    beta: float   # failures + 1

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def total_pulls(self) -> int:
        return int(self.alpha + self.beta - 2)


class ThompsonBandit:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.db_path))) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS bandit_arms (
                    category TEXT NOT NULL,
                    model TEXT NOT NULL,
                    alpha REAL NOT NULL DEFAULT 1.0,
                    beta REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (category, model)
                )
            """)
            con.commit()

    def _get_or_init(self, category: str, model: str) -> ArmStats:
        with self._lock, closing(sqlite3.connect(str(self.db_path))) as con:
            row = con.execute(
                "SELECT alpha, beta FROM bandit_arms WHERE category=? AND model=?",
                (category, model),
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO bandit_arms(category, model, alpha, beta) VALUES(?,?,1.0,1.0)",
                    (category, model),
                )
                con.commit()
                return ArmStats(category, model, 1.0, 1.0)
            return ArmStats(category, model, row[0], row[1])

    def select(self, category: str, candidate_models: list[str]) -> str:
        """Sample a success rate from each arm and pick the argmax."""
        best_model = candidate_models[0]
        best_sample = -1.0
        samples: list[tuple[str, float]] = []
        for m in candidate_models:
            arm = self._get_or_init(category, m)
            sample = random.betavariate(arm.alpha, arm.beta)
            samples.append((m, sample))
            if sample > best_sample:
                best_sample = sample
                best_model = m
        log.info("bandit select category=%s samples=%s -> %s",
                 category, [(m, f"{s:.3f}") for m, s in samples], best_model)
        return best_model

    def update(self, category: str, model: str, reward: float) -> None:
        """reward in [0, 1]. We convert to a fractional Beta update."""
        reward = max(0.0, min(1.0, reward))
        with self._lock, closing(sqlite3.connect(str(self.db_path))) as con:
            con.execute("""
                INSERT INTO bandit_arms(category, model, alpha, beta) VALUES(?,?, 1+?, 1+?)
                ON CONFLICT(category, model) DO UPDATE SET
                    alpha = alpha + excluded.alpha - 1,
                    beta = beta + excluded.beta - 1
            """, (category, model, reward, 1.0 - reward))
            con.commit()
        log.info("bandit update category=%s model=%s reward=%.2f", category, model, reward)
    
    def apply_global_priors(self, priors: list[dict]) -> int:
        """
        Merge global bandit stats into local arms. We don't overwrite local knowledge;
        we blend by summing counts. Local pulls stay more influential over time.

        priors: list of {"category": str, "model": str, "alpha": num, "beta": num}
        Returns number of arms merged.
        """
        merged = 0
        with self._lock, sqlite3.connect(self.db_path) as con:
            for p in priors:
                cat = p.get("category")
                model = p.get("model")
                g_alpha = float(p.get("alpha", 1.0))
                g_beta = float(p.get("beta", 1.0))
                if not cat or not model:
                    continue

                row = con.execute(
                    "SELECT alpha, beta FROM bandit_arms WHERE category=? AND model=?",
                    (cat, model),
                ).fetchone()

                if row is None:
                    # New arm: use global priors directly
                    con.execute(
                        "INSERT INTO bandit_arms(category, model, alpha, beta) VALUES(?,?,?,?)",
                        (cat, model, g_alpha, g_beta),
                    )
                else:
                    # Existing arm: blend, but discount global (weight 0.3) so
                    # local experience wins over time.
                    l_alpha, l_beta = row
                    new_alpha = l_alpha + 0.3 * (g_alpha - 1.0)
                    new_beta = l_beta + 0.3 * (g_beta - 1.0)
                    con.execute(
                        "UPDATE bandit_arms SET alpha=?, beta=? WHERE category=? AND model=?",
                        (max(1.0, new_alpha), max(1.0, new_beta), cat, model),
                    )
                merged += 1
            con.commit()
        log.info("applied %d global priors to bandit", merged)
        return merged
    def snapshot(self) -> list[ArmStats]:
        with closing(sqlite3.connect(str(self.db_path))) as con:
            rows = con.execute("SELECT category, model, alpha, beta FROM bandit_arms").fetchall()
        return [ArmStats(*r) for r in rows]
