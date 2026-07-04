"""Route requests using a Thompson sampling bandit constrained by daily budget."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ..memory.budget import BudgetManager
from ..providers.base import LLMProvider
from .bandit import ThompsonBandit
from .complexity import Complexity, classify

log = logging.getLogger("mlfix.router")


@dataclass
class RouteDecision:
    provider: LLMProvider
    complexity: Complexity
    category: str
    reason: str


class Router:
    """
    Two-layer routing:
      1. Complexity → candidate provider list (fallback chain)
      2. Bandit picks among candidates that fit within budget

    Category comes from triage (Phase 4), passed in at route time.
    """
    def __init__(
        self,
        easy_provider: LLMProvider,
        medium_provider: LLMProvider,
        hard_provider: LLMProvider,
        budget: BudgetManager,
        bandit: ThompsonBandit,
    ) -> None:
        # Ordered strongest -> cheapest per complexity
        self.by_complexity = {
            Complexity.EASY: [easy_provider],
            Complexity.MEDIUM: [medium_provider, easy_provider],
            Complexity.HARD: [hard_provider, medium_provider, easy_provider],
        }
        self.budget = budget
        self.bandit = bandit
        # Index providers by model name for bandit-based selection
        self._by_model = {p.model: p for p in {easy_provider, medium_provider, hard_provider}}

    def route(self, error: str, code: str, category: str) -> RouteDecision:
        complexity = classify(error, code)
        candidates = self.by_complexity[complexity]

        # Filter to those within budget
        affordable = [p for p in candidates if self.budget.can_use(p.model)[0]]
        if not affordable:
            raise RuntimeError(
                f"All providers over budget for complexity={complexity.value}. Try again tomorrow."
            )

        # Bandit picks among affordable candidates
        candidate_models = [p.model for p in affordable]
        picked_model = self.bandit.select(category, candidate_models)
        provider = self._by_model[picked_model]

        reason = f"complexity={complexity.value}, category={category}, bandit -> {picked_model}"
        log.info(reason)
        return RouteDecision(
            provider=provider,
            complexity=complexity,
            category=category,
            reason=reason,
        )

    def record_outcome(self, category: str, model: str, reward: float) -> None:
        """Called by the pipeline after a judge verdict + user decision arrives."""
        self.bandit.update(category, model, reward)