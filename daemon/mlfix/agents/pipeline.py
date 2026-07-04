"""End-to-end pipeline with memory, bandit routing, and optional backend sync."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from ..backend.client import BackendClient
from ..memory.budget import BudgetManager
from ..memory.episodic import Episode, EpisodicMemory, new_episode_id
from ..memory.semantic import SemanticMemory
from ..providers.base import LLMProvider
from ..router.router import Router
from ..mcp_servers.executor import LocalExecutor
from .critic import CriticAgent
from .fixer import FixerAgent
from .judge import JudgeAgent
from .triage import TriageAgent
from .types import ExecutionResult, PipelineTrace

log = logging.getLogger("mlfix.pipeline")


def _looks_runnable(code: str) -> bool:
    lowered = code.lower()
    hazards = ["open(", "requests.", "urllib", "socket.", "torch.load(", "wandb.", "cuda"]
    return not any(h in lowered for h in hazards)


class Pipeline:
    def __init__(
        self,
        router: Router,
        budget: BudgetManager,
        triage_provider: LLMProvider,
        critic_provider: LLMProvider,
        executor: LocalExecutor,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        backend: BackendClient | None = None,
    ) -> None:
        self.router = router
        self.budget = budget
        self.triage = TriageAgent(triage_provider)
        self.critic = CriticAgent(critic_provider)
        self.judge = JudgeAgent()
        self.executor = executor
        self.episodic = episodic
        self.semantic = semantic
        self.backend = backend

    async def run(self, code: str, error: str, language: str = "python") -> tuple[PipelineTrace, str]:
        stages: list[str] = []
        episode_id = new_episode_id()

        stages.append("triage")
        triage_result = await self.triage.classify(code, error)
        if triage_result.tokens_in or triage_result.tokens_out:
            self.budget.record(
                self.triage.provider.model,
                triage_result.tokens_in,
                triage_result.tokens_out,
            )

        stages.append("retrieve")
        examples = self.semantic.retrieve(
            error=error, code=code, category=triage_result.category.value, k=3,
        )

        stages.append("specialist")
        decision = self.router.route(error, code, triage_result.category.value)
        fixer = FixerAgent(decision.provider)
        fix = await fixer.fix(code, error, language, triage_result.category, examples)
        self.budget.record(fix.model, fix.tokens_in, fix.tokens_out)

        stages.append("critic")
        critic_verdict = await self.critic.review(code, fix.fixed_code, fix.explanation, error)
        self.budget.record(
            self.critic.provider.model,
            critic_verdict.tokens_in,
            critic_verdict.tokens_out,
        )

        final_code = critic_verdict.revised_code or fix.fixed_code
        if critic_verdict.revised_code:
            fix.fixed_code = critic_verdict.revised_code

        execution: ExecutionResult | None = None
        if critic_verdict.approved and language == "python" and _looks_runnable(final_code):
            stages.append("execute")
            raw = await self.executor.run_python(final_code)
            execution = ExecutionResult(
                ran=True,
                exit_code=raw["exit_code"],
                stdout=raw["stdout"],
                stderr=raw["stderr"],
                duration_ms=raw["duration_ms"],
                timed_out=raw["timed_out"],
            )

        stages.append("judge")
        verdict = self.judge.decide(critic_verdict, execution)

        total_in = triage_result.tokens_in + fix.tokens_in + critic_verdict.tokens_in
        total_out = triage_result.tokens_out + fix.tokens_out + critic_verdict.tokens_out

        trace = PipelineTrace(
            category=triage_result.category,
            triage=triage_result,
            specialist_used=fix.model,
            fix=fix,
            critic=critic_verdict,
            execution=execution,
            judge=verdict,
            total_tokens_in=total_in,
            total_tokens_out=total_out,
            stages_run=stages,
        )

        exec_ok = None if execution is None else ((execution.exit_code == 0) and not execution.timed_out)
        ep = Episode(
            episode_id=episode_id,
            ts=datetime.utcnow().isoformat(),
            category=triage_result.category.value,
            model_used=fix.model,
            stages_run=stages,
            error=error,
            original_code=code,
            fixed_code=fix.fixed_code,
            explanation=fix.explanation,
            confidence=fix.confidence,
            critic_approved=critic_verdict.approved,
            critic_issues=critic_verdict.issues,
            executed=execution is not None,
            execution_success=exec_ok,
            judge_success=verdict.success,
            judge_reason=verdict.reason,
            tokens_in=total_in,
            tokens_out=total_out,
        )
        self.episodic.record(ep)

        initial_reward = 1.0 if verdict.success else 0.0
        self.router.record_outcome(triage_result.category.value, fix.model, initial_reward)

        # Fire-and-forget backend upload. Doesn't block the response.
        if self.backend and self.backend.enabled:
            asyncio.create_task(self.backend.upload_episode(ep, upload_code=False))

        return trace, episode_id