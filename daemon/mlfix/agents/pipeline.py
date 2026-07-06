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

MAX_ITERATIONS = 3


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

        # Triage runs once — category doesn't change across iterations
        stages.append("triage")
        triage_result = await self.triage.classify(code, error)
        if triage_result.tokens_in or triage_result.tokens_out:
            self.budget.record(
                self.triage.provider.model,
                triage_result.tokens_in,
                triage_result.tokens_out,
            )

        # Retrieval runs once — same past fixes apply to all iterations
        stages.append("retrieve")
        examples = self.semantic.retrieve(
            error=error, code=code, category=triage_result.category.value, k=3,
        )

        # Route once — bandit picks a model
        decision = self.router.route(error, code, triage_result.category.value)
        fixer = FixerAgent(decision.provider)

        # ITERATION LOOP
        attempt_history: list[dict] = []  # accumulates failures for the LLM to see
        fix = None
        critic_verdict = None
        execution = None
        verdict = None
        total_in = triage_result.tokens_in
        total_out = triage_result.tokens_out

        for iteration in range(MAX_ITERATIONS):
            stages.append(f"specialist_iter{iteration + 1}")

            # Build the prompt with failure history from previous iterations
            enhanced_error = self._build_error_with_history(error, attempt_history)

            fix = await fixer.fix(
                code, enhanced_error, language, triage_result.category, examples,
            )
            self.budget.record(fix.model, fix.tokens_in, fix.tokens_out)
            total_in += fix.tokens_in
            total_out += fix.tokens_out

            stages.append(f"critic_iter{iteration + 1}")
            critic_verdict = await self.critic.review(code, fix.fixed_code, fix.explanation, error)
            self.budget.record(
                self.critic.provider.model,
                critic_verdict.tokens_in,
                critic_verdict.tokens_out,
            )
            total_in += critic_verdict.tokens_in
            total_out += critic_verdict.tokens_out

            final_code = critic_verdict.revised_code or fix.fixed_code
            if critic_verdict.revised_code:
                fix.fixed_code = critic_verdict.revised_code

            execution = None
            if critic_verdict.approved and language == "python" and _looks_runnable(final_code):
                stages.append(f"execute_iter{iteration + 1}")
                raw = await self.executor.run_python(final_code)
                execution = ExecutionResult(
                    ran=True,
                    exit_code=raw["exit_code"],
                    stdout=raw["stdout"],
                    stderr=raw["stderr"],
                    duration_ms=raw["duration_ms"],
                    timed_out=raw["timed_out"],
                )

            stages.append(f"judge_iter{iteration + 1}")
            verdict = self.judge.decide(critic_verdict, execution)

            if verdict.success:
                log.info("succeeded on iteration %d", iteration + 1)
                break

            if not verdict.should_retry:
                log.info("judge says don't retry; stopping")
                break

            # Record the failure so next iteration sees it
            failure_info = {
                "iteration": iteration + 1,
                "attempted_fix": fix.fixed_code[:500],
                "why_failed": verdict.reason,
            }
            if execution and execution.stderr:
                failure_info["execution_stderr"] = execution.stderr[:500]
            if critic_verdict.issues:
                failure_info["critic_issues"] = critic_verdict.issues
            attempt_history.append(failure_info)

            log.info("iteration %d failed: %s. Retrying.", iteration + 1, verdict.reason)

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

        if self.backend and self.backend.enabled:
            asyncio.create_task(self.backend.upload_episode(ep, upload_code=False))

        return trace, episode_id

    def _build_error_with_history(self, original_error: str, history: list[dict]) -> str:
        """Prepend previous failed attempts so the specialist sees what didn't work."""
        if not history:
            return original_error

        parts = [
            "Original error:",
            original_error,
            "",
            "Previous fix attempts that FAILED:",
        ]
        for h in history:
            parts.append(f"\nAttempt {h['iteration']}:")
            parts.append(f"  Fix tried: {h['attempted_fix']}")
            parts.append(f"  Why it failed: {h['why_failed']}")
            if "execution_stderr" in h:
                parts.append(f"  Runtime error: {h['execution_stderr']}")
            if "critic_issues" in h:
                parts.append(f"  Critic flagged: {', '.join(h['critic_issues'])}")
        parts.append("\nDo NOT repeat these mistakes. Try a different approach.")
        return "\n".join(parts)