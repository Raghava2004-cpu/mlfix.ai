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
    MAX_ITERATIONS = 3

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

        # 1. Triage — runs once, category won't change across retries
        stages.append("triage")
        triage_result = await self.triage.classify(code, error)
        if triage_result.tokens_in or triage_result.tokens_out:
            self.budget.record(
                self.triage.provider.model,
                triage_result.tokens_in,
                triage_result.tokens_out,
            )
        log.info("triage: category=%s conf=%.2f",
                 triage_result.category.value, triage_result.confidence)

        # 2. Retrieve — runs once, same past fixes apply to all iterations
        stages.append("retrieve")
        examples = self.semantic.retrieve(
            error=error, code=code, category=triage_result.category.value, k=3,
        )
        log.info("retrieved %d examples from semantic memory", len(examples))

        # 3. Route — bandit picks a specialist model, sticks with it across iterations
        decision = self.router.route(error, code, triage_result.category.value)
        fixer = FixerAgent(decision.provider)

        # ── ITERATION LOOP ──────────────────────────────────────────
        attempt_history: list[dict] = []
        fix = None
        critic_verdict = None
        execution = None
        verdict = None
        total_in = triage_result.tokens_in
        total_out = triage_result.tokens_out
        successful_iteration = 0

        for iteration in range(self.MAX_ITERATIONS):
            iter_num = iteration + 1
            log.info("── ITERATION %d/%d ──", iter_num, self.MAX_ITERATIONS)

            effective_error = self._build_error_with_history(error, attempt_history)

            stages.append(f"specialist_iter{iter_num}")
            fix = await fixer.fix(code, effective_error, language, triage_result.category, examples)
            self.budget.record(fix.model, fix.tokens_in, fix.tokens_out)
            total_in += fix.tokens_in
            total_out += fix.tokens_out
            log.info("specialist iter%d: model=%s conf=%.2f", iter_num, fix.model, fix.confidence)

            stages.append(f"critic_iter{iter_num}")
            critic_verdict = await self.critic.review(
                code, fix.fixed_code, fix.explanation, error,
            )
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
                stages.append(f"execute_iter{iter_num}")
                raw = await self.executor.run_python(final_code)
                execution = ExecutionResult(
                    ran=True,
                    exit_code=raw["exit_code"],
                    stdout=raw["stdout"],
                    stderr=raw["stderr"],
                    duration_ms=raw["duration_ms"],
                    timed_out=raw["timed_out"],
                )

            stages.append(f"judge_iter{iter_num}")
            verdict = self.judge.decide(critic_verdict, execution)
            log.info("judge iter%d: success=%s reason=%s",
                     iter_num, verdict.success, verdict.reason)

            if verdict.success:
                successful_iteration = iter_num
                log.info("succeeded on iteration %d", iter_num)
                break

            if not verdict.should_retry:
                log.info("judge says don't retry; giving up")
                break

            failure = {
                "iteration": iter_num,
                "attempted_fix": fix.fixed_code[:600],
                "why_failed": verdict.reason,
            }
            if execution and execution.stderr:
                failure["execution_stderr"] = execution.stderr[:500]
            if critic_verdict.issues:
                failure["critic_issues"] = critic_verdict.issues
            attempt_history.append(failure)

            log.info("iteration %d failed; retrying with failure context", iter_num)
        # ── END LOOP ────────────────────────────────────────────────

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

        # Bandit reward: rewards decay for later iterations to encourage first-shot success
        if verdict.success:
            reward = {1: 1.0, 2: 0.7, 3: 0.5}.get(successful_iteration, 0.4)
        else:
            reward = 0.0
        self.router.record_outcome(triage_result.category.value, fix.model, reward)

        if self.backend and self.backend.enabled:
            asyncio.create_task(self.backend.upload_episode(ep, upload_code=False))

        return trace, episode_id

    def _build_error_with_history(self, original_error: str, history: list[dict]) -> str:
        """Prepend prior failed attempts so the specialist doesn't repeat them."""
        if not history:
            return original_error

        parts = [
            "Original error:",
            original_error,
            "",
            "=" * 60,
            "PREVIOUS FIX ATTEMPTS THAT FAILED:",
            "=" * 60,
        ]
        for h in history:
            parts.append(f"\n-- Attempt {h['iteration']} --")
            parts.append(f"Fix tried:\n{h['attempted_fix']}")
            parts.append(f"Why it failed: {h['why_failed']}")
            if "execution_stderr" in h:
                parts.append(f"Runtime error:\n{h['execution_stderr']}")
            if "critic_issues" in h:
                parts.append(f"Critic issues: {', '.join(h['critic_issues'])}")

        parts.append("\n" + "=" * 60)
        parts.append("Do NOT repeat these mistakes. Try a fundamentally different approach.")
        parts.append("=" * 60)
        return "\n".join(parts)