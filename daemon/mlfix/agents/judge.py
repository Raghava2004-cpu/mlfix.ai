"""Judge — decides success from execution + critic output. Rule-based, no LLM."""
from __future__ import annotations

from .types import CriticVerdict, ExecutionResult, JudgeVerdict


class JudgeAgent:
    """Not every stage needs an LLM. Judge is deterministic."""

    def decide(
        self,
        critic: CriticVerdict,
        execution: ExecutionResult | None,
    ) -> JudgeVerdict:
        if not critic.approved:
            return JudgeVerdict(
                success=False,
                reason=f"critic rejected: {'; '.join(critic.issues) or 'no reason given'}",
                should_retry=True,
            )

        if execution is None:
            # No execution attempted (e.g. non-runnable snippet). Trust critic.
            return JudgeVerdict(
                success=True,
                reason="critic approved (no execution performed)",
                should_retry=False,
            )

        if execution.timed_out:
            return JudgeVerdict(
                success=False,
                reason="execution timed out",
                should_retry=False,
            )

        if execution.exit_code == 0:
            return JudgeVerdict(
                success=True,
                reason="execution exited 0",
                should_retry=False,
            )

        return JudgeVerdict(
            success=False,
            reason=f"execution failed with exit code {execution.exit_code}",
            should_retry=True,
        )