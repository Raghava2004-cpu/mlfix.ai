"""Critic — reviews the fix. Can approve, reject, or patch."""
from __future__ import annotations

import json
import logging

from ..providers.base import LLMProvider
from .fixer import _extract_json
from .types import CriticVerdict

log = logging.getLogger("mlfix.critic")


CRITIC_SYSTEM = """You review code fixes for correctness. You do NOT rewrite unless needed.

Check the fix for:
1. Hallucinated APIs (methods/args that don't exist).
2. Introduced bugs (variable renamed inconsistently, wrong import).
3. Fix doesn't address the actual error.
4. Loss of the original code's intent.

Respond ONLY with JSON:
{
  "approved": true|false,
  "issues": ["<issue 1>", "<issue 2>"],
  "revised_code": "<only if you had to patch it, else null>"
}"""


CRITIC_USER = """Original error:
{error}

Original code:
```python
{original}
```

Proposed fix:
```python
{fixed}
```

Proposed explanation: {explanation}

Review it."""


class CriticAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def review(
        self,
        original_code: str,
        fixed_code: str,
        explanation: str,
        error: str,
    ) -> CriticVerdict:
        user = CRITIC_USER.format(
            error=error.strip()[:1500],
            original=original_code.rstrip(),
            fixed=fixed_code.rstrip(),
            explanation=explanation,
        )
        resp = await self.provider.complete(CRITIC_SYSTEM, user)

        try:
            data = _extract_json(resp.text)
        except json.JSONDecodeError:
            log.warning("critic JSON parse failed; approving by default")
            return CriticVerdict(
                approved=True,
                issues=["critic-parse-failed"],
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
            )

        revised = data.get("revised_code")
        return CriticVerdict(
            approved=bool(data.get("approved", True)),
            issues=list(data.get("issues", [])),
            revised_code=revised if revised and str(revised).strip() else None,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
        )