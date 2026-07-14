"""Fixer agent — now specialist-aware."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..providers.base import LLMProvider
from .specialists import USER_TEMPLATE, system_prompt_for
from .types import ErrorCategory, FixCandidate

log = logging.getLogger("mlfix.fixer")


@dataclass
class FixResult:
    fixed_code: str
    explanation: str
    confidence: float
    model: str
    tokens_in: int
    tokens_out: int


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


class FixerAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    async def fix(
        self,
        code: str,
        error: str,
        language: str = "python",
        category: ErrorCategory = ErrorCategory.GENERIC,
        examples: list[dict] | None = None,
        ast_summary: str | None = None,
    ) -> FixCandidate:
        from .specialists import build_user_prompt, system_prompt_for
        system = system_prompt_for(category)
        user = build_user_prompt(code, error, language, examples, ast_summary)

        resp = await self.provider.complete(system, user)

        try:
            data = _extract_json(resp.text)
        except json.JSONDecodeError as e:
            log.error("Failed to parse fixer JSON:\n%s", resp.text)
            raise ValueError(f"Model did not return valid JSON: {e}") from e

        return FixCandidate(
            fixed_code=str(data.get("fixed_code", "")),
            explanation=str(data.get("explanation", "")),
            confidence=float(data.get("confidence", 0.5)),
            model=resp.model,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
        )