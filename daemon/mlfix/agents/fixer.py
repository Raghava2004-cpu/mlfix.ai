"""Fixer agent — now specialist-aware."""
from __future__ import annotations
import re
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
    """Extract a JSON object from model output, tolerating common malformations.

    The model sometimes returns:
    - Markdown fences around the whole thing
    - Raw newlines inside JSON string values (illegal but common)
    - ```python fences INSIDE a "fixed_code" value
    """
    text = text.strip()

    def clean_fixed_code(data: dict) -> dict:
        fixed_code = data.get("fixed_code")
        if isinstance(fixed_code, str):
            data["fixed_code"] = re.sub(r"```(?:python)?", "", fixed_code).strip()
        return data

    # Strip outer markdown fence
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # Attempt 1: parse as-is
    try:
        return clean_fixed_code(json.loads(text))
    except json.JSONDecodeError:
        pass

    # Attempt 2: find first { and last }
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    candidate = text[start:end + 1]

    try:
        return clean_fixed_code(json.loads(candidate))
    except json.JSONDecodeError:
        pass

    # Attempt 3: repair invalid control characters and inner markdown fences
    repaired = _repair_json_string_values(candidate)
    try:
        return clean_fixed_code(json.loads(repaired))
    except json.JSONDecodeError as e:
        # Last resort: give up with a clear error
        raise json.JSONDecodeError(
            f"could not parse or repair model JSON: {e.msg}", repaired, e.pos
        ) from e


def _repair_json_string_values(text: str) -> str:
    """
    Model sometimes emits JSON like:
        {"fixed_code": "
    def foo():
        pass
    ", "explanation": "..."}
    which contains literal newlines and possibly ```python fences inside strings.

    Strategy: walk char-by-char. When inside a string value, escape control chars
    and strip ```python / ``` fences. Not perfect, but handles the common cases.
    """
    out = []
    i = 0
    in_string = False
    escape_next = False
    while i < len(text):
        ch = text[i]

        if escape_next:
            out.append(ch)
            escape_next = False
            i += 1
            continue

        if ch == "\\":
            out.append(ch)
            escape_next = True
            i += 1
            continue

        if ch == '"':
            in_string = not in_string
            out.append(ch)
            i += 1
            continue

        if in_string:
            # Strip ```python or ``` fences that got embedded in string values
            if text[i:i+10] == "```python\n":
                i += 10
                continue
            if text[i:i+4] == "```\n":
                i += 4
                continue
            if text[i:i+3] == "```":
                i += 3
                continue
            # Escape raw control characters
            if ch == "\n":
                out.append("\\n")
                i += 1
                continue
            if ch == "\r":
                out.append("\\r")
                i += 1
                continue
            if ch == "\t":
                out.append("\\t")
                i += 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)

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