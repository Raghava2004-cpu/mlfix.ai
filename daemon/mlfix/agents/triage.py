"""Fast triage: classify the error into an ErrorCategory. Uses cheapest model."""
from __future__ import annotations

import json
import logging
import re

from ..providers.base import LLMProvider
from .fixer import _extract_json  # reuse JSON extraction
from .types import ErrorCategory, TriageResult

log = logging.getLogger("mlfix.triage")


# Fast rule-based signals — no LLM call needed for many cases
RULE_SIGNALS: list[tuple[re.Pattern, ErrorCategory, str]] = [
    (re.compile(r"cuda\s+out of memory|cudnn.*out of memory", re.I),
     ErrorCategory.CUDA_OOM, "cuda oom pattern"),
    (re.compile(r"mat1 and mat2|size mismatch|shape.*mismatch|can't multiply|dimension.*must match", re.I),
     ErrorCategory.SHAPE_MISMATCH, "shape mismatch pattern"),
    (re.compile(r"modulenotfounderror|no module named", re.I),
     ErrorCategory.DEPENDENCY, "missing module"),
    (re.compile(r"importerror", re.I),
     ErrorCategory.IMPORT, "import error"),
    (re.compile(r"nan.*loss|loss.*is\s+nan|loss.*became\s+nan", re.I),
     ErrorCategory.TRAINING_DYNAMICS, "nan loss"),
    (re.compile(r"^\s*syntaxerror|invalid syntax|unexpected indent|indentationerror", re.I),
     ErrorCategory.SYNTAX, "syntax pattern"),
    (re.compile(r"typeerror", re.I),
     ErrorCategory.TYPE, "type error"),
]


TRIAGE_SYSTEM = """You classify Python/ML errors into one of these categories:
- shape_mismatch: tensor/array shape errors (matmul, broadcast, dim mismatch)
- cuda_oom: GPU out-of-memory
- dependency: missing package / version conflict
- data_leakage: train/test contamination, target leakage
- training_dynamics: NaN loss, no convergence, exploding gradients
- syntax: SyntaxError, IndentationError
- import: ImportError (not missing module — that's dependency)
- type: TypeError
- generic: anything else

Respond ONLY with JSON:
{"category": "<one of above>", "confidence": <0.0-1.0>, "signals": ["<what tipped you off>"]}"""


TRIAGE_USER = """Error:
{error}

Code (first 40 lines):
```python
{code}
```

Classify."""


class TriageAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def rule_based(self, error: str) -> TriageResult | None:
        """Return a triage result if any hard rule matches. Skips the LLM call."""
        for pattern, cat, signal in RULE_SIGNALS:
            if pattern.search(error):
                log.info("triage rule hit: %s -> %s", signal, cat.value)
                return TriageResult(
                    category=cat,
                    confidence=0.95,
                    signals=[signal],
                )
        return None

    async def classify(self, code: str, error: str) -> TriageResult:
        # Try rules first — free and fast
        rule_hit = self.rule_based(error)
        if rule_hit is not None:
            return rule_hit

        # Fall through to LLM
        code_head = "\n".join(code.splitlines()[:40])
        user = TRIAGE_USER.format(error=error.strip()[:1500], code=code_head)
        resp = await self.provider.complete(TRIAGE_SYSTEM, user)

        try:
            data = _extract_json(resp.text)
            cat = ErrorCategory(data.get("category", "generic"))
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("triage parse failed: %s. Defaulting to generic.", e)
            cat = ErrorCategory.GENERIC
            data = {"confidence": 0.3, "signals": ["triage-parse-failed"]}

        return TriageResult(
            category=cat,
            confidence=float(data.get("confidence", 0.5)),
            signals=list(data.get("signals", [])),
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
        )