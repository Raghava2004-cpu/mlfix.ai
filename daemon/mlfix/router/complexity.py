"""Classify how hard an error is. Rule-based for Phase 3. Bandit-based in Phase 5."""
from __future__ import annotations

import re
from enum import Enum


class Complexity(str, Enum):
    EASY = "easy"       # syntax errors, typos, undefined names
    MEDIUM = "medium"   # standard runtime errors, type errors
    HARD = "hard"       # ML-specific: shapes, CUDA, training dynamics


# Signals that push toward HARD
HARD_PATTERNS = [
    r"cuda\s+out of memory",
    r"cuda\s+error",
    r"mat1 and mat2",
    r"size mismatch",
    r"shape.*mismatch",
    r"can't multiply",
    r"expected.*got.*tensor",
    r"gradient",
    r"backward",
    r"nan.*loss|loss.*nan",
    r"dataloader",
    r"distributed",
    r"nccl",
    r"deadlock",
]

# Signals that push toward EASY
EASY_PATTERNS = [
    r"^\s*syntaxerror",
    r"^\s*indentationerror",
    r"invalid syntax",
    r"nameerror.*is not defined",
    r"modulenotfounderror",
    r"importerror",
    r"unexpected indent",
]


def classify(error: str, code: str = "") -> Complexity:
    text = f"{error}\n{code}".lower()

    for p in HARD_PATTERNS:
        if re.search(p, text):
            return Complexity.HARD

    for p in EASY_PATTERNS:
        if re.search(p, text):
            return Complexity.EASY

    # default to MEDIUM
    return Complexity.MEDIUM