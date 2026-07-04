"""Shared types across the agent pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    SHAPE_MISMATCH = "shape_mismatch"
    CUDA_OOM = "cuda_oom"
    DEPENDENCY = "dependency"
    DATA_LEAKAGE = "data_leakage"
    TRAINING_DYNAMICS = "training_dynamics"  # NaN loss, no convergence
    SYNTAX = "syntax"
    IMPORT = "import"
    TYPE = "type"
    GENERIC = "generic"


@dataclass
class TriageResult:
    category: ErrorCategory
    confidence: float
    signals: list[str] = field(default_factory=list)  # what tipped the classifier
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class FixCandidate:
    fixed_code: str
    explanation: str
    confidence: float
    model: str
    tokens_in: int
    tokens_out: int


@dataclass
class CriticVerdict:
    approved: bool
    issues: list[str]
    revised_code: str | None = None  # if critic patched it
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class ExecutionResult:
    ran: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


@dataclass
class JudgeVerdict:
    success: bool
    reason: str
    should_retry: bool = False


@dataclass
class PipelineTrace:
    """Full record of what happened, for the UI and later RL replay buffer."""
    category: ErrorCategory
    triage: TriageResult
    specialist_used: str
    fix: FixCandidate
    critic: CriticVerdict | None = None
    execution: ExecutionResult | None = None
    judge: JudgeVerdict | None = None
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    stages_run: list[str] = field(default_factory=list)