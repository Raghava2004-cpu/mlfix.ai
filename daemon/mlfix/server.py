"""Local FastAPI daemon — Phase 6c with AWS backend sync."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from .agents.pipeline import Pipeline
from .backend.client import BackendClient
from .mcp_servers.executor import LocalExecutor
from .memory.budget import BudgetManager
from .memory.episodic import EpisodicMemory
from .memory.semantic import SemanticMemory
from .providers.gemini import GeminiProvider
from .router.bandit import ThompsonBandit
from .router.router import Router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("mlfix")


DATA_DIR = Path.home() / ".mlfix"
DB_PATH = DATA_DIR / "mlfix.db"
LANCE_PATH = DATA_DIR / "lancedb"

_pipeline: Pipeline | None = None
_budget: BudgetManager | None = None
_episodic: EpisodicMemory | None = None
_semantic: SemanticMemory | None = None
_bandit: ThompsonBandit | None = None
_backend: BackendClient | None = None


async def _sync_from_backend() -> None:
    """Pull global bandit priors and merge locally. Called on startup and via /sync command."""
    if not _backend or not _backend.enabled or not _bandit:
        return
    data = await _backend.fetch_sync()
    if not data:
        return
    priors = data.get("bandit_arms", [])
    if priors:
        merged = _bandit.apply_global_priors(priors)
        log.info("startup sync: merged %d global priors", merged)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _budget, _episodic, _semantic, _bandit, _backend
    log.info("mlfix daemon starting")
    try:
        _budget = BudgetManager(DB_PATH)
        _episodic = EpisodicMemory(DB_PATH)
        _semantic = SemanticMemory(LANCE_PATH)
        _bandit = ThompsonBandit(DB_PATH)

        # Backend is optional — disabled cleanly if env vars aren't set
        _backend = BackendClient()

        easy = GeminiProvider(model="gemini-2.5-flash-lite")
        medium = GeminiProvider(model="gemini-2.5-flash")
        hard = GeminiProvider(model="gemini-2.5-flash")

        router = Router(easy, medium, hard, budget=_budget, bandit=_bandit)
        executor = LocalExecutor(timeout_s=8.0)

        _pipeline = Pipeline(
            router=router,
            budget=_budget,
            triage_provider=easy,
            critic_provider=medium,
            executor=executor,
            episodic=_episodic,
            semantic=_semantic,
            backend=_backend,
        )
        log.info("Pipeline ready")

        # Kick off startup sync in background — don't block startup on it
        asyncio.create_task(_sync_from_backend())
    except Exception:
        log.exception("Failed to init pipeline")
    yield
    log.info("mlfix daemon shutting down")


app = FastAPI(title="mlfix", version="0.6.0", lifespan=lifespan)


class FixRequest(BaseModel):
    code: str = Field(...)
    error: str = Field(...)
    language: str = Field(default="python")


class FixResponse(BaseModel):
    episode_id: str
    fixed_code: str
    explanation: str
    confidence: float
    model_used: str
    category: str
    stages_run: list[str]
    critic_approved: bool
    critic_issues: list[str]
    executed: bool
    execution_success: bool | None
    execution_stderr: str | None
    judge_success: bool
    judge_reason: str
    total_tokens_in: int
    total_tokens_out: int
    examples_used: int


class FeedbackRequest(BaseModel):
    episode_id: str
    decision: str = Field(..., pattern="^(accept|reject)$")


class UsageResponse(BaseModel):
    day: str
    per_model: list[dict]


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": "0.6.0",
        "pipeline_ready": _pipeline is not None,
        "backend_enabled": _backend.enabled if _backend else False,
    }


@app.get("/usage", response_model=UsageResponse)
async def usage() -> UsageResponse:
    if _budget is None:
        raise HTTPException(503, "Budget manager not ready")
    stats = _budget.get_all_today()
    return UsageResponse(
        day=stats[0].day if stats else "",
        per_model=[
            {
                "model": s.model,
                "requests": s.requests,
                "tokens_in": s.tokens_in,
                "tokens_out": s.tokens_out,
                "daily_limit": _budget.daily_limits.get(s.model),
            }
            for s in stats
        ],
    )


@app.get("/bandit")
async def bandit_snapshot() -> dict:
    if _bandit is None:
        raise HTTPException(503, "Bandit not ready")
    arms = _bandit.snapshot()
    return {
        "arms": [
            {
                "category": a.category,
                "model": a.model,
                "alpha": round(a.alpha, 2),
                "beta": round(a.beta, 2),
                "estimated_success_rate": round(a.mean, 3),
                "total_pulls": a.total_pulls,
            }
            for a in arms
        ],
    }


@app.post("/sync")
async def sync_now() -> dict:
    """Manually trigger a sync from the backend."""
    if _backend is None or not _backend.enabled:
        return {"ok": False, "reason": "backend disabled"}
    data = await _backend.fetch_sync()
    if data is None:
        return {"ok": False, "reason": "backend unreachable"}
    priors = data.get("bandit_arms", [])
    merged = _bandit.apply_global_priors(priors) if _bandit else 0
    return {"ok": True, "priors_merged": merged, "arms_received": len(priors)}


@app.post("/fix", response_model=FixResponse)
async def fix(req: FixRequest) -> FixResponse:
    if _pipeline is None:
        raise HTTPException(503, "Pipeline not ready.")
    try:
        trace, episode_id = await _pipeline.run(req.code, req.error, req.language)
    except RuntimeError as e:
        raise HTTPException(429, str(e)) from e
    except Exception as e:
        log.exception("pipeline failed")
        raise HTTPException(500, str(e)) from e

    exec_ok = None
    exec_err = None
    executed = trace.execution is not None
    if trace.execution:
        exec_ok = (trace.execution.exit_code == 0) and not trace.execution.timed_out
        exec_err = trace.execution.stderr or None

    examples_used = 1 if "retrieve" in trace.stages_run else 0

    return FixResponse(
        episode_id=episode_id,
        fixed_code=trace.fix.fixed_code,
        explanation=trace.fix.explanation,
        confidence=trace.fix.confidence,
        model_used=trace.specialist_used,
        category=trace.category.value,
        stages_run=trace.stages_run,
        critic_approved=trace.critic.approved if trace.critic else True,
        critic_issues=trace.critic.issues if trace.critic else [],
        executed=executed,
        execution_success=exec_ok,
        execution_stderr=exec_err,
        judge_success=trace.judge.success if trace.judge else True,
        judge_reason=trace.judge.reason if trace.judge else "",
        total_tokens_in=trace.total_tokens_in,
        total_tokens_out=trace.total_tokens_out,
        examples_used=examples_used,
    )


@app.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    if _pipeline is None or _episodic is None or _semantic is None:
        raise HTTPException(503, "Not ready")

    ep = _episodic.get(req.episode_id)
    if ep is None:
        raise HTTPException(404, f"Unknown episode: {req.episode_id}")

    if not _episodic.set_user_decision(req.episode_id, req.decision):
        raise HTTPException(500, "Failed to record decision")

    reward = 1.0 if req.decision == "accept" else 0.0
    _pipeline.router.record_outcome(ep.category, ep.model_used, reward)

    if req.decision == "accept" and ep.judge_success:
        _semantic.add(
            episode_id=ep.episode_id,
            category=ep.category,
            error=ep.error,
            original_code=ep.original_code,
            fixed_code=ep.fixed_code,
            explanation=ep.explanation,
        )

    # Sync the updated episode (with user decision) to backend
    if _backend and _backend.enabled:
        updated_ep = _episodic.get(req.episode_id)
        if updated_ep:
            asyncio.create_task(_backend.upload_episode(updated_ep, upload_code=False))

    return {"ok": True, "episode_id": req.episode_id, "decision": req.decision}