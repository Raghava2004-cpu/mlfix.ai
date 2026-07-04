from mlfix.agents.judge import JudgeAgent
from mlfix.agents.types import CriticVerdict, ExecutionResult


def test_critic_rejects_means_failure():
    j = JudgeAgent()
    v = j.decide(CriticVerdict(approved=False, issues=["bad"]), execution=None)
    assert not v.success
    assert v.should_retry


def test_no_exec_trust_critic():
    j = JudgeAgent()
    v = j.decide(CriticVerdict(approved=True, issues=[]), execution=None)
    assert v.success


def test_exec_success():
    j = JudgeAgent()
    exec_r = ExecutionResult(ran=True, exit_code=0, stdout="ok", stderr="", duration_ms=10)
    v = j.decide(CriticVerdict(approved=True, issues=[]), execution=exec_r)
    assert v.success


def test_exec_failure():
    j = JudgeAgent()
    exec_r = ExecutionResult(ran=True, exit_code=1, stdout="", stderr="err", duration_ms=10)
    v = j.decide(CriticVerdict(approved=True, issues=[]), execution=exec_r)
    assert not v.success
    assert v.should_retry