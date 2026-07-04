from mlfix.agents.triage import TriageAgent
from mlfix.agents.types import ErrorCategory


class _StubProvider:
    name = "stub"
    model = "stub"
    async def complete(self, system, user):
        raise AssertionError("should not be called; rule-based should hit")


def test_rule_catches_shape():
    agent = TriageAgent(_StubProvider())
    r = agent.rule_based("RuntimeError: mat1 and mat2 shapes cannot be multiplied")
    assert r is not None
    assert r.category == ErrorCategory.SHAPE_MISMATCH


def test_rule_catches_oom():
    agent = TriageAgent(_StubProvider())
    r = agent.rule_based("RuntimeError: CUDA out of memory")
    assert r is not None
    assert r.category == ErrorCategory.CUDA_OOM


def test_rule_misses_generic():
    agent = TriageAgent(_StubProvider())
    r = agent.rule_based("ValueError: some obscure thing")
    assert r is None