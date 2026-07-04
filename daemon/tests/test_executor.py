import asyncio
import pytest

from mlfix.mcp_servers.executor import LocalExecutor


@pytest.mark.asyncio
async def test_executor_success():
    ex = LocalExecutor(timeout_s=5.0)
    result = await ex.run_python("print('hello')")
    assert result["exit_code"] == 0
    assert "hello" in result["stdout"]
    assert not result["timed_out"]


@pytest.mark.asyncio
async def test_executor_failure():
    ex = LocalExecutor(timeout_s=5.0)
    result = await ex.run_python("raise ValueError('boom')")
    assert result["exit_code"] != 0
    assert "ValueError" in result["stderr"]


@pytest.mark.asyncio
async def test_executor_timeout():
    ex = LocalExecutor(timeout_s=1.0)
    result = await ex.run_python("import time; time.sleep(5)")
    assert result["timed_out"]
    assert result["exit_code"] is None