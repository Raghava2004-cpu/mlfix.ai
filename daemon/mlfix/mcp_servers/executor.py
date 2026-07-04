"""MCP-style local executor for Python snippets.

We're not running a full MCP server-over-stdio yet — the pipeline calls this
in-process. Interface is designed so it can be lifted into a real MCP server
later without changes.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path

log = logging.getLogger("mlfix.executor")


class LocalExecutor:
    """Runs Python code in a subprocess with a timeout."""

    def __init__(self, timeout_s: float = 10.0, python_bin: str | None = None) -> None:
        self.timeout_s = timeout_s
        self.python_bin = python_bin or sys.executable

    async def run_python(self, code: str) -> dict:
        """Run `code` and return {exit_code, stdout, stderr, duration_ms, timed_out}."""
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="mlfix_") as td:
            script = Path(td) / "snippet.py"
            script.write_text(code, encoding="utf-8")

            log.info("executing snippet: %d bytes, timeout=%.1fs", len(code), self.timeout_s)

            proc = await asyncio.create_subprocess_exec(
                self.python_bin, str(script),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=td,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout_s,
                )
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout, stderr = b"", b"[mlfix] execution timed out"
                timed_out = True

        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "exit_code": proc.returncode if not timed_out else None,
            "stdout": stdout.decode("utf-8", errors="replace")[:4000],
            "stderr": stderr.decode("utf-8", errors="replace")[:4000],
            "duration_ms": duration_ms,
            "timed_out": timed_out,
        }