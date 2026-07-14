"""Find files related to the one being fixed.

Given a Python file path, we:
1. Parse the file's imports.
2. For each 'from X import Y' or 'import X', try to resolve X to a file in the same repo.
3. Read those files, cap the total content at a token budget.

We only follow relative-looking imports (same package). We do NOT try to read
site-packages / third-party libraries.
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("mlfix.repo_context")


# Rough token estimate — 4 chars ≈ 1 token for code. Under-estimate on purpose so we stay safe.
def _estimate_tokens(s: str) -> int:
    return len(s) // 4


@dataclass
class RelatedFile:
    path: str          # relative to repo root
    content: str
    reason: str        # why we included it (e.g. "imported as 'model'")


@dataclass
class RepoContext:
    target_file: str | None = None
    related: list[RelatedFile] = field(default_factory=list)
    total_tokens: int = 0
    truncated: bool = False

    def to_prompt_block(self) -> str:
        if not self.related:
            return ""
        parts = ["[RELATED FILES]", ""]
        for f in self.related:
            parts.append(f"# ── {f.path}  ({f.reason}) ──")
            parts.append(f.content)
            parts.append("")
        if self.truncated:
            parts.append("[note: some related files were truncated to stay in budget]")
        return "\n".join(parts)


def _extract_import_modules(code: str) -> list[str]:
    """Return the top-level module names imported by this code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module)
    return mods


def _resolve_to_local_file(module_name: str, target_file: Path, repo_root: Path) -> Path | None:
    """Try to find a local .py file for this import. Returns None for stdlib/third-party."""
    parts = module_name.split(".")

    # Candidate 1: sibling of target file
    candidate = target_file.parent / f"{parts[-1]}.py"
    if candidate.exists() and candidate.is_file():
        return candidate

    # Candidate 2: dotted path from repo root
    dotted = repo_root.joinpath(*parts).with_suffix(".py")
    if dotted.exists() and dotted.is_file():
        return dotted

    # Candidate 3: package __init__.py at that path
    pkg_init = repo_root.joinpath(*parts) / "__init__.py"
    if pkg_init.exists() and pkg_init.is_file():
        return pkg_init

    return None


def _find_repo_root(start: Path) -> Path:
    """Walk up looking for a marker file (.git, pyproject.toml, setup.py)."""
    markers = {".git", "pyproject.toml", "setup.py", "requirements.txt"}
    cur = start.parent if start.is_file() else start
    for _ in range(8):  # cap walking
        if any((cur / m).exists() for m in markers):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parent if start.is_file() else start


def gather(
    code: str,
    target_file_path: str | None,
    token_budget: int = 3000,
    max_files: int = 4,
) -> RepoContext:
    """Return related files for the given code.

    - If target_file_path is None (unsaved snippet), returns empty context.
    - Only includes files under the same repo root.
    - Stops adding files once token_budget or max_files is reached.
    """
    ctx = RepoContext(target_file=target_file_path)
    if not target_file_path:
        return ctx

    try:
        target = Path(target_file_path).resolve()
    except Exception as e:
        log.warning("bad target path: %s", e)
        return ctx

    if not target.exists():
        log.info("target file does not exist on disk: %s", target)
        return ctx

    repo_root = _find_repo_root(target)
    log.info("repo root: %s", repo_root)

    modules = _extract_import_modules(code)
    log.info("modules to resolve: %s", modules[:10])

    seen: set[Path] = set()
    remaining = token_budget

    for mod in modules:
        if len(ctx.related) >= max_files:
            break
        resolved = _resolve_to_local_file(mod, target, repo_root)
        if resolved is None or resolved == target or resolved in seen:
            continue

        # Refuse to read files outside repo root — safety
        try:
            resolved.relative_to(repo_root)
        except ValueError:
            log.warning("skipping file outside repo root: %s", resolved)
            continue

        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.warning("could not read %s: %s", resolved, e)
            continue

        tokens = _estimate_tokens(content)
        if tokens > remaining:
            # truncate to what's left in the budget
            char_limit = remaining * 4
            if char_limit < 200:
                ctx.truncated = True
                break
            content = content[:char_limit] + "\n# ...[truncated]"
            tokens = _estimate_tokens(content)
            ctx.truncated = True

        rel_path = str(resolved.relative_to(repo_root))
        ctx.related.append(RelatedFile(
            path=rel_path,
            content=content,
            reason=f"imported as '{mod}'",
        ))
        seen.add(resolved)
        remaining -= tokens
        ctx.total_tokens += tokens

    log.info("gathered %d related files, ~%d tokens", len(ctx.related), ctx.total_tokens)
    return ctx