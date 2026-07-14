import tempfile
from pathlib import Path

from mlfix.analysis.repo_context import gather


def test_no_file_path_returns_empty():
    ctx = gather("import x", target_file_path=None)
    assert ctx.related == []


def test_nonexistent_file_returns_empty():
    ctx = gather("import x", target_file_path="/nonexistent/path.py")
    assert ctx.related == []


def test_finds_sibling_file():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text("")  # repo marker
        (root / "model.py").write_text("class Model:\n    pass\n")
        main = root / "main.py"
        main.write_text("from model import Model\n\nm = Model()")

        ctx = gather(main.read_text(), str(main))
        assert len(ctx.related) == 1
        assert "model.py" in ctx.related[0].path
        assert "class Model" in ctx.related[0].content


def test_respects_token_budget():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text("")
        # A huge sibling file
        big_content = "# padding line\n" * 5000
        (root / "big.py").write_text(big_content)
        main = root / "main.py"
        main.write_text("import big\n")

        ctx = gather(main.read_text(), str(main), token_budget=100)
        # Either truncated or excluded — but must not blow the budget
        assert ctx.total_tokens <= 200  # small slack


def test_skips_stdlib_imports():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text("")
        main = root / "main.py"
        main.write_text("import os\nimport sys\n")

        ctx = gather(main.read_text(), str(main))
        # No local os.py or sys.py exists → nothing included
        assert ctx.related == []


def test_prompt_block_when_empty():
    ctx = gather("x = 1", target_file_path=None)
    assert ctx.to_prompt_block() == ""