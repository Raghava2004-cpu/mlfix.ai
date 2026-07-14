from mlfix.analysis.ast_analyzer import analyze


def test_extracts_imports():
    code = "import torch\nimport numpy as np\nfrom typing import List"
    r = analyze(code)
    assert r.syntax_valid
    assert "torch" in r.imports
    assert "numpy as np" in r.imports


def test_extracts_functions():
    code = "def train(model, data):\n    pass"
    r = analyze(code)
    assert any("train" in f for f in r.functions)


def test_extracts_classes():
    code = "class MyModel(nn.Module):\n    pass"
    r = analyze(code)
    assert any("MyModel" in c for c in r.classes)


def test_detects_undefined_name():
    code = "x = missing_variable + 1"
    r = analyze(code)
    assert "missing_variable" in r.undefined_names


def test_handles_syntax_error():
    code = "def broken(:\n    pass"
    r = analyze(code)
    assert not r.syntax_valid
    assert r.syntax_error is not None


def test_prompt_summary_format():
    code = "import torch\ndef fn(x): return x"
    r = analyze(code)
    summary = r.to_prompt_summary()
    assert "[AST ANALYSIS]" in summary
    assert "torch" in summary