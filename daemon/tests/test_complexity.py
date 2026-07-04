from mlfix.router.complexity import Complexity, classify


def test_syntax_error_is_easy():
    assert classify("SyntaxError: invalid syntax") == Complexity.EASY


def test_name_error_is_easy():
    assert classify("NameError: name 'foo' is not defined") == Complexity.EASY


def test_shape_mismatch_is_hard():
    err = "RuntimeError: mat1 and mat2 shapes cannot be multiplied (3x4 and 5x4)"
    assert classify(err) == Complexity.HARD


def test_cuda_oom_is_hard():
    err = "RuntimeError: CUDA out of memory"
    assert classify(err) == Complexity.HARD


def test_generic_type_error_is_medium():
    assert classify("TypeError: unsupported operand type(s)") == Complexity.MEDIUM