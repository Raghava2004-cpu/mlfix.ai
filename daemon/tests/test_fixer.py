from mlfix.agents.fixer import _extract_json


def test_extract_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    text = '```json\n{"a": 1}\n```'
    assert _extract_json(text) == {"a": 1}


def test_extract_bare_fence():
    text = '```\n{"a": 1}\n```'
    assert _extract_json(text) == {"a": 1}


def test_extract_with_preamble():
    text = 'Here is the fix:\n{"a": 1}\nThanks!'
    assert _extract_json(text) == {"a": 1}
    
def test_extract_with_raw_newlines_in_string():
    """Model output with literal newlines inside a JSON string value."""
    text = '''{
      "fixed_code": "def foo(): pass",
      "explanation": "fixed",
      "confidence": 1.0
    }'''
    result = _extract_json(text)
    assert "def foo()" in result["fixed_code"]
    assert result["explanation"] == "fixed"    

def test_extract_with_inner_python_fence():
    """Model wraps its code in ```python fences INSIDE the fixed_code value."""
    text = '''{
      "fixed_code": "```python def foo(): pass ```",
      "explanation": "fixed",
      "confidence": 1.0
    }'''
    result = _extract_json(text)
    assert "def foo()" in result["fixed_code"]
    assert "```" not in result["fixed_code"]    