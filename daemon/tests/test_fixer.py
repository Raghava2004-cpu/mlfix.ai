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