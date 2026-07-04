import json
import urllib.request
import urllib.error
import time

base = 'http://127.0.0.1:8765'

cases = [
    {
        'code': 'import numpy as np\n\narr = np.array([1, 2, 3])\nprint(arr + 1)',
        'error': 'ValueError: shape mismatch',
        'language': 'python',
    },
    {
        'code': 'def add(a, b):\n    return a + b\n\nprint(add(1, 2))',
        'error': 'ValueError: shape mismatch',
        'language': 'python',
    },
    {
        'code': 'x = [1, 2, 3]\nprint(x[0] + 1)',
        'error': 'ValueError: shape mismatch',
        'language': 'python',
    },
    {
        'code': 'import torch\n\nprint(torch.tensor([1, 2, 3]).sum())',
        'error': 'RuntimeError: shape mismatch',
        'language': 'python',
    },
    {
        'code': 'import numpy as np\n\narr = np.array([[1, 2], [3, 4]])\nprint(arr.T)',
        'error': 'ValueError: shape mismatch',
        'language': 'python',
    },
]

for idx, payload in enumerate(cases, 1):
    req = urllib.request.Request(
        f'{base}/fix',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    print(f'[{idx}/5] sending fix request...')
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.load(resp)
    print(f'  category={data["category"]} model={data["model_used"]} examples={data["examples_used"]}')
    decision = 'accept' if idx % 2 == 1 else 'reject'
    fb_req = urllib.request.Request(
        f'{base}/feedback',
        data=json.dumps({'episode_id': data['episode_id'], 'decision': decision}).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(fb_req, timeout=1800) as resp:
        fb = json.load(resp)
    print(f'  feedback={fb}')
    time.sleep(1)
