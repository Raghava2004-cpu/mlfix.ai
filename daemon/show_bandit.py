import urllib.request
with urllib.request.urlopen('http://127.0.0.1:8765/bandit', timeout=60) as r:
    print(r.read().decode())
