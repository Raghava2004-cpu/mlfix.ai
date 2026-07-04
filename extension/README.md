# mlfix

Fix ML/Python bugs with a multi-agent AI pipeline. Ships with a local daemon that runs on your machine — no data leaves unless you turn on cloud sync.

## What makes it different

Generic AI code fixers guess. mlfix runs a **pipeline**:

1. **Triage** — classifies the error (shape mismatch? CUDA OOM? data leakage?)
2. **Retrieve** — pulls similar past fixes from local vector memory
3. **Specialist** — a prompt tuned for that specific error class
4. **Critic** — reviews the fix for hallucinated APIs
5. **Sandbox** — actually runs the fix and checks it works
6. **Judge** — decides if it's ready to show you

You see a diff. Accept or reject. The system learns from your choice.

## Getting started

1. Install the extension.
2. Get a free API key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
3. Set it in `daemon/.env` (the welcome dialog will guide you).
4. Select buggy code and press **Ctrl+Alt+F** (or **Cmd+Alt+F** on Mac).

## Commands

| Keybind | Command |
|---|---|
| `Ctrl+Alt+F` | Fix selected code |
| `Ctrl+Alt+T` | Fix using an error highlighted in the terminal |
| — | mlfix: Show Menu (opens all commands) |
| — | mlfix: Show Bandit Stats |
| — | mlfix: Show Today's Usage |
| — | mlfix: Sync with Backend |

## Data & privacy

By default, everything runs on your machine. Backend sync is off unless you set `MLFIX_BACKEND_URL`. When on, only hashed metadata is uploaded — never your raw code.

## Feedback

Bugs, features, questions: [GitHub Issues](https://github.com/yourname/mlfix/issues).

## License

MIT