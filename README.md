# mlfix

A VS Code extension that fixes ML/Python code using a multi-agent pipeline with reinforcement learning, prompt retrieval, and cloud-synced cross-user learning.

Built with Gemini (free tier), FastAPI, LanceDB, and AWS (Lambda / DynamoDB / S3 / API Gateway).

![status](https://img.shields.io/badge/status-active-brightgreen)
![python](https://img.shields.io/badge/python-3.11-blue)
![vscode](https://img.shields.io/badge/vscode-1.85+-blue)
![license](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Select buggy code, hit `Ctrl+Alt+F`, paste the error (or use `Ctrl+Alt+T` to grab it from your terminal). A multi-agent pipeline classifies the error, retrieves similar past fixes, picks the right model, generates a fix, reviews it, executes it in a sandbox, and shows you a diff. Accept or reject — either signal trains the system.

## Architecture

VS Code Extension (TypeScript)
↓ HTTP
Local Daemon (Python, FastAPI)
├── Triage Agent      (categorises the error)
├── Retrieval         (LanceDB vector search over past fixes)
├── Router            (Thompson Sampling bandit picks a model)
├── Specialist Agent  (category-specific prompt)
├── Critic Agent      (reviews the fix)
├── Sandboxed Executor (runs the fix, checks it)
└── Judge             (accept / reject decision)
↓ async, opt-in
AWS Backend
├── API Gateway → Lambda (ingest, sync)
├── DynamoDB    (episodes, global bandit arms, prompt registry)
├── S3          (replay buffer)
└── EventBridge → Lambda (nightly aggregator)

## Why this design

- **Multi-agent** because a single prompt can't cover shape mismatches, CUDA OOM, dependency conflicts, and NaN losses equally well.
- **Bandit routing** because different models cost different amounts and win on different problems. We learn which is best per category.
- **Local-first** so it works offline, with your data on your machine by default.
- **Cloud sync** because a new user shouldn't start cold — they inherit priors from everyone else's successful fixes.
- **Sandboxed execution** so the pipeline can *verify* fixes, not just generate them.

## Features

- Multi-agent pipeline: **triage → retrieve → specialist → critic → execute → judge**
- Specialists for: `shape_mismatch`, `cuda_oom`, `dependency`, `data_leakage`, `training_dynamics`, `syntax`, `import`, `type`, `generic`
- **Thompson Sampling** bandit that learns per-category model preferences
- **Vector retrieval** of similar past fixes (LanceDB, `all-MiniLM-L6-v2`)
- **Token accounting** with daily free-tier caps and automatic fallback
- **Cloud sync** (AWS) with privacy-preserving hashed uploads
- Diff-based review UI with accept / reject feedback
- Status bar, terminal-error capture (`Ctrl+Alt+T`), onboarding flow

## Quick start

### Prerequisites

- Node.js 20+
- Python 3.11+
- [uv](https://astral.sh/uv) for Python dependency management
- A free [Gemini API key](https://aistudio.google.com/app/apikey)
- (Optional) AWS account with CDK bootstrapped for cloud sync

### Install

```bash
git clone https://github.com/yourname/mlfix.git
cd mlfix

# Set up the daemon
cd daemon
uv sync
cp .env.example .env    # then edit .env and paste your GEMINI_API_KEY

# Set up the extension
cd ../extension
npm install
```

### Run

Open the `extension/` folder in VS Code and press **F5**. A dev host window opens with mlfix loaded.

- `Ctrl+Alt+F` — fix selected code (paste error when prompted)
- `Ctrl+Alt+T` — fix using an error you've selected in the terminal
- Click the status bar `mlfix` item for the full menu

### Deploy the AWS backend (optional)

```bash
cd aws
npm install
export AWS_PROFILE=mlfix     # PowerShell: $env:AWS_PROFILE = "mlfix"
npx cdk deploy
```

CDK prints an API URL and key ID. Fetch the key value and paste both into `daemon/.env`:

```bash
aws apigateway get-api-key --api-key <KEY_ID> --include-value --profile mlfix
```

To tear it down: `npx cdk destroy`.

## Repo layout

mlfix/
├── extension/       VS Code extension (TypeScript)
├── daemon/          Local FastAPI daemon (Python)
│   └── mlfix/
│       ├── agents/      triage, fixer, critic, judge, pipeline
│       ├── router/      Thompson bandit + complexity classifier
│       ├── memory/      episodic (SQLite), semantic (LanceDB), budget
│       ├── providers/   Gemini (extensible)
│       ├── backend/     AWS sync client
│       └── mcp_servers/ sandboxed executor
├── aws/             CDK infra + Lambda functions
└── prompts/         (reserved for versioned prompt templates)

## Development

```bash
# Run all tests
cd daemon
uv run pytest -v

# Format
uv run ruff format .
```

## Data & privacy

- All data lives in `~/.mlfix/` on your machine by default
- Cloud sync is **off unless** `MLFIX_BACKEND_URL` is set in `daemon/.env`
- When on: only anonymised metadata leaves your machine. Raw code stays local by default; a 16-char SHA-256 hash is uploaded so we can detect duplicates
- Cloud data expires after 30 days (S3 lifecycle rule)

## Roadmap

- [ ] Add OpenAI + Anthropic providers alongside Gemini
- [ ] DPO-style prompt refinement from the replay buffer
- [ ] MCP server over stdio (currently in-process)
- [ ] Batch mode for CI: fix an entire failing test suite

## License

MIT — see [LICENSE](./LICENSE).

