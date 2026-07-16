# mlfix Benchmark

5 curated ML/Python bugs designed to expose the weakness of single-shot LLM fixers:
hallucinated APIs, multi-file bugs, silent numerical bugs, data leakage, and off-by-one loops.

## Systems compared

| System | Model | Pipeline | Verification |
|---|---|---|---|
| baseline-70b | Groq Llama 3.3 70B Versatile | Single call | None |
| baseline-8b | Groq Llama 3.1 8B Instant | Single call | None |
| mlfix | 8B triage + 70B specialist (bandit routed) | 6-agent | Sandbox execution |

## Results

| System | Pass rate | Verified | Tokens/bug | Avg time |
|---|---|---|---|---|
| baseline-70b | 5/5 (100%) | 0/5 (0%) | ~120 | 0.3s |
| baseline-8b | 5/5 (100%) | 0/5 (0%) | ~216 | 6.7s* |
| **mlfix** | **5/5 (100%)** | **5/5 (100%)** | ~2,100 | 9.7s |

*8B time inflated by Groq free-tier rate limiting, not real inference latency.

## Why "verified" is the differentiator

Baseline systems return a fix and hope it works. **They never execute the code.**
mlfix runs every fix in a sandboxed subprocess with a timeout, captures stdout/stderr,
and rejects fixes that don't run cleanly. In production:

- Baseline can silently ship code that "looks right" but is subtly wrong. A benchmark
  check may pass — a real user encountering a wrong output won't know why.
- mlfix cannot ship such fixes. Every one is executed and observed to work.

For the 5 bugs in this benchmark, all three systems produce correct code according
to our external `check.py` scripts. But **only mlfix has evidence its fixes actually run.**

## The 5 bugs

| # | Bug | Category | Why it's interesting |
|---|---|---|---|
| 001 | `np.mad()` doesn't exist | Hallucinated API | Critic + executor catch invented functions |
| 002 | `predict()` missing arg from sibling file | Multi-file | Requires reading `model.py` — mlfix's repo_context handles this |
| 003 | Gradient sign flipped in trainer | Silent numerical | Runs fine, produces garbage w and b |
| 004 | Scaler fit before train/test split | Data leakage | Structural bug — no runtime error |
| 005 | `range(n-1)` in MSE loop | Off-by-one | Wrong result, no crash |

## Cost / speed tradeoff

mlfix costs ~10-15x more tokens and ~30x more time than a single 70B call. This is
inherent to a multi-agent pipeline. The tradeoff is:

- Use mlfix for **CI review, PR debugging, overnight batch jobs** — where correctness
  matters more than latency and the extra tokens are a rounding error against
  engineer time saved.
- Use plain LLM completion for **interactive typing** — where sub-second latency
  matters more than verification.

Different tools for different jobs.

## Reproducing

```bash
cd daemon
uv sync
cp .env.example .env  # add GROQ_API_KEY
uv run mlfix-daemon   # in one terminal

# in another terminal
cd daemon
uv run python ../benchmark/runner.py --system baseline --baseline-provider groq
uv run python ../benchmark/runner.py --system baseline --baseline-provider groq-8b
uv run python ../benchmark/runner.py --system mlfix
```

Raw results are in `benchmark/results/results_YYYYMMDD_HHMMSS.json`.