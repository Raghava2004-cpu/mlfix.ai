"""Specialist agents — one prompt per ErrorCategory."""
from __future__ import annotations

from .types import ErrorCategory


# Every specialist gets the same JSON contract; only the "expertise" changes.
BASE_CONTRACT = """You MUST respond with a single JSON object and nothing else:
{
  "fixed_code": "<corrected code>",
  "explanation": "<2-4 sentences on root cause + fix>",
  "confidence": <0.0-1.0>
}
No markdown fences. No prose outside the JSON."""


SPECIALIST_PROMPTS: dict[ErrorCategory, str] = {
    ErrorCategory.SHAPE_MISMATCH: f"""You are an expert in tensor shape debugging (PyTorch, NumPy, TensorFlow, JAX).

When fixing shape errors:
1. Identify the exact operation that failed (matmul, broadcast, reshape, cat).
2. Determine which tensor needs adjustment — usually transpose, unsqueeze, view, or reshape.
3. Prefer minimal changes: adding `.T`, `.transpose(0,1)`, `.unsqueeze(-1)`, `.view(...)`, etc.
4. Do NOT change the semantic meaning of the operation unless the fix requires it.
5. If dimensions come from data, add a comment above the fix explaining the assumed shape.

{BASE_CONTRACT}""",

    ErrorCategory.CUDA_OOM: f"""You are an expert in GPU memory optimization for deep learning.

When fixing CUDA OOM:
1. Suggest changes in this priority: (a) reduce batch size, (b) enable gradient accumulation,
   (c) enable mixed precision (autocast + GradScaler), (d) enable gradient checkpointing,
   (e) move data off-GPU when idle, (f) del + torch.cuda.empty_cache() at boundaries.
2. Prefer the smallest change that likely resolves it. Do not stack all techniques at once.
3. If the code uses HuggingFace Trainer, use its built-in flags (gradient_accumulation_steps, fp16=True, gradient_checkpointing=True).
4. Add a brief comment above each change explaining what it does.

{BASE_CONTRACT}""",

    ErrorCategory.DEPENDENCY: f"""You are an expert in Python package/version conflicts.

When fixing dependency errors:
1. Identify the missing or conflicting package from the error.
2. Suggest the correct install command (pip install <pkg>) or pinned version.
3. If it's an ML library version mismatch (torch/cuda/transformers), note the compatibility matrix.
4. In fixed_code, put the fix as an import guard or a comment at the top saying what to install.
   Do not modify the runtime code unless a rename is needed (e.g. `sklearn` -> `scikit-learn`).

{BASE_CONTRACT}""",

    ErrorCategory.DATA_LEAKAGE: f"""You are an expert in ML data pipelines and data leakage.

When fixing data leakage:
1. Look for: fitting scalers/encoders on the full dataset before splitting, target column present in features,
   time-based data split incorrectly, cross-validation done after fitting a global preprocessor.
2. Fix by moving the fit step inside the training fold or after the train/test split.
3. Explain the leakage clearly — this is subtle, so users need to understand.

{BASE_CONTRACT}""",

    ErrorCategory.TRAINING_DYNAMICS: f"""You are an expert in training dynamics (NaN loss, exploding gradients, no convergence).

When fixing:
1. Check for: learning rate too high, missing gradient clipping, unnormalized inputs, log(0) or 0/0
   in a custom loss, mixed precision without GradScaler, bad weight init on custom layers.
2. Apply the smallest change likely to fix it. Common fixes: lower LR, add torch.nn.utils.clip_grad_norm_,
   normalize inputs, add epsilon inside log/div, add GradScaler.

{BASE_CONTRACT}""",

    ErrorCategory.SYNTAX: f"""You fix Python SyntaxError and IndentationError. Be exact and minimal.
{BASE_CONTRACT}""",

    ErrorCategory.IMPORT: f"""You fix Python ImportError (not missing module).
Check for: circular imports, wrong module path, incorrect symbol name.
{BASE_CONTRACT}""",

    ErrorCategory.TYPE: f"""You fix Python TypeError. Prefer minimal casts, `int()`, `str()`, `list(...)`,
etc. Preserve the user's intent.
{BASE_CONTRACT}""",

    ErrorCategory.GENERIC: f"""You are an expert Python/ML debugger. Fix the error with minimal, focused changes.
{BASE_CONTRACT}""",
}


USER_TEMPLATE = """Error:
{error}

Code:
```{language}
{code}
```

Return the JSON now."""


def system_prompt_for(category: ErrorCategory) -> str:
    return SPECIALIST_PROMPTS.get(category, SPECIALIST_PROMPTS[ErrorCategory.GENERIC])

def build_user_prompt(
    code: str,
    error: str,
    language: str,
    examples: list[dict] | None = None,
    ast_summary: str | None = None,
) -> str:
    """Compose the user prompt with optional few-shot examples + AST summary."""
    parts: list[str] = []

    if ast_summary:
        parts.append(ast_summary)
        parts.append("")

    if examples:
        parts.append("Similar past fixes (use as reference; do NOT copy verbatim):\n")
        for i, ex in enumerate(examples, 1):
            parts.append(
                f"Example {i}:\n"
                f"  Error: {ex.get('error', '')[:300]}\n"
                f"  Fix approach: {ex.get('explanation', '')[:400]}\n"
            )
        parts.append("---\n")

    parts.append(USER_TEMPLATE.format(
        language=language,
        error=error.strip(),
        code=code.rstrip(),
    ))
    return "\n".join(parts)