"""Score mlfix on all OpenEnv tasks. Feeds rewards into the bandit."""
import asyncio
from ..agents.pipeline import Pipeline
from .openenv_client import OpenEnvClient


async def benchmark(pipeline: Pipeline, env: OpenEnvClient, task_id: int):
    obs = await env.reset(task_id)
    buggy_code = obs["buggy_code"]
    error = obs["description"]  # or synthesize from n_bugs

    # Run mlfix pipeline
    trace, _ = await pipeline.run(buggy_code, error, "python")

    # Submit the fix to the OpenEnv
    result = await env.step("submit_fix", code=trace.fix.fixed_code)
    reward = result["reward"]  # 0.0 to 1.0

    # THE KEY LINE: update the bandit with real reward, not just binary accept/reject
    pipeline.router.record_outcome(
        trace.category.value,
        trace.specialist_used,
        reward,  # <-- fine-grained reward from your OpenEnv
    )

    return {
        "task_id": task_id,
        "difficulty": obs["difficulty"],
        "reward": reward,
        "correctness": result.get("sub_scores", {}).get("correctness"),
        "explanation": result.get("sub_scores", {}).get("explanation"),
        "model_used": trace.specialist_used,
    }


async def main():
    from ..server import create_pipeline

    _pipeline, *_ = create_pipeline()
    env = OpenEnvClient()
    results = []
    for task_id in [0, 1, 2]:
        for run in range(5):  # 5 runs per task for statistical reliability
            r = await benchmark(_pipeline, env, task_id)
            results.append(r)
            print(f"task={task_id} run={run} reward={r['reward']:.3f}")
    return results


if __name__ == "__main__":
    asyncio.run(main())
