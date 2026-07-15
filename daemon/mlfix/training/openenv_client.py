"""Client that treats mlfix as an OpenEnv agent."""
import httpx


class OpenEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    async def reset(self, task_id: int) -> dict:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{self.base_url}/reset", json={"task_id": task_id})
        return r.json()

    async def step(self, action_type: str, **fields) -> dict:
        payload = {"action_type": action_type, **fields}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{self.base_url}/step", json=payload)
        return r.json()