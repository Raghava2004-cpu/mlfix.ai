"""Thin HTTP client for the mlfix AWS backend. Fails soft — never breaks local flow."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import asdict
from typing import Any

import httpx

from ..memory.episodic import Episode

log = logging.getLogger("mlfix.backend")


class BackendClient:
    """
    Async HTTP client with graceful degradation.
    All methods return either the parsed body or None on failure.
    They log warnings but never raise.
    """
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        user_id: str | None = None,
        timeout_s: float = 5.0,
        enabled: bool = True,
    ) -> None:
        self.base_url = (
            os.getenv("MLFIX_BACKEND_URL", "") if base_url is None else base_url
        ).rstrip("/")
        self.api_key = (
            os.getenv("MLFIX_BACKEND_API_KEY", "") if api_key is None else api_key
        )
        self.user_id = (
            os.getenv("MLFIX_USER_ID", "anonymous") if user_id is None else user_id
        )
        self.enabled = enabled and bool(self.base_url) and bool(self.api_key)
        self.timeout_s = timeout_s

        if not self.enabled:
            log.info("backend disabled (missing URL/key or explicitly off)")
        else:
            log.info("backend enabled: %s user=%s", self.base_url, self.user_id)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _hash_code(code: str) -> str:
        """We upload a hash instead of raw code — privacy-preserving."""
        return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]

    async def upload_episode(self, ep: Episode, upload_code: bool = False) -> bool:
        """Send an episode to the backend. Returns True on success."""
        if not self.enabled:
            return False

        payload_ep = asdict(ep)
        # Privacy: strip raw code by default, send only a hash
        payload_ep["code_hash"] = self._hash_code(ep.original_code)
        if not upload_code:
            payload_ep.pop("original_code", None)
            payload_ep.pop("fixed_code", None)

        body = {"user_id": self.user_id, "episode": payload_ep}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(
                    f"{self.base_url}/episodes",
                    headers=self._headers(),
                    json=body,
                )
            if r.status_code >= 400:
                log.warning("upload_episode failed: %d %s", r.status_code, r.text[:200])
                return False
            return True
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            log.warning("upload_episode error: %s", e)
            return False

    async def fetch_sync(self) -> dict[str, Any] | None:
        """Fetch bandit priors + prompt overrides."""
        if not self.enabled:
            return None
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.get(
                    f"{self.base_url}/sync",
                    headers=self._headers(),
                )
            if r.status_code >= 400:
                log.warning("fetch_sync failed: %d %s", r.status_code, r.text[:200])
                return None
            return r.json()
        except (httpx.RequestError, asyncio.TimeoutError) as e:
            log.warning("fetch_sync error: %s", e)
            return None
