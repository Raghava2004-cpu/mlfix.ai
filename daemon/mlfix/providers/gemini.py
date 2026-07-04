"""Gemini provider using the google-genai SDK."""
from __future__ import annotations

import logging
import os

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import LLMResponse

log = logging.getLogger("mlfix.gemini")


class RetriableGeminiError(Exception):
    """Raised on temporary Gemini failures so tenacity can retry."""


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to daemon/.env"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type(RetriableGeminiError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def complete(self, system: str, user: str) -> LLMResponse:
        try:
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.2,   # low: we want deterministic fixes
                    max_output_tokens=2048,
                ),
            )
        except Exception as e:
            msg = str(e).lower()
            if (
                "429" in msg
                or "503" in msg
                or "resource_exhausted" in msg
                or "unavailable" in msg
                or "rate" in msg
            ):
                log.warning("Gemini temporary failure, will retry: %s", e)
                raise RetriableGeminiError(str(e)) from e
            raise

        text = resp.text or ""
        usage = resp.usage_metadata
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out = getattr(usage, "candidates_token_count", 0) or 0

        log.info(
            "gemini call: model=%s tokens_in=%d tokens_out=%d",
            self.model, tokens_in, tokens_out,
        )

        return LLMResponse(
            text=text,
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
