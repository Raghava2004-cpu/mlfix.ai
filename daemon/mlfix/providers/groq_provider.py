"""Groq provider — fast, free-tier friendly (14k req/day)."""
from __future__ import annotations

import logging
import os

from groq import AsyncGroq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .base import LLMResponse

log = logging.getLogger("mlfix.groq")


class GroqRateLimitError(Exception):
    """Raised on 429 so tenacity can retry."""


class GroqProvider:
    name = "groq"

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to daemon/.env")
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    @retry(
        retry=retry_if_exception_type(GroqRateLimitError),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def complete(self, system: str, user: str) -> LLMResponse:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "quota" in msg:
                log.warning("groq rate limited, will retry: %s", e)
                raise GroqRateLimitError(str(e)) from e
            raise

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        log.info(
            "groq call: model=%s tokens_in=%d tokens_out=%d",
            self.model, tokens_in, tokens_out,
        )

        return LLMResponse(
            text=text,
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )