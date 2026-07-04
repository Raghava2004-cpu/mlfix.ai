"""Common interface for LLM providers. Every provider returns the same shape."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int


class LLMProvider(Protocol):
    """All providers must implement this."""
    name: str
    model: str

    async def complete(self, system: str, user: str) -> LLMResponse:
        ...