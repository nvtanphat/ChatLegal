from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Generate a text response from a prompt."""

    @abstractmethod
    def healthcheck(self) -> bool:
        """Return True when model backend is reachable."""

    @abstractmethod
    def raw_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Run chat with provider-specific message format."""

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> Iterable[str]:
        """Generate a streaming text response from a prompt."""

