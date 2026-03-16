from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

from src.core.base_llm import BaseLLM


class OllamaClient(BaseLLM):
    def __init__(self, host: str, model: str, timeout_seconds: int = 120) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def raw_chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "stream": False,
            "options": {"temperature": kwargs.get("temperature", 0.1)},
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.host}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("message", {}).get("content", "").strip()

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.raw_chat(messages=messages, temperature=temperature)

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
    ) -> Iterable[str]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }

        with httpx.stream(
            "POST", 
            f"{self.host}/api/chat", 
            json=payload, 
            timeout=self.timeout_seconds
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                import json
                chunk = json.loads(line)
                if content := chunk.get("message", {}).get("content"):
                    yield content
                if chunk.get("done"):
                    break

    def healthcheck(self) -> bool:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(f"{self.host}/api/tags")
                response.raise_for_status()
                tags = response.json().get("models", [])
            model_names = {tag.get("name") for tag in tags}
            if self.model not in model_names:
                logger.warning("Ollama reachable but model '{}' is not pulled yet.", self.model)
            return True
        except Exception as exc:
            logger.error("Ollama healthcheck failed: {}", exc)
            return False

