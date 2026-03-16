from __future__ import annotations

from src.core.base_llm import BaseLLM
from src.core.ollama_client import OllamaClient
from src.settings import load_settings


def create_llm() -> BaseLLM:
    settings = load_settings()
    llm_cfg = settings.get("llm", {})
    provider = llm_cfg.get("provider", "ollama")

    if provider == "ollama":
        return OllamaClient(
            host=llm_cfg.get("host", "http://localhost:11434"),
            model=llm_cfg.get("model", "qwen3:8b"),
            timeout_seconds=int(llm_cfg.get("timeout_seconds", 120)),
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")

