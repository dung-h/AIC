"""Role-specific OpenAI-compatible provider configuration.

Keep text generation, vision chat, and embeddings independent: an endpoint
being OpenAI-compatible does not imply it supports image inputs or embeddings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.utils.paths import load_env


@dataclass(frozen=True)
class ProviderConfig:
    role: str
    base_url: str
    api_key: str
    model: str

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


_ROLE_KEYS = {
    "text": ("TEXT_BASE_URL", "TEXT_API_KEY", "TEXT_MODEL"),
    "vision": ("VLM_BASE_URL", "VLM_API_KEY", "VLM_MODEL"),
    "embedding": ("EMBEDDING_BASE_URL", "EMBEDDING_API_KEY", "EMBEDDING_MODEL"),
}


def provider_for(role: str, env: dict[str, str] | None = None) -> ProviderConfig:
    """Resolve one provider role without mixing endpoint capabilities.

    Legacy DigitalOcean variables remain a fallback only for the vision role,
    preserving current behavior until a dedicated VLM provider is configured.
    """
    if role not in _ROLE_KEYS:
        raise ValueError(f"unknown provider role: {role}")
    # Passing an env mapping is primarily for tests and explicit callers; do
    # not leak unrelated local .env provider choices into that configuration.
    values = {**({} if env is not None else load_env()), **os.environ, **(env or {})}
    base_key, api_key, model_key = _ROLE_KEYS[role]
    base = values.get(base_key, "")
    key = values.get(api_key, "")
    model = values.get(model_key, "")
    if role == "vision" and not (base and key and model):
        base = base or values.get("DO_INFERENCE_BASE", "")
        key = key or values.get("DO_INFERENCE_KEY", "")
        model = model or values.get("DO_VLM_MODEL", "gemma-4-31B-it")
    return ProviderConfig(role=role, base_url=base.rstrip("/"), api_key=key, model=model)
