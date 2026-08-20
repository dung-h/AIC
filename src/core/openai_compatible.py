"""Small normalized client for OpenAI-compatible text chat providers."""
from __future__ import annotations

from src.core.providers import ProviderConfig


def chat_text(provider: ProviderConfig, prompt: str, *, max_tokens: int = 64) -> str:
    """Return normal content, falling back to providers' reasoning channel."""
    if not provider.configured:
        raise RuntimeError(f"{provider.role} provider is not configured")
    from openai import OpenAI

    client = OpenAI(api_key=provider.api_key, base_url=provider.base_url)
    response = client.chat.completions.create(
        model=provider.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    message = response.choices[0].message
    return (message.content or getattr(message, "reasoning_content", None) or "").strip()
