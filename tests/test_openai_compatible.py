from types import SimpleNamespace

from src.core.openai_compatible import chat_text
from src.core.providers import ProviderConfig


def test_chat_text_uses_reasoning_content_when_content_empty(monkeypatch):
    class Client:
        def __init__(self, **_):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **__: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="", reasoning_content="0.5"))]
            )))
    monkeypatch.setattr("openai.OpenAI", Client)
    provider = ProviderConfig("text", "https://example/v1", "key", "model")
    assert chat_text(provider, "judge") == "0.5"
