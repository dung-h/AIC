from src.core.providers import provider_for
import pytest


def test_text_provider_is_role_separated():
    provider = provider_for("text", {
        "TEXT_BASE_URL": "https://text.example/v1/",
        "TEXT_API_KEY": "test-key",
        "TEXT_MODEL": "text-model",
    })
    assert provider.configured
    assert provider.base_url == "https://text.example/v1"
    assert provider.model == "text-model"


def test_vision_requires_standardized_role_keys():
    provider = provider_for("vision", {
        "VLM_BASE_URL": "https://vision.example/v1",
        "VLM_API_KEY": "test-key",
        "VLM_MODEL": "vision-model",
    })
    assert provider.configured
    assert provider.model == "vision-model"


def test_unsupported_provider_names_do_not_configure_vision():
    provider = provider_for("vision", {
        "UNSUPPORTED_VISION_BASE": "https://vision.example/v1",
        "UNSUPPORTED_VISION_KEY": "test-key",
        "UNSUPPORTED_VISION_MODEL": "vision-model",
    })
    assert not provider.configured


def test_embedding_requires_its_own_capability():
    provider = provider_for("embedding", {"TEXT_BASE_URL": "https://text.example/v1"})
    assert not provider.configured


def test_remote_embedding_helper_fails_closed_without_its_role_provider(monkeypatch):
    from src.core import offline_fallback
    from src.core.providers import ProviderConfig

    missing = ProviderConfig("missing", "", "", "")
    monkeypatch.setattr(offline_fallback, "provider_for", lambda _role: missing)

    with pytest.raises(RuntimeError, match="EMBEDDING_BASE_URL"):
        offline_fallback.TextEmbedderOnline()
