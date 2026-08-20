from src.core.providers import provider_for


def test_text_provider_is_role_separated():
    provider = provider_for("text", {
        "TEXT_BASE_URL": "https://text.example/v1/",
        "TEXT_API_KEY": "test-key",
        "TEXT_MODEL": "text-model",
    })
    assert provider.configured
    assert provider.base_url == "https://text.example/v1"
    assert provider.model == "text-model"


def test_vision_keeps_legacy_digitalocean_fallback():
    provider = provider_for("vision", {
        "DO_INFERENCE_BASE": "https://vision.example/v1",
        "DO_INFERENCE_KEY": "test-key",
    })
    assert provider.configured
    assert provider.model == "gemma-4-31B-it"


def test_embedding_requires_its_own_capability():
    provider = provider_for("embedding", {"TEXT_BASE_URL": "https://text.example/v1"})
    assert not provider.configured
