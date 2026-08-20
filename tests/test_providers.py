from src.core.providers import provider_for
from src.utils.paths import load_env
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


def test_vision_legacy_alias_is_normalized_at_config_boundary(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DO_INFERENCE_BASE=https://vision.example/v1\n"
        "DO_INFERENCE_KEY=test-key\n"
        "DO_VLM_MODEL=vision-model\n",
        encoding="utf-8",
    )
    provider = provider_for("vision", load_env(env_path))
    assert provider.configured
    assert provider.model == "vision-model"


def test_embedding_requires_its_own_capability():
    provider = provider_for("embedding", {"TEXT_BASE_URL": "https://text.example/v1"})
    assert not provider.configured


def test_remote_helpers_fail_closed_without_their_role_provider(monkeypatch):
    from src.core import offline_fallback, query_rewriter, reranker
    from src.core.providers import ProviderConfig

    missing = ProviderConfig("missing", "", "", "")
    monkeypatch.setattr(query_rewriter, "provider_for", lambda _role: missing)
    monkeypatch.setattr(reranker, "provider_for", lambda _role: missing)
    monkeypatch.setattr(offline_fallback, "provider_for", lambda _role: missing)

    with pytest.raises(RuntimeError, match="TEXT_BASE_URL"):
        query_rewriter._llm([{"role": "user", "content": "rewrite"}])
    with pytest.raises(RuntimeError, match="VLM_BASE_URL"):
        reranker._vlm_visual_score("image", "scene")
    with pytest.raises(RuntimeError, match="EMBEDDING_BASE_URL"):
        offline_fallback.TextEmbedderOnline()


def test_query_expansion_has_no_implicit_remote_fallback():
    from src.pipelines.query_expansion import generate_variants

    assert generate_variants("một cảnh quay", env={}) == ["một cảnh quay"]
