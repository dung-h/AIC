"""Wave 1 policy contract tests.

These tests exercise only the immutable policy boundary.  They do not load a
pipeline, model, index, or network provider.
"""

from __future__ import annotations

import pytest

from src.runtime_policy import RuntimePolicy


def test_legacy_default_policy_remains_local_and_online() -> None:
    policy = RuntimePolicy()

    assert policy.execution_mode == "production"
    assert policy.network_mode == "online"
    assert policy.vqa_answer_provider == "local"
    assert policy.kis_remote_translation is False
    assert policy.trake_remote_embeddings is False
    assert policy.trake_visual_alignment_policy == "legacy"
    assert policy.trake_visual_candidate_video_limit is None


def test_trake_alignment_knobs_are_validated_at_runtime_policy_boundary(monkeypatch) -> None:
    monkeypatch.setenv("HCMAI_TRAKE_VISUAL_ALIGNMENT_POLICY", "multi_video_v1")
    monkeypatch.setenv("HCMAI_TRAKE_VISUAL_CANDIDATE_VIDEO_LIMIT", "20")
    monkeypatch.setenv("HCMAI_TRAKE_MULTIMODAL_ALIGNMENT_POLICY", "multi_video_v1")

    policy = RuntimePolicy.from_env()

    assert policy.trake_visual_alignment_policy == "multi_video_v1"
    assert policy.trake_visual_candidate_video_limit == 20
    assert policy.trake_multimodal_alignment_policy == "multi_video_v1"

    with pytest.raises(ValueError, match="TRAKE visual alignment"):
        RuntimePolicy(trake_visual_alignment_policy="unmeasured")


@pytest.mark.parametrize(
    "overrides, forbidden",
    [
        ({"vqa_answer_provider": "openai"}, "vqa_answer_provider=openai"),
        ({"kis_remote_translation": True}, "kis_remote_translation=true"),
        ({"trake_remote_embeddings": True}, "trake_remote_embeddings=true"),
        (
            {
                "vqa_answer_provider": "openai",
                "kis_remote_translation": True,
                "trake_remote_embeddings": True,
            },
            "vqa_answer_provider=openai",
        ),
    ],
)
def test_benchmark_strict_rejects_every_remote_feature(overrides, forbidden) -> None:
    with pytest.raises(ValueError, match="offline/benchmark_strict") as exc_info:
        RuntimePolicy(execution_mode="benchmark_strict", **overrides)

    assert forbidden in str(exc_info.value)


def test_benchmark_strict_derives_offline_network_mode() -> None:
    policy = RuntimePolicy(execution_mode="benchmark_strict")

    assert policy.network_mode == "offline"
    assert policy.vqa_answer_provider == "local"


def test_explicit_offline_mode_rejects_remote_features_in_any_execution_mode() -> None:
    with pytest.raises(ValueError, match="kis_remote_translation=true"):
        RuntimePolicy(
            execution_mode="production",
            network_mode="offline",
            kis_remote_translation=True,
        )

    with pytest.raises(ValueError, match="vqa_answer_provider=openai"):
        RuntimePolicy(
            execution_mode="interactive_safe",
            network_mode="offline",
            vqa_answer_provider="openai",
        )


def test_online_remote_provider_remains_available_for_explicit_non_strict_runs() -> None:
    policy = RuntimePolicy(
        execution_mode="interactive_safe",
        network_mode="online",
        vqa_answer_provider="openai",
        kis_remote_translation=True,
        trake_remote_embeddings=True,
    )

    assert policy.network_mode == "online"
    assert policy.vqa_answer_provider == "openai"
    assert policy.kis_remote_translation is True
    assert policy.trake_remote_embeddings is True


def test_online_production_may_explicitly_select_remote_answer_provider() -> None:
    policy = RuntimePolicy(
        execution_mode="production",
        network_mode="online",
        vqa_answer_provider="openai",
    )

    assert policy.vqa_answer_provider == "openai"
    assert policy.network_mode == "online"


def test_external_grounding_is_explicit_online_capability_with_allowlisted_sources() -> None:
    policy = RuntimePolicy(
        network_mode="online",
        vqa_external_grounding=True,
        vqa_external_search_url="https://search.example.org",
        vqa_external_allowed_domains=("example.org",),
    )

    assert policy.vqa_external_grounding is True
    assert policy.vqa_external_allowed_domains == ("example.org",)

    with pytest.raises(ValueError, match="VQA_EXTERNAL_SEARCH_URL"):
        RuntimePolicy(network_mode="online", vqa_external_grounding=True)

    with pytest.raises(ValueError, match="forbids external VQA grounding"):
        RuntimePolicy(
            execution_mode="benchmark_strict",
            vqa_external_grounding=True,
            vqa_external_search_url="https://search.example.org",
            vqa_external_allowed_domains=("example.org",),
        )


def test_hypothesis_and_semantic_verifier_are_explicit_online_capabilities() -> None:
    policy = RuntimePolicy(
        network_mode="online",
        vqa_external_grounding=True,
        vqa_external_search_url="https://search.example.org",
        vqa_external_allowed_domains=("example.org",),
        vqa_hypothesis_generation=True,
        vqa_semantic_evidence_verifier=True,
    )

    assert policy.vqa_hypothesis_generation is True
    assert policy.vqa_semantic_evidence_verifier is True

    with pytest.raises(ValueError, match="requires vqa_external_grounding"):
        RuntimePolicy(network_mode="online", vqa_hypothesis_generation=True)

    with pytest.raises(ValueError, match="hypothesis/evidence verification"):
        RuntimePolicy(
            execution_mode="benchmark_strict",
            vqa_semantic_evidence_verifier=True,
        )


def test_ddg_image_grounding_does_not_require_a_self_hosted_search_url() -> None:
    policy = RuntimePolicy(
        network_mode="online",
        vqa_external_search_backend="ddg",
        vqa_external_image_grounding=True,
        vqa_external_image_allow_any_host=True,
    )

    assert policy.vqa_external_search_url == ""
    assert policy.vqa_external_search_backend == "ddg"
    assert policy.vqa_external_image_grounding is True


def test_competition_local_vlm_defaults_to_four_bit() -> None:
    assert RuntimePolicy().local_vlm_load_in_4bit is True


def test_vqa_asr_global_dir_is_an_explicit_promotion_switch(monkeypatch) -> None:
    monkeypatch.setenv("VQA_ASR_GLOBAL_DIR", "/srv/aic/asr_global_v3")

    policy = RuntimePolicy.from_env()

    assert policy.vqa_asr_global_dir == "/srv/aic/asr_global_v3"


def test_override_revalidates_the_network_boundary_without_mutating_base() -> None:
    base = RuntimePolicy()
    strict = base.override(execution_mode="benchmark_strict")

    assert base.execution_mode == "production"
    assert base.network_mode == "online"
    assert strict.execution_mode == "benchmark_strict"
    assert strict.network_mode == "offline"

    with pytest.raises(ValueError, match="trake_remote_embeddings=true"):
        base.override(network_mode="offline", trake_remote_embeddings=True)


def test_from_env_defaults_strict_execution_to_offline(monkeypatch) -> None:
    monkeypatch.setenv("HCMAI_EXECUTION_MODE", "benchmark_strict")
    monkeypatch.delenv("HCMAI_NETWORK_MODE", raising=False)
    monkeypatch.setenv("VQA_ANSWER_PROVIDER", "local")
    monkeypatch.setenv("HCMAI_KIS_REMOTE_TRANSLATION", "false")
    monkeypatch.setenv("HCMAI_TRAKE_REMOTE_EMBEDDINGS", "false")

    policy = RuntimePolicy.from_env()

    assert policy.execution_mode == "benchmark_strict"
    assert policy.network_mode == "offline"


def test_from_env_rejects_strict_execution_with_explicit_online_mode(monkeypatch) -> None:
    monkeypatch.setenv("HCMAI_EXECUTION_MODE", "benchmark_strict")
    monkeypatch.setenv("HCMAI_NETWORK_MODE", "online")
    monkeypatch.setenv("VQA_ANSWER_PROVIDER", "local")
    monkeypatch.setenv("HCMAI_KIS_REMOTE_TRANSLATION", "false")
    monkeypatch.setenv("HCMAI_TRAKE_REMOTE_EMBEDDINGS", "false")

    with pytest.raises(ValueError, match="network mode|offline"):
        RuntimePolicy.from_env()
