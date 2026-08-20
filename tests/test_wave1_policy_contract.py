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


def test_competition_local_vlm_defaults_to_four_bit() -> None:
    assert RuntimePolicy().local_vlm_load_in_4bit is True


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
