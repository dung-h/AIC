"""Central production policy defaults for HCMAI runtimes.

Environment variables are read only here.  Callers may pass an explicit
``RuntimePolicy`` or a request/CLI override, but those overrides are local to
that pipeline invocation and never mutate this policy.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import activate_runtime_env


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name} must be one of {sorted(_TRUE | _FALSE)}; got {raw!r}")


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {{{allowed}}}; got {value!r}")
    return value


@dataclass(frozen=True)
class RuntimePolicy:
    """Immutable defaults shared by all production entrypoints."""

    kis_remote_translation: bool = False
    kis_nllb_routing: bool = False
    kis_nllb_threshold: float = 0.05
    trake_mode: str = "visual"
    trake_remote_embeddings: bool = False
    vkis_selector: str = "hybrid0.5"
    vqa_modality_routing: bool = False
    vqa_modality_model_dir: str = ""
    # Adaptive remains the measured production default until the newer
    # anchor-preserving allocator clears its dev/holdout promotion gate.
    # ``anchor_preserving`` is accepted here solely as an explicit A/B policy;
    # this policy object must not silently promote an unmeasured selector.
    vqa_visual_selector_policy: str = "adaptive"
    vqa_answer_provider: str = "local"
    local_vlm_path: str | None = None
    preload_trake: bool = False
    execution_mode: str = "production"
    research_routes_enabled: bool = False
    vqa_fallback_policy: str = "fail_closed"
    # Explicit network boundary.  ``None`` keeps the historical constructor
    # behavior while allowing benchmark_strict to derive an offline default.
    # The field is appended to preserve existing positional-call semantics.
    network_mode: str | None = None
    # The measured competition default is Qwen2.5-VL-7B in 4-bit.  Keeping
    # quantization in the immutable policy prevents the service entrypoint
    # from accidentally loading the same checkpoint in full precision.
    local_vlm_load_in_4bit: bool = True

    def __post_init__(self) -> None:
        # Constructor/CLI overrides are normalized at the policy boundary too,
        # not only by ``override()`` or ``from_env()``.  This keeps all callers
        # under the same validation/ownership rule while preserving immutability.
        for field_name in (
            "trake_mode", "vkis_selector", "vqa_visual_selector_policy",
            "vqa_answer_provider", "execution_mode", "vqa_fallback_policy",
            "network_mode",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, str(value).strip().lower())
        if not math.isfinite(float(self.kis_nllb_threshold)) or self.kis_nllb_threshold < 0:
            raise ValueError("kis_nllb_threshold must be a finite non-negative number")
        if not self.vqa_modality_model_dir:
            object.__setattr__(
                self,
                "vqa_modality_model_dir",
                str(Path(__file__).resolve().parents[1] / "models" / "bge-m3"),
            )
        if self.trake_mode not in {"visual", "asr"}:
            raise ValueError(f"unsupported TRAKE mode: {self.trake_mode!r}")
        if self.vkis_selector not in {
            "max", "mean", "top3", "smooth3", "smooth5", "hybrid0.5", "hybrid0.7"
        }:
            raise ValueError(f"unsupported VKIS selector: {self.vkis_selector!r}")
        if self.vqa_visual_selector_policy not in {
            "legacy", "balanced", "adaptive", "anchor_preserving"
        }:
            raise ValueError(
                f"unsupported VQA visual selector policy: {self.vqa_visual_selector_policy!r}"
            )
        if self.vqa_answer_provider not in {"local", "openai"}:
            raise ValueError(
                f"unsupported VQA answer provider: {self.vqa_answer_provider!r}"
            )
        if self.execution_mode not in {
            "production", "benchmark_strict", "interactive_safe", "research"
        }:
            raise ValueError(f"unsupported execution mode: {self.execution_mode!r}")
        if self.vqa_fallback_policy not in {"fail_closed", "visual_with_trace"}:
            raise ValueError(
                f"unsupported VQA fallback policy: {self.vqa_fallback_policy!r}"
            )
        if self.execution_mode in {"production", "benchmark_strict"} and self.vqa_fallback_policy != "fail_closed":
            raise ValueError(
                "production and benchmark_strict require fail_closed VQA fallback"
            )
        if self.network_mode is None:
            object.__setattr__(
                self,
                "network_mode",
                "offline" if self.execution_mode == "benchmark_strict" else "online",
            )
        if self.network_mode not in {"online", "offline"}:
            raise ValueError(f"unsupported network mode: {self.network_mode!r}")
        if self.execution_mode == "benchmark_strict" and self.network_mode != "offline":
            raise ValueError("benchmark_strict requires network_mode=offline")

        # ``benchmark_strict`` is an offline execution contract, regardless
        # of an omitted/legacy network setting.  An explicit offline mode is
        # also strict in production or interactive callers.  Rejecting an
        # unsafe policy at construction is fail-closed and prevents any
        # downstream provider/index factory from selecting a remote path.
        strict_network = (
            self.execution_mode == "benchmark_strict"
            or self.network_mode == "offline"
        )
        if strict_network:
            violations: list[str] = []
            if self.vqa_answer_provider == "openai":
                violations.append("vqa_answer_provider=openai")
            if self.kis_remote_translation:
                violations.append("kis_remote_translation=true")
            if self.trake_remote_embeddings:
                violations.append("trake_remote_embeddings=true")
            if violations:
                raise ValueError(
                    "offline/benchmark_strict policy forbids remote features: "
                    + ", ".join(violations)
                )

    def override(self, **values) -> "RuntimePolicy":
        """Return a validated per-run policy without mutating this default."""
        unknown = set(values) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"unknown RuntimePolicy override(s): {sorted(unknown)}")
        if "trake_mode" in values and values["trake_mode"] is not None:
            values["trake_mode"] = str(values["trake_mode"]).strip().lower()
        if "vkis_selector" in values and values["vkis_selector"] is not None:
            values["vkis_selector"] = str(values["vkis_selector"]).strip().lower()
        if "vqa_visual_selector_policy" in values and values["vqa_visual_selector_policy"] is not None:
            values["vqa_visual_selector_policy"] = str(values["vqa_visual_selector_policy"]).strip().lower()
        if "vqa_answer_provider" in values and values["vqa_answer_provider"] is not None:
            values["vqa_answer_provider"] = str(values["vqa_answer_provider"]).strip().lower()
        if "execution_mode" in values and values["execution_mode"] is not None:
            values["execution_mode"] = str(values["execution_mode"]).strip().lower()
            if (
                values["execution_mode"] == "benchmark_strict"
                and "network_mode" not in values
            ):
                values["network_mode"] = "offline"
        if "vqa_fallback_policy" in values and values["vqa_fallback_policy"] is not None:
            values["vqa_fallback_policy"] = str(values["vqa_fallback_policy"]).strip().lower()
        if "network_mode" in values and values["network_mode"] is not None:
            values["network_mode"] = str(values["network_mode"]).strip().lower()
        return type(self)(**{**self.__dict__, **values})

    @classmethod
    def from_env(cls) -> "RuntimePolicy":
        """Build the process default exactly once at an entrypoint boundary."""
        # Make `.env` available to every policy-driven entrypoint while keeping
        # an explicitly exported value authoritative over the local file.
        activate_runtime_env()
        project_root = Path(__file__).resolve().parents[1]
        execution_mode = _env_choice(
            "HCMAI_EXECUTION_MODE",
            "production",
            {"production", "benchmark_strict", "interactive_safe", "research"},
        )
        return cls(
            kis_remote_translation=_env_bool("HCMAI_KIS_REMOTE_TRANSLATION", False),
            kis_nllb_routing=_env_bool("HCMAI_KIS_NLLB_ROUTING", False),
            kis_nllb_threshold=float(os.getenv("HCMAI_KIS_NLLB_THRESHOLD", "0.05")),
            trake_mode=_env_choice("HCMAI_TRAKE_MODE", "visual", {"visual", "asr"}),
            trake_remote_embeddings=_env_bool("HCMAI_TRAKE_REMOTE_EMBEDDINGS", False),
            vkis_selector=_env_choice(
                "HCMAI_VKIS_SELECTOR",
                "hybrid0.5",
                {"max", "mean", "top3", "smooth3", "smooth5", "hybrid0.5", "hybrid0.7"},
            ),
            vqa_modality_routing=_env_bool("VQA_MODALITY_ROUTING", False),
            vqa_modality_model_dir=os.getenv(
                "VQA_MODALITY_MODEL_DIR",
                str(project_root / "models" / "bge-m3"),
            ),
            vqa_visual_selector_policy=_env_choice(
                "VQA_VISUAL_SELECTOR_POLICY", "adaptive",
                {"legacy", "balanced", "adaptive", "anchor_preserving"},
            ),
            vqa_answer_provider=_env_choice(
                "VQA_ANSWER_PROVIDER", "local", {"local", "openai"}
            ),
            local_vlm_path=os.getenv("HCMAI_LOCAL_VLM_PATH") or None,
            preload_trake=_env_bool("HCMAI_PRELOAD_TRAKE", False),
            execution_mode=execution_mode,
            research_routes_enabled=_env_bool("HCMAI_ENABLE_RESEARCH_ROUTES", False),
            vqa_fallback_policy=_env_choice(
                "HCMAI_VQA_FALLBACK_POLICY",
                "fail_closed",
                {"fail_closed", "visual_with_trace"},
            ),
            network_mode=_env_choice(
                "HCMAI_NETWORK_MODE",
                "offline" if execution_mode == "benchmark_strict" else "online",
                {"online", "offline"},
            ),
            local_vlm_load_in_4bit=_env_bool("HCMAI_LOCAL_VLM_4BIT", True),
        )
