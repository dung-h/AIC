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
from urllib.parse import urlparse

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


def _env_csv(name: str) -> tuple[str, ...]:
    """Read a comma-separated, non-secret allow-list from the runtime env."""
    return tuple(
        value.strip().casefold().lstrip(".")
        for value in os.getenv(name, "").split(",")
        if value.strip()
    )


@dataclass(frozen=True)
class RuntimePolicy:
    """Immutable defaults shared by all production entrypoints."""

    kis_remote_translation: bool = False
    kis_nllb_routing: bool = False
    kis_nllb_threshold: float = 0.05
    trake_mode: str = "visual"
    trake_remote_embeddings: bool = False
    # Alignment policy is owned at the same immutable boundary as the TRAKE
    # mode.  It must not be read ad-hoc by a lazy pipeline after a request has
    # begun, otherwise the trace cannot reproduce the run.
    trake_visual_alignment_policy: str = "legacy"
    trake_visual_candidate_video_limit: int | None = None
    trake_multimodal_alignment_policy: str = "legacy"
    vkis_selector: str = "hybrid0.5"
    vqa_modality_routing: bool = False
    vqa_modality_model_dir: str = ""
    # A staged ASR global artifact can be selected explicitly without
    # overwriting the previous production index.  The default registry path
    # remains unchanged until a measured promotion sets this value.
    vqa_asr_global_dir: str | None = None
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
    # External knowledge may only add retrieval hypotheses; it never supplies
    # a submitted answer or canonical frame.  These fields are appended to
    # preserve historical positional-call compatibility.
    vqa_external_grounding: bool = False
    vqa_external_search_backend: str = "searxng"
    vqa_external_search_url: str = ""
    vqa_external_allowed_domains: tuple[str, ...] = ()
    vqa_external_timeout_seconds: float = 5.0
    vqa_external_image_grounding: bool = False
    vqa_external_image_allowed_domains: tuple[str, ...] = ()
    vqa_external_image_allow_any_host: bool = False
    vqa_external_image_max_references: int = 3
    # Online, explicitly opt-in reasoning capabilities.  Hypotheses are
    # retrieval-only and semantic verification can only accept/reject an
    # already canonical candidate.  Neither changes the default baseline.
    vqa_hypothesis_generation: bool = False
    vqa_semantic_evidence_verifier: bool = False

    def __post_init__(self) -> None:
        # Constructor/CLI overrides are normalized at the policy boundary too,
        # not only by ``override()`` or ``from_env()``.  This keeps all callers
        # under the same validation/ownership rule while preserving immutability.
        for field_name in (
            "trake_mode", "trake_visual_alignment_policy",
            "trake_multimodal_alignment_policy", "vkis_selector", "vqa_visual_selector_policy",
            "vqa_answer_provider", "execution_mode", "vqa_fallback_policy",
            "network_mode", "vqa_external_search_backend",
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
        if self.trake_visual_alignment_policy not in {
            "legacy", "lattice_v1", "multi_video_v1"
        }:
            raise ValueError(
                "unsupported TRAKE visual alignment policy: "
                f"{self.trake_visual_alignment_policy!r}"
            )
        if self.trake_multimodal_alignment_policy not in {"legacy", "multi_video_v1"}:
            raise ValueError(
                "unsupported TRAKE multimodal alignment policy: "
                f"{self.trake_multimodal_alignment_policy!r}"
            )
        if self.trake_visual_candidate_video_limit is not None:
            candidate_limit = int(self.trake_visual_candidate_video_limit)
            if candidate_limit < 1:
                raise ValueError("trake_visual_candidate_video_limit must be positive")
            object.__setattr__(self, "trake_visual_candidate_video_limit", candidate_limit)
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
        if self.vqa_external_search_backend not in {"searxng", "ddg"}:
            raise ValueError(
                "vqa_external_search_backend must be 'searxng' or 'ddg'"
            )
        if self.execution_mode == "benchmark_strict" and self.network_mode != "offline":
            raise ValueError("benchmark_strict requires network_mode=offline")

        object.__setattr__(
            self,
            "vqa_external_search_url",
            str(self.vqa_external_search_url or "").strip().rstrip("/"),
        )
        raw_domains = self.vqa_external_allowed_domains
        if isinstance(raw_domains, str):
            raw_domains = raw_domains.split(",")
        object.__setattr__(
            self,
            "vqa_external_allowed_domains",
            tuple(dict.fromkeys(
                str(domain).strip().casefold().lstrip(".")
                for domain in raw_domains
                if str(domain).strip()
            )),
        )
        timeout = float(self.vqa_external_timeout_seconds)
        if not math.isfinite(timeout) or not 0.1 <= timeout <= 30.0:
            raise ValueError("vqa_external_timeout_seconds must be in [0.1, 30]")
        object.__setattr__(self, "vqa_external_timeout_seconds", timeout)
        raw_image_domains = self.vqa_external_image_allowed_domains
        if isinstance(raw_image_domains, str):
            raw_image_domains = raw_image_domains.split(",")
        object.__setattr__(
            self,
            "vqa_external_image_allowed_domains",
            tuple(dict.fromkeys(
                str(domain).strip().casefold().lstrip(".")
                for domain in raw_image_domains
                if str(domain).strip()
            )),
        )
        max_references = int(self.vqa_external_image_max_references)
        if not 1 <= max_references <= 8:
            raise ValueError("vqa_external_image_max_references must be in [1, 8]")
        object.__setattr__(self, "vqa_external_image_max_references", max_references)

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
        if self.vqa_external_grounding or self.vqa_external_image_grounding:
            if strict_network:
                raise ValueError(
                    "offline/benchmark_strict policy forbids external VQA grounding"
                )
            if self.vqa_external_search_backend == "searxng":
                parsed = urlparse(self.vqa_external_search_url)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    raise ValueError(
                        "SearXNG external grounding requires an absolute "
                        "VQA_EXTERNAL_SEARCH_URL"
                    )
        if self.vqa_external_grounding:
            if not self.vqa_external_allowed_domains:
                raise ValueError(
                    "vqa_external_grounding=true requires "
                    "VQA_EXTERNAL_ALLOWED_DOMAINS"
                )
        if self.vqa_external_image_grounding:
            if (
                not self.vqa_external_image_allowed_domains
                and not self.vqa_external_image_allow_any_host
            ):
                raise ValueError(
                    "vqa_external_image_grounding=true requires "
                    "VQA_EXTERNAL_IMAGE_ALLOWED_DOMAINS or "
                    "VQA_EXTERNAL_IMAGE_ALLOW_ANY_HOST=true"
                )
        if self.vqa_hypothesis_generation and not self.vqa_external_grounding:
            raise ValueError(
                "vqa_hypothesis_generation=true requires vqa_external_grounding=true; "
                "hypotheses cannot directly change local ranking"
            )
        if strict_network and (
            self.vqa_hypothesis_generation or self.vqa_semantic_evidence_verifier
        ):
            raise ValueError(
                "offline/benchmark_strict policy forbids remote VQA hypothesis/evidence verification"
            )

    def override(self, **values) -> "RuntimePolicy":
        """Return a validated per-run policy without mutating this default."""
        unknown = set(values) - set(self.__dataclass_fields__)
        if unknown:
            raise TypeError(f"unknown RuntimePolicy override(s): {sorted(unknown)}")
        if "trake_mode" in values and values["trake_mode"] is not None:
            values["trake_mode"] = str(values["trake_mode"]).strip().lower()
        for field_name in (
            "trake_visual_alignment_policy", "trake_multimodal_alignment_policy",
        ):
            if field_name in values and values[field_name] is not None:
                values[field_name] = str(values[field_name]).strip().lower()
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
            trake_visual_alignment_policy=_env_choice(
                "HCMAI_TRAKE_VISUAL_ALIGNMENT_POLICY",
                "legacy",
                {"legacy", "lattice_v1", "multi_video_v1"},
            ),
            trake_visual_candidate_video_limit=(
                int(os.environ["HCMAI_TRAKE_VISUAL_CANDIDATE_VIDEO_LIMIT"])
                if os.getenv("HCMAI_TRAKE_VISUAL_CANDIDATE_VIDEO_LIMIT")
                else None
            ),
            trake_multimodal_alignment_policy=_env_choice(
                "HCMAI_TRAKE_MULTIMODAL_ALIGNMENT_POLICY",
                "legacy",
                {"legacy", "multi_video_v1"},
            ),
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
            vqa_asr_global_dir=os.getenv("VQA_ASR_GLOBAL_DIR") or None,
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
            vqa_external_grounding=_env_bool("VQA_EXTERNAL_GROUNDING", False),
            vqa_external_search_backend=_env_choice(
                "VQA_EXTERNAL_SEARCH_BACKEND", "searxng", {"searxng", "ddg"}
            ),
            vqa_external_search_url=os.getenv("VQA_EXTERNAL_SEARCH_URL", ""),
            vqa_external_allowed_domains=_env_csv("VQA_EXTERNAL_ALLOWED_DOMAINS"),
            vqa_external_timeout_seconds=float(
                os.getenv("VQA_EXTERNAL_TIMEOUT_SECONDS", "5")
            ),
            vqa_external_image_grounding=_env_bool("VQA_EXTERNAL_IMAGE_GROUNDING", False),
            vqa_external_image_allowed_domains=_env_csv(
                "VQA_EXTERNAL_IMAGE_ALLOWED_DOMAINS"
            ),
            vqa_external_image_allow_any_host=_env_bool(
                "VQA_EXTERNAL_IMAGE_ALLOW_ANY_HOST", False
            ),
            vqa_external_image_max_references=int(
                os.getenv("VQA_EXTERNAL_IMAGE_MAX_REFERENCES", "3")
            ),
            vqa_hypothesis_generation=_env_bool("VQA_HYPOTHESIS_GENERATION", False),
            vqa_semantic_evidence_verifier=_env_bool(
                "VQA_SEMANTIC_EVIDENCE_VERIFIER", False
            ),
        )
