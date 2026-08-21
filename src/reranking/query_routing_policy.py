"""Explicit query-type routing plans for video retrieval.

The integration boundary is intentionally small and pipeline-independent::

    plan = build_routing_plan(question_type, config)
    rows = route_video_candidates(channel_candidates, plan)

``channel_candidates`` maps ``visual``, ``asr``, ``ocr`` and optional
precision event lanes (currently ``poetry``) to ranked
iterables.  Weights and rescue thresholds are configuration/dev-sweep inputs;
this module never reads holdout metrics or silently tunes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .video_rrf import weighted_video_rrf


CHANNELS = ("visual", "asr", "ocr", "poetry")
QUESTION_TYPES = ("visual", "spoken_fact", "screen_text", "temporal_relation", "unknown")
VISUAL_QUESTION_TYPES = frozenset(
    {"visual", "color", "action", "count", "person", "place"}
)


class RoutingPolicyError(ValueError):
    """Raised when a routing configuration or request is invalid."""


def canonical_question_type(question_type: str | None) -> str:
    """Map annotation labels to one of the five explicit routing plans."""

    normalized = str(question_type or "unknown").strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    if normalized in VISUAL_QUESTION_TYPES:
        return "visual"
    if normalized in {"spoken", "speech", "audio", "asr", "spoken_fact"}:
        return "spoken_fact"
    if normalized in {"ocr", "text", "screen", "screen_text"}:
        return "screen_text"
    if normalized in {"temporal", "sequence", "trake", "temporal_relation"}:
        return "temporal_relation"
    return "unknown"


def _finite_number(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RoutingPolicyError(f"{field_name} must be numeric") from exc
    if not isfinite(result):
        raise RoutingPolicyError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class RescueGate:
    """Rank/evidence gate for specialist-only video rescue.

    ``strong_rank`` remains the conservative default rescue path. The
    evidence-aware path is intentionally narrower: it can admit a lower-ranked
    specialist hit only when the candidate contains actual local text *and*
    carries a strong lexical or quoted-fact provenance record. This keeps a
    generic OCR/ASR utterance from becoming a video-level vote merely because
    it happens to have a non-empty ``text`` field.
    """

    enabled: bool = True
    strong_rank: int = 5
    support_rank: int = 20
    min_specialist_channels: int = 1
    allow_single_strong_rescue: bool = True
    require_evidence: bool = True
    min_scores: Mapping[str, float] = field(default_factory=dict)
    evidence_keys: tuple[str, ...] = (
        "evidence", "text", "asr_text", "ocr_text", "transcript",
    )
    evidence_aware_rescue_enabled: bool = True
    evidence_aware_max_rank: int = 20
    provenance_max_rank: int = 3
    provenance_min_score: float = 0.8
    provenance_modes: tuple[str, ...] = (
        "bm25_coverage", "lexical_exact", "exact", "exact_match",
        "quote", "quoted_fact",
    )

    def __post_init__(self) -> None:
        if int(self.strong_rank) < 1 or int(self.support_rank) < 1:
            raise RoutingPolicyError("rescue ranks must be positive")
        if int(self.evidence_aware_max_rank) < 1:
            raise RoutingPolicyError("evidence_aware_max_rank must be positive")
        if int(self.provenance_max_rank) < 1:
            raise RoutingPolicyError("provenance_max_rank must be positive")
        if int(self.min_specialist_channels) < 1:
            raise RoutingPolicyError("min_specialist_channels must be >= 1")
        keys = tuple(str(key).strip() for key in self.evidence_keys if str(key).strip())
        if self.require_evidence and not keys:
            raise RoutingPolicyError("evidence_keys cannot be empty when evidence is required")
        object.__setattr__(self, "strong_rank", int(self.strong_rank))
        object.__setattr__(self, "support_rank", int(self.support_rank))
        object.__setattr__(self, "min_specialist_channels", int(self.min_specialist_channels))
        object.__setattr__(self, "evidence_aware_max_rank", int(self.evidence_aware_max_rank))
        object.__setattr__(self, "provenance_max_rank", int(self.provenance_max_rank))
        object.__setattr__(
            self, "provenance_min_score",
            _finite_number(self.provenance_min_score, "provenance_min_score"),
        )
        object.__setattr__(self, "evidence_keys", keys)
        provenance_modes = tuple(
            dict.fromkeys(
                str(mode).strip().lower()
                for mode in self.provenance_modes
                if str(mode).strip()
            )
        )
        if self.evidence_aware_rescue_enabled and not provenance_modes:
            raise RoutingPolicyError(
                "provenance_modes cannot be empty when evidence-aware rescue is enabled"
            )
        object.__setattr__(self, "provenance_modes", provenance_modes)
        object.__setattr__(
            self,
            "min_scores",
            MappingProxyType({str(channel): _finite_number(score, f"min_scores[{channel}]")
                              for channel, score in self.min_scores.items()}),
        )

    @classmethod
    def disabled(cls) -> "RescueGate":
        return cls(
            enabled=False,
            require_evidence=False,
            evidence_keys=(),
            evidence_aware_rescue_enabled=False,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RescueGate":
        allowed = {
            "enabled", "strong_rank", "support_rank", "min_specialist_channels",
            "allow_single_strong_rescue", "require_evidence", "min_scores", "evidence_keys",
            "evidence_aware_rescue_enabled", "evidence_aware_max_rank",
            "provenance_max_rank", "provenance_min_score", "provenance_modes",
        }
        unknown = set(value) - allowed
        if unknown:
            raise RoutingPolicyError(f"unknown rescue gate fields: {sorted(unknown)}")
        return cls(**dict(value))


def _freeze_weights(
    mapping: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    frozen = {}
    for question_type, weights in mapping.items():
        normalized = {
            str(channel): _finite_number(weight, f"weight[{question_type}][{channel}]")
            for channel, weight in weights.items()
        }
        if any(weight < 0 for weight in normalized.values()):
            raise RoutingPolicyError("routing weights must be non-negative")
        frozen[str(question_type)] = MappingProxyType(normalized)
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class RoutingConfig:
    """Versionable routing inputs suitable for a dev sweep."""

    enabled: bool
    weights_by_type: Mapping[str, Mapping[str, float]]
    rescue_by_type: Mapping[str, RescueGate]
    rrf_k: int = 60
    retrieval_top_k: int = 100
    output_top_k: int = 20

    def __post_init__(self) -> None:
        if int(self.rrf_k) < 0:
            raise RoutingPolicyError("rrf_k must be >= 0")
        if int(self.retrieval_top_k) < 1 or int(self.output_top_k) < 1:
            raise RoutingPolicyError("retrieval_top_k and output_top_k must be positive")
        weights = {
            canonical_question_type(question_type): dict(value)
            for question_type, value in self.weights_by_type.items()
        }
        rescue = {
            canonical_question_type(question_type): value
            for question_type, value in self.rescue_by_type.items()
        }
        if any(not isinstance(value, RescueGate) for value in rescue.values()):
            raise RoutingPolicyError("rescue_by_type values must be RescueGate instances")
        object.__setattr__(self, "weights_by_type", _freeze_weights(weights))
        object.__setattr__(self, "rescue_by_type", MappingProxyType(rescue))
        object.__setattr__(self, "rrf_k", int(self.rrf_k))
        object.__setattr__(self, "retrieval_top_k", int(self.retrieval_top_k))
        object.__setattr__(self, "output_top_k", int(self.output_top_k))

    @classmethod
    def baseline(cls, *, enabled: bool = False) -> "RoutingConfig":
        """Transparent starting config; values are dev-sweep inputs."""

        return cls(
            enabled=enabled,
            weights_by_type={
                "visual": {"visual": 1.0},
                # Both text lanes remain retrieval support. The primary
                # modality below is still the only required evidence for
                # answer acceptance; RRF never requires both specialists.
                # ``poetry`` is an optional ASR-event lane: it is only
                # materialized for an explicit verse request and every row is
                # locally joined to a same-video entity context. It is not an
                # answer modality and cannot exist for ordinary spoken facts.
                "spoken_fact": {"visual": 0.75, "asr": 1.0, "ocr": 0.5, "poetry": 1.0},
                "screen_text": {"visual": 0.75, "asr": 0.5, "ocr": 1.0},
                "temporal_relation": {"visual": 1.0, "asr": 0.5, "ocr": 0.5},
                "unknown": {"visual": 1.0, "asr": 0.5, "ocr": 0.5},
            },
            rescue_by_type={
                "visual": RescueGate.disabled(),
                "spoken_fact": RescueGate(strong_rank=5, support_rank=20),
                "screen_text": RescueGate(strong_rank=5, support_rank=20),
                "temporal_relation": RescueGate(strong_rank=5, support_rank=20,
                                                 min_specialist_channels=2,
                                                 allow_single_strong_rescue=False),
                "unknown": RescueGate(strong_rank=5, support_rank=20,
                                      min_specialist_channels=2,
                                      allow_single_strong_rescue=False),
            },
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoutingConfig":
        allowed = {
            "enabled", "weights_by_type", "rescue_by_type", "rrf_k",
            "retrieval_top_k", "output_top_k",
        }
        unknown = set(value) - allowed
        if unknown:
            raise RoutingPolicyError(f"unknown routing config fields: {sorted(unknown)}")
        rescue = {
            question_type: (gate if isinstance(gate, RescueGate)
                            else RescueGate.from_mapping(gate))
            for question_type, gate in value["rescue_by_type"].items()
        }
        return cls(
            enabled=bool(value["enabled"]),
            weights_by_type=value["weights_by_type"],
            rescue_by_type=rescue,
            rrf_k=value.get("rrf_k", 60),
            retrieval_top_k=value.get("retrieval_top_k", 100),
            output_top_k=value.get("output_top_k", 20),
        )


@dataclass(frozen=True)
class QueryRoutingPlan:
    """Resolved immutable plan passed to the RRF integration boundary."""

    question_type: str
    routing_enabled: bool
    channels: tuple[str, ...]
    primary_channel: str
    specialist_channels: tuple[str, ...]
    weights: Mapping[str, float]
    rescue_gate: RescueGate
    required_channels: tuple[str, ...]
    rrf_k: int
    retrieval_top_k: int
    output_top_k: int
    visual_channel: str = "visual"

    def __post_init__(self) -> None:
        channels = tuple(dict.fromkeys(str(channel) for channel in self.channels))
        specialists = tuple(dict.fromkeys(str(channel) for channel in self.specialist_channels))
        required = tuple(dict.fromkeys(str(channel) for channel in self.required_channels))
        if self.visual_channel not in channels:
            raise RoutingPolicyError("visual channel must be present in every plan")
        if self.primary_channel not in channels:
            raise RoutingPolicyError("primary channel must be present in the plan")
        if any(channel not in channels for channel in specialists + required):
            raise RoutingPolicyError("specialist/required channel is absent from plan")
        weights = {str(channel): _finite_number(weight, f"weights[{channel}]")
                   for channel, weight in self.weights.items()}
        if any(weight < 0 for weight in weights.values()):
            raise RoutingPolicyError("routing weights must be non-negative")
        if any(channel not in weights for channel in channels):
            raise RoutingPolicyError("every plan channel needs an explicit weight")
        if not any(weights[channel] > 0 for channel in channels):
            raise RoutingPolicyError("at least one plan channel must have positive weight")
        object.__setattr__(self, "question_type", canonical_question_type(self.question_type))
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "specialist_channels", specialists)
        object.__setattr__(self, "required_channels", required)
        object.__setattr__(self, "weights", MappingProxyType(weights))


_PRIMARY_CHANNEL = {
    "visual": "visual",
    "spoken_fact": "asr",
    "screen_text": "ocr",
    "temporal_relation": "visual",
    "unknown": "visual",
}


def build_routing_plan(question_type: str | None, config: RoutingConfig) -> QueryRoutingPlan:
    """Resolve one query-type plan from caller-supplied configuration."""

    canonical = canonical_question_type(question_type)
    if "visual" not in config.weights_by_type:
        raise RoutingPolicyError("config must define a visual plan")
    if not config.enabled:
        visual_weight = config.weights_by_type["visual"].get("visual", 0.0)
        return QueryRoutingPlan(
            question_type=canonical,
            routing_enabled=False,
            channels=("visual",),
            primary_channel="visual",
            specialist_channels=(),
            weights={"visual": visual_weight},
            rescue_gate=RescueGate.disabled(),
            required_channels=("visual",),
            rrf_k=config.rrf_k,
            retrieval_top_k=config.retrieval_top_k,
            output_top_k=config.output_top_k,
        )
    weights = config.weights_by_type.get(canonical)
    if weights is None:
        raise RoutingPolicyError(f"config has no weights for question type {canonical}")
    unknown_channels = set(weights) - set(CHANNELS)
    if unknown_channels:
        raise RoutingPolicyError(f"unsupported routing channels: {sorted(unknown_channels)}")
    channels = tuple(channel for channel in CHANNELS if float(weights.get(channel, 0.0)) > 0)
    primary = _PRIMARY_CHANNEL[canonical]
    if primary not in channels:
        raise RoutingPolicyError(f"primary channel {primary} has no positive weight")
    gate = config.rescue_by_type.get(canonical, RescueGate.disabled())
    specialists = tuple(channel for channel in channels if channel != "visual")
    if specialists and not gate.enabled:
        raise RoutingPolicyError(
            "specialist channels require an enabled evidence/rank rescue gate"
        )
    return QueryRoutingPlan(
        question_type=canonical,
        routing_enabled=True,
        channels=channels,
        primary_channel=primary,
        specialist_channels=specialists,
        weights=weights,
        rescue_gate=gate,
        # Visual retrieval is the invariant anchor. The primary text lane is
        # required for its own evidence contract; the other specialist lane is
        # parallel support only and must never become an answer precondition.
        required_channels=tuple(dict.fromkeys(("visual", primary))),
        rrf_k=config.rrf_k,
        retrieval_top_k=config.retrieval_top_k,
        output_top_k=config.output_top_k,
    )


def route_video_candidates(
    channel_candidates: Mapping[str, Iterable[dict | tuple]],
    plan: QueryRoutingPlan,
) -> list[dict]:
    """Apply a resolved plan to ranked channel candidates.

    Routing-off explicitly ignores ASR/OCR and calls the visual-only path.
    Routing-on requires the plan's primary channel and applies the configured
    evidence/rank rescue gate through :func:`weighted_video_rrf`.
    """

    supplied = {str(channel): candidates for channel, candidates in channel_candidates.items()}
    missing = [channel for channel in plan.required_channels if channel not in supplied]
    if missing:
        raise RoutingPolicyError(f"missing required channel candidates: {missing}")

    if not plan.routing_enabled:
        return weighted_video_rrf(
            {"visual": supplied[plan.visual_channel]},
            {"visual": float(plan.weights[plan.visual_channel])},
            rrf_k=plan.rrf_k,
            topk=plan.output_top_k,
            visual_channel=plan.visual_channel,
            specialist_rescue_enabled=False,
        )

    selected = {
        channel: supplied[channel]
        for channel in plan.channels
        if channel in supplied
    }
    gate = plan.rescue_gate
    return weighted_video_rrf(
        selected,
        dict(plan.weights),
        rrf_k=plan.rrf_k,
        topk=plan.output_top_k,
        visual_channel=plan.visual_channel,
        specialist_strong_rank=gate.strong_rank,
        specialist_support_rank=gate.support_rank,
        min_specialist_channels=gate.min_specialist_channels,
        allow_single_strong_rescue=gate.allow_single_strong_rescue,
        specialist_min_scores=gate.min_scores,
        require_specialist_evidence=gate.require_evidence and gate.enabled,
        evidence_keys=gate.evidence_keys,
        specialist_rescue_enabled=gate.enabled,
        evidence_aware_rescue_enabled=(
            gate.enabled and gate.evidence_aware_rescue_enabled
        ),
        evidence_aware_max_rank=gate.evidence_aware_max_rank,
        provenance_max_rank=gate.provenance_max_rank,
        provenance_min_score=gate.provenance_min_score,
        provenance_modes=gate.provenance_modes,
    )


__all__ = [
    "CHANNELS", "QUESTION_TYPES", "QueryRoutingPlan", "RescueGate",
    "RoutingConfig", "RoutingPolicyError", "build_routing_plan",
    "canonical_question_type", "route_video_candidates",
]
