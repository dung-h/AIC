"""Explicit fallback state machine for task flows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from src.runtime_context import RuntimeContext


FlowState = Literal[
    "baseline_success",
    "specialist_success",
    "specialist_no_hit",
    "baseline_degraded",
    "failed",
]


def _normalize(values: Iterable[str] | None) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip().lower() for value in (values or ()) if str(value).strip()}))


@dataclass(frozen=True)
class FlowDecision:
    state: FlowState
    owner: str
    requested_modalities: tuple[str, ...] = ()
    active_modalities: tuple[str, ...] = ()
    fallback_reason: str | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.state in {"baseline_success", "specialist_success"}

    @property
    def used_fallback(self) -> bool:
        return self.state == "baseline_degraded"

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "owner": self.owner,
            "requested_modalities": list(self.requested_modalities),
            "active_modalities": list(self.active_modalities),
            "fallback_reason": self.fallback_reason,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


def decide_specialist_flow(
    context: RuntimeContext,
    *,
    owner: str,
    required_modalities: Iterable[str] | None,
    available_modalities: Iterable[str] | None,
    specialist_hit: bool,
    baseline_available: bool = True,
) -> FlowDecision:
    """Decide one flow transition; never silently changes a strict run."""
    requested = _normalize(required_modalities)
    available = _normalize(available_modalities)
    missing = tuple(modality for modality in requested if modality not in available)
    if missing:
        reason = "missing_required_modality:" + ",".join(missing)
        if context.allows_degraded_interactive and baseline_available:
            return FlowDecision(
                state="baseline_degraded",
                owner=owner,
                requested_modalities=requested,
                active_modalities=available,
                fallback_reason=reason,
            )
        return FlowDecision(
            state="failed",
            owner=owner,
            requested_modalities=requested,
            active_modalities=available,
            error=reason,
        )
    if specialist_hit:
        return FlowDecision(
            state="specialist_success",
            owner=owner,
            requested_modalities=requested,
            active_modalities=available,
        )
    if requested:
        if context.strict:
            return FlowDecision(
                state="failed",
                owner=owner,
                requested_modalities=requested,
                active_modalities=available,
                error="specialist_returned_no_hit",
            )
        if context.allows_degraded_interactive and baseline_available:
            return FlowDecision(
                state="baseline_degraded",
                owner=owner,
                requested_modalities=requested,
                active_modalities=available,
                fallback_reason="specialist_returned_no_hit",
            )
        return FlowDecision(
            state="specialist_no_hit",
            owner=owner,
            requested_modalities=requested,
            active_modalities=available,
            fallback_reason="specialist_returned_no_hit",
        )
    if baseline_available:
        return FlowDecision(
            state="baseline_success",
            owner=owner,
            requested_modalities=requested,
            active_modalities=available,
        )
    return FlowDecision(
        state="failed",
        owner=owner,
        requested_modalities=requested,
        active_modalities=available,
        error="baseline_unavailable",
    )


__all__ = ["FlowDecision", "FlowState", "decide_specialist_flow"]
