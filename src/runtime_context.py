"""Immutable execution context shared by production task flows.

The context is deliberately small and provider agnostic.  It carries policy,
benchmark/interactive mode, provenance and an artifact snapshot; it does not
own model instances or mutate global process state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4

from src.runtime_policy import RuntimePolicy


EXECUTION_MODES = frozenset({"production", "benchmark_strict", "interactive_safe", "research"})


def _freeze_value(value: Any) -> Any:
    """Recursively freeze an artifact snapshot value.

    ``MappingProxyType(dict(snapshot))`` only protects the outer mapping.  A
    nested dict/list could otherwise be mutated after a context was created,
    changing the provenance seen by a later flow in the same request.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class RuntimeContext:
    """Per-request context; no flow may replace its policy implicitly."""

    policy: RuntimePolicy = field(default_factory=RuntimePolicy.from_env)
    mode: str | None = None
    request_id: str = ""
    split: str | None = None
    artifact_snapshot: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.policy, RuntimePolicy):
            raise TypeError("RuntimeContext.policy must be a RuntimePolicy")
        mode = self.policy.execution_mode if self.mode is None else str(self.mode).strip().lower()
        if mode not in EXECUTION_MODES:
            raise ValueError(f"unsupported execution mode: {mode!r}")
        object.__setattr__(self, "mode", mode)
        if not self.request_id:
            object.__setattr__(self, "request_id", uuid4().hex)
        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            )
        object.__setattr__(
            self,
            "artifact_snapshot",
            _freeze_value(self.artifact_snapshot or {}),
        )

    @classmethod
    def from_policy(
        cls,
        policy: RuntimePolicy | None = None,
        *,
        mode: str | None = None,
        split: str | None = None,
        request_id: str | None = None,
        artifact_snapshot: Mapping[str, Any] | None = None,
    ) -> "RuntimeContext":
        return cls(
            policy=RuntimePolicy.from_env() if policy is None else policy,
            mode=mode,
            split=split,
            request_id=request_id or "",
            artifact_snapshot=artifact_snapshot or {},
        )

    @property
    def strict(self) -> bool:
        return self.mode in {"production", "benchmark_strict"}

    @property
    def allows_degraded_interactive(self) -> bool:
        return (
            self.mode in {"interactive_safe", "research"}
            and self.policy.vqa_fallback_policy == "visual_with_trace"
        )

    def with_artifacts(self, snapshot: Mapping[str, Any]) -> "RuntimeContext":
        return RuntimeContext(
            policy=self.policy,
            mode=self.mode,
            request_id=self.request_id,
            split=self.split,
            artifact_snapshot=snapshot,
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "mode": self.mode,
            "split": self.split,
            "created_at": self.created_at,
            "policy": _json_value(self.policy.__dict__),
            "artifacts": _json_value(self.artifact_snapshot),
        }


__all__ = ["EXECUTION_MODES", "RuntimeContext"]
