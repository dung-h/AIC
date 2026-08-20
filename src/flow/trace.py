"""JSON-safe, low-overhead trace for production flow decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

from src.flow.decision import FlowDecision
from src.runtime_context import RuntimeContext


@dataclass
class FlowTrace:
    task: str
    context: RuntimeContext
    owner: str
    events: list[dict[str, Any]] = field(default_factory=list)
    started_perf: float = field(default_factory=perf_counter)
    finished_ms: float | None = None

    def event(self, name: str, **payload: Any) -> None:
        self.events.append({"name": str(name), **payload})

    def decision(self, decision: FlowDecision) -> None:
        self.event("decision", **decision.to_dict())

    def finish(self) -> None:
        if self.finished_ms is None:
            self.finished_ms = round((perf_counter() - self.started_perf) * 1000.0, 3)

    def to_dict(self) -> dict[str, Any]:
        if self.finished_ms is None:
            self.finish()
        return {
            "task": self.task,
            "owner": self.owner,
            "request_id": self.context.request_id,
            "mode": self.context.mode,
            "split": self.context.split,
            "elapsed_ms": self.finished_ms,
            "events": list(self.events),
            "context": self.context.to_dict(),
        }


__all__ = ["FlowTrace"]
