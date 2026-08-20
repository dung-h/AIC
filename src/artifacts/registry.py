"""Small, provider-neutral registry for model/index/catalog readiness."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


_SCOPE_RANK = {"missing": 0, "diagnostic": 1, "partial": 2, "global": 3}


class ArtifactUnavailable(RuntimeError):
    """Raised when a flow requires an artifact that is not ready."""


@dataclass(frozen=True)
class ArtifactStatus:
    name: str
    ready: bool
    scope: str = "missing"
    path: str | None = None
    row_count: int | None = None
    dimension: int | None = None
    model_id: str | None = None
    version: str | None = None
    coverage: str | None = None
    canonical_mapping: bool = False
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scope not in _SCOPE_RANK:
            raise ValueError(f"unsupported artifact scope: {self.scope!r}")
        if self.ready and self.scope == "missing":
            raise ValueError("ready artifact cannot have missing scope")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["metadata"] = dict(self.metadata)
        return value


class ArtifactRegistry:
    def __init__(self, statuses: Mapping[str, ArtifactStatus] | None = None):
        self._statuses: dict[str, ArtifactStatus] = dict(statuses or {})

    def register(self, status: ArtifactStatus) -> None:
        if not isinstance(status, ArtifactStatus):
            raise TypeError("status must be an ArtifactStatus")
        self._statuses[status.name] = status

    def get(self, name: str) -> ArtifactStatus:
        return self._statuses.get(
            str(name),
            ArtifactStatus(name=str(name), ready=False, reason="artifact_not_registered"),
        )

    def require(self, name: str, *, scope: str = "global") -> ArtifactStatus:
        if scope not in _SCOPE_RANK:
            raise ValueError(f"unsupported required scope: {scope!r}")
        status = self.get(name)
        if not status.ready or _SCOPE_RANK[status.scope] < _SCOPE_RANK[scope]:
            raise ArtifactUnavailable(
                f"artifact {name!r} requires scope={scope}, got ready={status.ready} "
                f"scope={status.scope}: {status.reason or 'not_ready'}"
            )
        return status

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: status.to_dict() for name, status in sorted(self._statuses.items())}

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: str | Path) -> "ArtifactRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        payload = data.get("artifacts", data)
        statuses = {}
        for name, value in payload.items():
            fields = dict(value)
            fields.pop("name", None)
            statuses[name] = ArtifactStatus(name=name, **fields)
        return cls(statuses)


__all__ = ["ArtifactRegistry", "ArtifactStatus", "ArtifactUnavailable"]
