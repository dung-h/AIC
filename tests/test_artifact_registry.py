import json

import pytest

from src.artifacts import ArtifactRegistry, ArtifactStatus, ArtifactUnavailable


def test_require_is_scope_aware():
    registry = ArtifactRegistry({
        "asr": ArtifactStatus("asr", True, "partial", row_count=10),
    })
    assert registry.require("asr", scope="partial").row_count == 10
    with pytest.raises(ArtifactUnavailable):
        registry.require("asr", scope="global")


def test_snapshot_roundtrip(tmp_path):
    path = tmp_path / "artifacts.json"
    registry = ArtifactRegistry({"visual": ArtifactStatus("visual", True, "global", dimension=1024)})
    registry.to_json(path)
    loaded = ArtifactRegistry.from_json(path)
    assert loaded.get("visual").dimension == 1024
    json.dumps(loaded.snapshot())
