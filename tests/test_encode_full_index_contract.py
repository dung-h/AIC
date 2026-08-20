from __future__ import annotations

from src.encode.encode_full_index import _runtime_artifact_names


def test_runtime_visual_variant_names_match_pipeline_contract() -> None:
    assert _runtime_artifact_names("vitl") == ("global_siglip_vitl.npy", "global_keyframes_vitl.parquet")
    assert _runtime_artifact_names("so400m384") == ("global_so400m384.npy", "global_keyframes_so400m384.parquet")
    assert _runtime_artifact_names("my_encoder_v3") == ("global_my_encoder_v3.npy", "global_keyframes_my_encoder_v3.parquet")
