from pathlib import Path

from src.utils.open_clip_local import cached_snapshot


def test_cached_snapshot_resolves_timm_repo_without_network(tmp_path: Path):
    snapshot = (
        tmp_path / "models--timm--ViT-L-16-SigLIP2-256" / "snapshots" / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    assert cached_snapshot("ViT-L-16-SigLIP2-256", cache_root=tmp_path) == snapshot


def test_cached_snapshot_rejects_incomplete_snapshot(tmp_path: Path):
    snapshot = (
        tmp_path / "models--timm--ViT-L-16-SigLIP2-256" / "snapshots" / "revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")

    assert cached_snapshot("ViT-L-16-SigLIP2-256", cache_root=tmp_path) is None
