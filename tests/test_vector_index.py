import json
import numpy as np
import pandas as pd
import pytest

from src.indexing.vector_index import IndexValidationError, ReadOnlyVectorIndex, build_index

def files(tmp_path):
    x = np.eye(8, 4, dtype=np.float32)
    fp = tmp_path / "features.npy"; np.save(fp, x)
    mp = tmp_path / "map.parquet"
    pd.DataFrame({"global_id": range(8), "video_id": ["v"] * 8, "kf_n": range(8), "frame_idx": range(8), "pts_time": np.arange(8, dtype=float)}).to_parquet(mp)
    return fp, mp

def test_exact_search_and_manifest(tmp_path):
    fp, mp = files(tmp_path); out = tmp_path / "x.faiss"
    build_index(fp, mp, out)
    idx = ReadOnlyVectorIndex.load(out)
    scores, ids = idx.search(np.eye(4, dtype=np.float32), 1)
    assert np.array_equal(ids[:, 0], np.arange(4)); assert np.allclose(scores[:, 0], 1)

def test_map_mutation_fails_fast(tmp_path):
    fp, mp = files(tmp_path); out = tmp_path / "x.faiss"; build_index(fp, mp, out)
    df = pd.read_parquet(mp); df.loc[0, "frame_idx"] = 999; df.to_parquet(mp)
    with pytest.raises(IndexValidationError): ReadOnlyVectorIndex.load(out)

def test_build_does_not_overwrite(tmp_path):
    fp, mp = files(tmp_path); out = tmp_path / "x.faiss"; build_index(fp, mp, out)
    with pytest.raises(FileExistsError): build_index(fp, mp, out)
