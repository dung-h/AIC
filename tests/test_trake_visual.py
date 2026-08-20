import numpy as np
import pandas as pd
import pytest
from src.pipelines.trake_visual import VisualTrakeDante

def pipe():
    m=pd.DataFrame({"video_id":["v"]*5,"pts_time":range(5),"kf_n":range(5),"frame_idx":range(5)})
    f=np.eye(5,dtype=np.float32)
    return VisualTrakeDante(m,f,lambda x: f[[1,3]])

def test_dante_order_and_lambda():
    r=pipe().align(["a","b"],video_id="v",lam=0)["results"][0]
    assert [x["kf_n"] for x in r["path"]] == [1,3]
    assert r["path"][0]["pts_time"] < r["path"][1]["pts_time"]

def test_impossible_alignment():
    p=pipe(); p.groups["v"].indices=p.groups["v"].indices[:1]
    assert p.align(["a","b"],video_id="v")["results"] == []

def test_empty_candidate_list_does_not_search_corpus():
    assert pipe().align(["a", "b"], candidate_videos=[])["results"] == []

@pytest.mark.parametrize("mode", ["asr", "hybrid"])
def test_unavailable_modes_are_explicit(mode):
    m=pd.DataFrame({"video_id":["v"],"pts_time":[0],"kf_n":[0],"frame_idx":[0]})
    with pytest.raises(ValueError, match="unavailable"):
        VisualTrakeDante(m,np.ones((1,2)),lambda x: np.ones((1,2)),mode=mode)
