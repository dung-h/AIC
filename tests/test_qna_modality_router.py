import numpy as np
import pandas as pd

from src.reranking.qna_modality_router import QNAModalityRouter


class _Embedder:
    def embed(self, texts, batch_size=1, normalize=True):
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def test_global_specialist_hit_carries_text_evidence_for_rrf_rescue():
    router = QNAModalityRouter.__new__(QNAModalityRouter)
    router.text_mode = "dense"
    router.embedder = _Embedder()
    router._indexes = {
        "asr": (
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            pd.DataFrame([
                {"video_id": "V1", "kf_n": 1, "frame_idx": 101,
                 "pts_time": 1.0, "chunk": "Nhiệt độ Nha Trang 25 độ."},
                {"video_id": "V2", "kf_n": 2, "frame_idx": 202,
                 "pts_time": 2.0, "chunk": "Một bản tin khác."},
            ]),
        )
    }

    rows = router.global_candidates("nhiệt độ", "asr", topk=1)

    assert rows[0]["video_id"] == "V1"
    assert rows[0]["text"] == "Nhiệt độ Nha Trang 25 độ."
    assert rows[0]["evidence"]["modality"] == "asr"
