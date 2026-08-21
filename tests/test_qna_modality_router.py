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


def test_router_loads_portable_raw_materializer_schema_before_global_merge(tmp_path):
    pd.DataFrame([
        {
            "video_id": "L27_V010", "text": "Hóa hồng Nhật Tảo quanh thiên địa",
            "start": 218.125, "end": 226.045, "kf_n": 146,
            "frame_idx": 5535, "pts_time": 221.4,
        }
    ]).to_parquet(tmp_path / "asr_chunks_l27_ts.parquet", index=False)
    np.save(tmp_path / "emb_cache_asr_l27_chunks.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    router = QNAModalityRouter(
        index_dir=tmp_path,
        embedder=_Embedder(),
        strict=False,
        expected_packs=("l27",),
        active_modalities=("asr",),
    )

    rows = router.global_candidates("Nguyễn Trung Trực", "asr", topk=1)

    assert rows[0]["video_id"] == "L27_V010"
    assert rows[0]["frame_idx"] == 5535
    assert rows[0]["pts_time"] == 221.4
    assert rows[0]["text"].startswith("Hóa hồng")


def test_global_poetry_lane_joins_asr_recital_to_nearby_ocr_entity_context():
    router = QNAModalityRouter.__new__(QNAModalityRouter)
    asr = pd.DataFrame([
        {
            "video_id": "L27_V010", "kf_n": 143, "frame_idx": 5400,
            "pts_time": 216.0,
            "chunk": "nhà thơ đã viết hai câu thơ đó là: Hóa hồng Nhật Tảo oanh thiên địa",
        },
        {
            "video_id": "L27_V010", "kf_n": 146, "frame_idx": 5535,
            "pts_time": 221.4,
            "chunk": "thơ đó là: Hóa hồng Nhật Tảo oanh thiên địa Kiếm bạt Kiên Giang khấp quỷ thần",
        },
    ])
    ocr = pd.DataFrame([
        {
            "video_id": "L27_V010", "kf_n": 153, "frame_idx": 5940,
            "pts_time": 237.6,
            "ocr_text": "Lễ hội Đình Thần Nguyễn Trung Trực, thành phố Rạch Giá",
        },
    ])
    router._load = lambda modality: (
        np.zeros((len(asr if modality == "asr" else ocr), 2), dtype=np.float32),
        asr if modality == "asr" else ocr,
    )

    rows = router.global_poetry_event_candidates(
        "Phóng sự về Nguyễn Trung Trực", "Hai câu thơ đó là gì?", topk=5
    )

    assert len(rows) == 1
    assert rows[0]["video_id"] == "L27_V010"
    # A global lane represents the evidenced event at video level; later
    # temporal expansion keeps both overlapping verse windows for the VLM.
    assert (rows[0]["kf_n"], rows[0]["frame_idx"]) in {(143, 5400), (146, 5535)}
    assert rows[0]["score_mode"] == "global_asr_poetry_event"
    assert rows[0]["provenance"]["supporting_context"]["modality"] == "ocr"


def test_global_poetry_lane_is_noop_for_non_poetry_question_without_loading_indexes():
    router = QNAModalityRouter.__new__(QNAModalityRouter)
    router._load = lambda _modality: (_ for _ in ()).throw(AssertionError("unexpected index load"))

    assert router.global_poetry_event_candidates(
        "Phóng sự về Nguyễn Trung Trực", "Đây là địa phương nào?"
    ) == []
