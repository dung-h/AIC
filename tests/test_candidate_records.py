from src.core.candidates import CandidateRecord, union_candidates


def candidate(video, kf, score, source):
    return CandidateRecord(video, kf, kf * 10, float(kf), score, source)


def test_union_deduplicates_frames_and_preserves_provenance():
    result = union_candidates({
        "visual": [candidate("V1", 2, .9, "visual"), candidate("V2", 1, .8, "visual")],
        "ocr": [candidate("V1", 2, .7, "ocr"), candidate("V3", 4, .6, "ocr")],
    })

    assert [(row.video_id, row.kf_n) for row in result] == [("V1", 2), ("V2", 1), ("V3", 4)]
    assert result[0].evidence["sources"] == ["visual", "ocr"]


def test_union_limits_each_branch_before_union_limit():
    result = union_candidates({
        "visual": [candidate("V1", 1, .9, "visual"), candidate("V1", 2, .8, "visual")],
        "asr": [candidate("V2", 1, .95, "asr"), candidate("V3", 1, .7, "asr")],
    }, topk_per_branch=1, topk_union=2)

    assert [(row.video_id, row.kf_n) for row in result] == [("V2", 1), ("V1", 1)]
