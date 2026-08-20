from src.service.contracts import RetrievalResult, normalize_result


def test_normalizes_kis_tuple_through_metadata_adapter():
    result = normalize_result(("v", 10, 3, .8), metadata_lookup=lambda v, f: (3, 1.5))
    assert result == RetrievalResult("v", 10, 3, 1.5, .8)


def test_normalizes_vkis_tuple_through_metadata_adapter():
    result = normalize_result(("v", 10, 12.5, .8), metadata_lookup=lambda v, f: (7, 12.5))
    assert result == RetrievalResult("v", 10, 7, 12.5, .8)


def test_normalizes_mapping_result():
    result = normalize_result({"video_id": "v", "frame_idx": 2, "pts_time": 3.0})
    assert result.video_id == "v" and result.frame_idx == 2 and result.pts_time == 3.0


def test_mapping_metadata_mismatch_fails_closed():
    try:
        normalize_result(
            {"video_id": "v", "frame_idx": 2, "kf_n": 9, "pts_time": 3.0},
            metadata_lookup=lambda _v, _f: (7, 3.0),
        )
    except ValueError as error:
        assert "canonical map" in str(error)
    else:
        raise AssertionError("metadata mismatch must fail closed")
