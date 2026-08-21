from __future__ import annotations

from src.vqa.asr_event_localizer import ASRPoetryEventLocalizer


def _l28_like_rows() -> list[dict]:
    return [
        {
            "video_id": "L28_V020", "kf_n": 130, "frame_idx": 6006,
            "pts_time": 240.24,
            "chunk": "Đình thần Nguyễn Trung Trực ở Kiên Giang hôm nay có đông khách viếng thăm.",
        },
        {
            "video_id": "L28_V020", "kf_n": 139, "frame_idx": 6650,
            "pts_time": 266.0,
            "chunk": "Sau đây là một phần giới thiệu về lễ tưởng niệm.",
        },
        {
            "video_id": "L28_V020", "kf_n": 142, "frame_idx": 6884,
            "pts_time": 275.36,
            "chunk": "Thư kiếm tùng nhung tự thiếu niên.",
        },
        {
            "video_id": "L28_V020", "kf_n": 143, "frame_idx": 7020,
            "pts_time": 280.80,
            "chunk": "Anh hùng nhược ngộ vô chung địa, bảo hận thâm cừu bất đới thiên.",
        },
        {
            "video_id": "L28_V021", "kf_n": 12, "frame_idx": 550,
            "pts_time": 22.0,
            "chunk": "Thư kiếm tùng nhung tự thiếu niên, một câu hát trong chương trình khác.",
        },
    ]


def _actual_p1_19_like_rows_with_duplicate_kf142() -> list[dict]:
    """Three canonical moments, with a duplicate ASR chunk at kf142."""
    return [
        # A different verse-like passage after a broad ``Kiên Giang`` mention
        # must not be mistaken for the Nguyễn Trung Trực recital.
        {
            "video_id": "L28_V020", "kf_n": 85, "frame_idx": 1755,
            "pts_time": 68.5,
            "chunk": "Kiên Giang có những kinh đào mang tên Cái Sắn thuở nào còn vang.",
        },
        {
            "video_id": "L28_V020", "kf_n": 86, "frame_idx": 1889,
            "pts_time": 74.4,
            "chunk": "Mấy ai nghe chẳng ngỡ ngàng dọc sông Cái Sắn là hàng cát kinh.",
        },
        {
            "video_id": "L28_V020", "kf_n": 130, "frame_idx": 6006,
            "pts_time": 240.24,
            "chunk": "Đình thần thờ vị anh hùng dân tộc Nguyễn Trung Trực tại Kiên Giang.",
        },
        {
            "video_id": "L28_V020", "kf_n": 142, "frame_idx": 6884,
            "pts_time": 275.36,
            "chunk": "Thư kiếm tùng nhung tự thiếu niên.",
        },
        {
            "video_id": "L28_V020", "kf_n": 142, "frame_idx": 6884,
            "pts_time": 275.36,
            "chunk": "Thư kiếm tùng nhung tự thiếu niên, lời thơ được ngâm đọc.",
        },
        {
            "video_id": "L28_V020", "kf_n": 143, "frame_idx": 7020,
            "pts_time": 280.80,
            "chunk": "Anh hùng nhược ngộ vô chung địa, bảo hận thâm cừu bất đới thiên.",
        },
    ]


def test_selects_l28_like_consecutive_verse_chunks_with_local_context():
    localizer = ASRPoetryEventLocalizer()

    rows = localizer.localize(
        _l28_like_rows(),
        query="Đền thờ anh hùng Nguyễn Trung Trực ở Kiên Giang",
        question="Hai câu thơ được đọc là gì?",
        shortlisted_video_ids=["L28_V020", "L28_V021"],
    )

    assert [row["kf_n"] for row in rows] == [142, 143]
    assert [row["frame_idx"] for row in rows] == [6884, 7020]
    assert all(row["modality"] == "asr" for row in rows)
    assert all(row["score_mode"] == "local_asr_poetry_event" for row in rows)
    assert all(row["provenance"]["source"] == "local_canonical_asr" for row in rows)
    assert all(row["provenance"]["cluster"]["kf_ns"] == [142, 143] for row in rows)
    assert all(row["provenance"]["supporting_context"]["kf_n"] == 130 for row in rows)
    assert all(row["provenance"]["supporting_context"]["gap_before_cluster_s"] <= 60 for row in rows)
    assert localizer.last_diagnostic["status"] == "ok"


def test_normal_spoken_fact_never_activates_even_when_asr_has_short_chunks():
    localizer = ASRPoetryEventLocalizer()

    rows = localizer.localize(
        _l28_like_rows(),
        query="Đền thờ anh hùng Nguyễn Trung Trực ở Kiên Giang",
        question="Ngôi đình này nằm ở tỉnh nào?",
        shortlisted_video_ids=["L28_V020"],
    )

    assert rows == []
    assert localizer.last_diagnostic == {"status": "inactive_non_poetry"}


def test_actual_p1_19_query_uses_name_phrase_anchor_and_dedupes_kf142():
    localizer = ASRPoetryEventLocalizer()

    rows = localizer.localize(
        _actual_p1_19_like_rows_with_duplicate_kf142(),
        query=(
            "Trong đoạn video có 2 câu thơ của một nhà thơ ca ngợi anh hùng "
            "Nguyễn Trung Trực trong đình thần Nguyễn Trung Trực tại Kiên Giang."
        ),
        question="Hai câu thơ đó là gì?",
        shortlisted_video_ids=["L28_V020"],
    )

    assert [row["kf_n"] for row in rows] == [142, 143]
    assert [row["frame_idx"] for row in rows] == [6884, 7020]
    assert rows[0]["provenance"]["supporting_context"]["kf_n"] == 130
    assert rows[0]["provenance"]["supporting_context"]["anchor_match"]["matched_phrase"] == "nguyen trung truc"
    assert rows[0]["provenance"]["canonical_dedup"] == {
        "key": ["L28_V020", 142],
        "input_row_count": 2,
        "strategy": "strongest_verse_text",
    }


def test_generic_poetry_query_fails_closed_without_entity_or_content_anchor():
    localizer = ASRPoetryEventLocalizer()

    rows = localizer.localize(
        _actual_p1_19_like_rows_with_duplicate_kf142(),
        query="Trong đoạn video có hai câu thơ được đọc.",
        question="Hai câu thơ đó là gì?",
        shortlisted_video_ids=["L28_V020"],
    )

    assert rows == []
    assert localizer.last_diagnostic["status"] == "insufficient_entity_context"
    assert localizer.last_diagnostic["entity_anchors"] == {
        "strategy": "bounded_content_fallback",
        "proper_phrases": [],
        "content_terms": [],
    }


def test_fails_closed_when_any_scoped_asr_row_lacks_canonical_frame():
    localizer = ASRPoetryEventLocalizer()
    rows = _l28_like_rows()
    del rows[3]["frame_idx"]

    result = localizer.localize(
        rows,
        query="Đền thờ anh hùng Nguyễn Trung Trực ở Kiên Giang",
        question="Hai câu thơ được đọc là gì?",
        shortlisted_video_ids=["L28_V020"],
    )

    assert result == []
    assert localizer.last_diagnostic == {"status": "missing_or_invalid_canonical_columns"}


def test_word_window_recital_accepts_following_entity_context_and_text_schema():
    """v3 word windows can contain the lead-in plus the complete couplet."""
    rows = [
        {
            "video_id": "L27_V010", "kf_n": 143, "frame_idx": 5430,
            "pts_time": 216.0,
            "text": (
                "nhà thơ đã viết hai câu thơ đó là: Hóa hồng Nhật Tảo "
                "oanh thiên địa"
            ),
        },
        {
            "video_id": "L27_V010", "kf_n": 146, "frame_idx": 5535,
            "pts_time": 221.4,
            "text": (
                "thơ đó là: Hóa hồng Nhật Tảo oanh thiên địa Kiếm bạt "
                "Kiên Giang khấp quỷ thần"
            ),
        },
        {
            "video_id": "L27_V010", "kf_n": 157, "frame_idx": 5871,
            "pts_time": 246.88,
            "text": "người anh hùng dân tộc Nguyễn Trung Trực đã hy sinh.",
        },
    ]

    localizer = ASRPoetryEventLocalizer()
    result = localizer.localize(
        rows,
        query="Phóng sự về Nguyễn Trung Trực",
        question="Hai câu thơ đó là gì?",
        shortlisted_video_ids=["L27_V010"],
    )

    assert [row["kf_n"] for row in result] == [143, 146]
    assert all(row["provenance"]["supporting_context"]["context_direction"] == "after"
               for row in result)
    assert all(row["provenance"]["supporting_context"]["kf_n"] == 157
               for row in result)
