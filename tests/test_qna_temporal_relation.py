"""Unit tests for the standalone temporal-relation resolver."""

import unittest

from src.pipelines.qna_temporal_relation import (
    AmbiguousEvidenceError,
    CanonicalMappingError,
    MissingEvidenceError,
    resolve_temporal_relation,
)


CANONICAL = [
    {"video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0},
    {"video_id": "V1", "kf_n": 8, "frame_idx": 80, "pts_time": 8.0},
    {"video_id": "V1", "kf_n": 9, "frame_idx": 90, "pts_time": 9.0},
]


class TemporalRelationTests(unittest.TestCase):
    def test_before_is_deterministic_and_contract_safe(self):
        result = resolve_temporal_relation(
            "the pan is placed",
            "the food is stirred",
            [
                {"event_index": 1, "video_id": "V1", "kf_n": 8, "frame_idx": 80, "pts_time": 8.0},
                {"event_index": 0, "video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0},
            ],
            CANONICAL,
        )
        signal = result.answer_signal()
        self.assertEqual(result.relation, "before")
        self.assertEqual(signal["status"], "valid")
        self.assertEqual(signal["video_id"], "V1")
        self.assertEqual(signal["frame_ids"], [30, 80])
        self.assertTrue(signal["answer"])
        self.assertEqual(len(signal["evidence"]), 2)

    def test_after_uses_semantic_event_order_not_input_order(self):
        result = resolve_temporal_relation(
            "event A",
            "event B",
            [
                {"event": "a", "video_id": "V1", "kf_n": 8, "frame_idx": 80, "pts_time": 8.0},
                {"event": "b", "video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0},
            ],
            CANONICAL,
        )
        self.assertEqual(result.relation, "after")
        self.assertEqual(result.answer_signal()["chronological_frame_ids"], [30, 80])

    def test_missing_event_is_rejected(self):
        with self.assertRaises(MissingEvidenceError):
            resolve_temporal_relation(
                "A", "B",
                [{"event_index": 0, "video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0}],
                CANONICAL,
            )

    def test_cross_video_evidence_is_rejected(self):
        with self.assertRaises(CanonicalMappingError):
            resolve_temporal_relation(
                "A", "B",
                [
                    {"event_index": 0, "video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0},
                    {"event_index": 1, "video_id": "V2", "kf_n": 8, "frame_idx": 80, "pts_time": 8.0},
                ],
                CANONICAL,
            )

    def test_noncanonical_frame_is_rejected(self):
        with self.assertRaises(CanonicalMappingError):
            resolve_temporal_relation(
                "A", "B",
                [
                    {"event_index": 0, "video_id": "V1", "kf_n": 3, "frame_idx": 999, "pts_time": 3.0},
                    {"event_index": 1, "video_id": "V1", "kf_n": 8, "frame_idx": 80, "pts_time": 8.0},
                ],
                CANONICAL,
            )

    def test_ambiguous_unscored_evidence_is_rejected(self):
        with self.assertRaises(AmbiguousEvidenceError):
            resolve_temporal_relation(
                "A", "B",
                [
                    {"event_index": 0, "video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0},
                    {"event_index": 0, "video_id": "V1", "kf_n": 9, "frame_idx": 90, "pts_time": 9.0},
                    {"event_index": 1, "video_id": "V1", "kf_n": 8, "frame_idx": 80, "pts_time": 8.0},
                ],
                CANONICAL,
            )

    def test_scored_candidates_choose_unique_best(self):
        result = resolve_temporal_relation(
            "A", "B",
            [
                {"event_index": 0, "video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0, "score": 0.3},
                {"event_index": 0, "video_id": "V1", "kf_n": 9, "frame_idx": 90, "pts_time": 9.0, "score": 0.9},
                {"event_index": 1, "video_id": "V1", "kf_n": 8, "frame_idx": 80, "pts_time": 8.0, "score": 0.5},
            ],
            CANONICAL,
        )
        self.assertEqual(result.evidence_a.frame_idx, 90)
        self.assertEqual(result.relation, "after")

    def test_equal_timestamp_is_rejected(self):
        canonical = [
            {"video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0},
            {"video_id": "V1", "kf_n": 8, "frame_idx": 80, "pts_time": 3.0},
        ]
        with self.assertRaises(AmbiguousEvidenceError):
            resolve_temporal_relation(
                "A", "B",
                [
                    {"event_index": 0, "video_id": "V1", "kf_n": 3, "frame_idx": 30, "pts_time": 3.0},
                    {"event_index": 1, "video_id": "V1", "kf_n": 8, "frame_idx": 80, "pts_time": 3.0},
                ],
                canonical,
            )


if __name__ == "__main__":
    unittest.main()
