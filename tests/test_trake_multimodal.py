import pytest

from src.trake.contracts import TrakeContractError, validate_ranked_sequences, validate_sequence_path
from src.trake.multimodal import EventLevelMultimodalDante, MissingModalityRetriever


class FakeEventRetriever:
    def __init__(self, modality, rows):
        self.modality = modality
        self.rows = rows
        self.calls = []

    def search_event(self, event, *, top_k=100, candidate_videos=None):
        self.calls.append((event.index, event.description, tuple(candidate_videos or ())))
        allowed = set(map(str, candidate_videos)) if candidate_videos is not None else None
        return [
            {
                **row,
                "event_index": event.index,
                "modality": self.modality,
            }
            for row in self.rows.get(event.index, [])
            if allowed is None or str(row["video_id"]) in allowed
        ][:top_k]


def _rows(video_id="v"):
    return {
        0: [{"video_id": video_id, "score": 0.92, "frame_idx": 10, "kf_n": 1, "pts_time": 1.0}],
        1: [{"video_id": video_id, "score": 0.91, "frame_idx": 20, "kf_n": 2, "pts_time": 2.0}],
    }


def test_event_level_routing_uses_modality_per_event_and_aligns_strictly():
    visual = FakeEventRetriever("visual", _rows())
    asr = FakeEventRetriever("asr", _rows())
    aligner = EventLevelMultimodalDante({"visual": visual, "asr": asr})
    result = aligner.align([
        {"description": "scene", "modalities": ["visual"]},
        {"description": "spoken fact", "modalities": ["asr"]},
    ])
    assert result["results"][0]["video_id"] == "v"
    assert result["results"][0]["frame_ids"] == [10, 20]
    assert result["results"][0]["modalities"] == ["visual", "asr"]
    assert [call[0] for call in visual.calls] == [0]
    assert [call[0] for call in asr.calls] == [1]


def test_event_level_router_fails_closed_when_required_modality_is_missing():
    aligner = EventLevelMultimodalDante({"visual": FakeEventRetriever("visual", _rows())})
    with pytest.raises(MissingModalityRetriever):
        aligner.align([{"description": "spoken fact", "modalities": ["asr"]}])


def test_strict_sequence_contract_rejects_wrong_length_or_order():
    path = [
        {"event_index": 0, "video_id": "v", "modality": "visual", "frame_idx": 10, "pts_time": 1.0},
        {"event_index": 1, "video_id": "v", "modality": "asr", "frame_idx": 9, "pts_time": 0.9},
    ]
    with pytest.raises(TrakeContractError, match="strictly increasing"):
        validate_sequence_path(path, 2, video_id="v")
    with pytest.raises(TrakeContractError, match="exactly 2"):
        validate_sequence_path(path[:1], 2, video_id="v")


def test_ranked_sequence_contract_rejects_empty_and_overlong_results():
    with pytest.raises(TrakeContractError, match="no ranked answers"):
        validate_ranked_sequences([], 2)
    answers = [{"video_id": "v", "frame_ids": [1, 2]}] * 101
    with pytest.raises(TrakeContractError, match="100"):
        validate_ranked_sequences(answers, 2)
