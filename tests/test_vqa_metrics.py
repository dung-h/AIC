from src.eval.vqa_metrics import VQAExample, summarize_vqa


def test_vqa_metrics_keep_retrieval_and_answer_separate():
    rows = [
        VQAExample("V1", "V1", 10.0, 10.0, True, True),
        VQAExample("V2", "V1", 20.0, 10.0, True, None),
    ]
    result = summarize_vqa(rows)
    assert result["video_recall"] == 0.5
    assert result["frame_grounding"] == 0.5
    assert result["answer_on_ground_truth_frame"] == 1.0
    assert result["answer_on_retrieved_frame"] == 1.0
    assert result["end_to_end_grounded_answer"] == 0.5
