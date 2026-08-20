from src.core.local_vlm import LocalVLM


def test_local_vlm_pairwise_is_lazy():
    assert LocalVLM("unused").model is None
