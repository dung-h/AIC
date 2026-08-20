from src.eval.benchmark_vqa_lattice_vlm import hit


def test_lattice_answer_hit_is_case_insensitive():
    assert hit("The car is white.", "White")
    assert not hit("The car is green.", "White")
