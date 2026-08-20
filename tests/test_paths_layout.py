from src.utils.paths import DATA_DIR, KEYFRAMES_DIR


def test_keyframe_root_has_no_redundant_nested_segment():
    assert KEYFRAMES_DIR == DATA_DIR / "keyframes"
