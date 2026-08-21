import numpy as np
import pytest

from src.utils.dante import dante_align


def test_dante_uses_canonical_time_for_nonuniform_timeline_penalty():
    # Position-based DANTE prefers [0, 2]. On the real timeline it spans
    # 101 seconds, while [1, 2] spans one second and must win for lambda > 0.
    scores = np.array([
        [1.0, 0.999, -1.0],
        [-1.0, -1.0, 0.997],
    ])

    _, legacy_path = dante_align(scores, lam=0.001)
    _, timestamped_path = dante_align(
        scores, lam=0.001, timeline_times=[0.0, 100.0, 101.0]
    )
    _, zero_lambda_path = dante_align(
        scores, lam=0.0, timeline_times=[0.0, 100.0, 101.0]
    )

    assert legacy_path == [0, 2]
    assert timestamped_path == [1, 2]
    assert zero_lambda_path == legacy_path


@pytest.mark.parametrize("times", ([0.0, 1.0, 1.0], [0.0, np.nan, 2.0]))
def test_dante_rejects_noncanonical_timeline(times):
    with pytest.raises(ValueError, match="timeline_times must be"):
        dante_align(np.ones((2, 3)), timeline_times=times)
