import numpy as np
from scripts.profile_hot_paths import measure, stats

def test_stats_reports_percentiles():
    result = stats([.001, .002, .003], 0)
    assert result["n"] == 3 and result["p50_ms"] == 2.0

def test_measure_excludes_warmup():
    calls = []
    result = measure(lambda: calls.append(1), repeats=2, warmup=1)
    assert result["n"] == 2 and len(calls) == 3
