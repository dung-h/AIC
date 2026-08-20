"""Read-only canonical metadata schema for multimedia indexes.

The canonical frame key is ``(video_id, kf_n, frame_idx)``.  ``global_id`` is
the stable row id of a particular visual index and is never inferred across
different indexes unless their identity columns are verified equal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

FRAME_COLUMNS = ("global_id", "video_id", "kf_n", "frame_idx", "pts_time")
FRAME_KEY = ("video_id", "kf_n", "frame_idx")


@dataclass(frozen=True)
class SchemaIssue:
    severity: str
    source: str
    message: str


def canonicalize_frame_map(df: pd.DataFrame, source: str = "") -> tuple[pd.DataFrame, list[SchemaIssue]]:
    """Return a lightweight canonical view; never mutates the input frame."""
    out = df.copy()
    issues: list[SchemaIssue] = []
    if "global_id" not in out and "g" in out:
        out["global_id"] = out["g"]
    if "global_id" not in out:
        # Row order is the only stable identity available for legacy maps.
        out["global_id"] = pd.Series(range(len(out)), index=out.index, dtype="int64")
    missing = [c for c in ("video_id", "kf_n", "frame_idx", "pts_time") if c not in out]
    if missing:
        issues.append(SchemaIssue("error", source, f"missing required columns: {missing}"))
    for c in FRAME_COLUMNS:
        if c not in out:
            out[c] = pd.Series(index=out.index, dtype="float64")
    return out[list(FRAME_COLUMNS)], issues


def identity_set(df: pd.DataFrame) -> set[tuple]:
    return set(map(tuple, df.loc[:, list(FRAME_KEY)].itertuples(index=False, name=None)))
