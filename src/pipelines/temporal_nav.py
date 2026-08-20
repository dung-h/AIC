"""
Temporal Navigation API for AIC HCMCA 2026 Interactive Retrieval.

Provides keyframe neighbors for timeline navigation (VBS-style).
Based on research: VBS winners achieve 1.9× speedup with temporal navigation.
"""
import os
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from paths import INDEX_DIR
except Exception:
    INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"


class TemporalNavigator:
    """
    Manages temporal navigation across video keyframes.

    Loads global keyframe index once, provides fast neighbor queries.
    Thread-safe for FastAPI async endpoints.
    """

    def __init__(self, index_path: str = str(INDEX_DIR)):
        """
        Initialize navigator with global keyframe index.

        Args:
            index_path: Path to directory containing global_keyframes_vitl.parquet

        Raises:
            FileNotFoundError: If index file doesn't exist
            ValueError: If index has invalid schema
        """
        kf_file = os.path.join(index_path, "global_keyframes_vitl.parquet")
        if not os.path.exists(kf_file):
            raise FileNotFoundError(f"Keyframe index not found: {kf_file}")

        self.kmap = pd.read_parquet(kf_file)

        # Validate schema
        required_cols = {'g', 'video_id', 'kf_n', 'frame_idx', 'pts_time'}
        if not required_cols.issubset(self.kmap.columns):
            raise ValueError(f"Invalid schema. Required: {required_cols}, got: {set(self.kmap.columns)}")

        # Pre-sort for efficient range queries
        self.kmap = self.kmap.sort_values(['video_id', 'kf_n']).reset_index(drop=True)

        # Build video → index range lookup for O(1) slicing
        self._build_video_index()

    def _build_video_index(self):
        """Build {video_id: (start_idx, end_idx)} for fast video slicing."""
        grouped = self.kmap.groupby('video_id', sort=False)
        self._video_ranges = {
            vid: (group.index[0], group.index[-1] + 1)
            for vid, group in grouped
        }

    def get_neighbors(
        self,
        video_id: str,
        kf_n: int,
        window: int = 10,
        include_metadata: bool = True
    ) -> Dict[str, any]:
        """
        Get temporal neighbors around a keyframe.

        Args:
            video_id: Video ID (e.g., "K01_V001")
            kf_n: Keyframe number (1-indexed)
            window: Number of neighbors on each side (default: 10 → ±10 = 21 total)
            include_metadata: Include pts_time, frame_idx in response

        Returns:
            {
                "video_id": str,
                "center_kf_n": int,
                "neighbors": [
                    {"kf_n": int, "frame_idx": int, "pts_time": float, "is_center": bool},
                    ...
                ],
                "total": int,
                "has_prev": bool,  # Can navigate left
                "has_next": bool   # Can navigate right
            }

        Raises:
            ValueError: If video_id not found or kf_n invalid
        """
        # Validate video exists
        if video_id not in self._video_ranges:
            raise ValueError(f"Video '{video_id}' not found in index (total: {len(self._video_ranges)} videos)")

        # Slice video keyframes (O(1) with pre-built index)
        start_idx, end_idx = self._video_ranges[video_id]
        vid_frames = self.kmap.iloc[start_idx:end_idx]

        # Find center keyframe
        center_mask = vid_frames['kf_n'] == kf_n
        if not center_mask.any():
            valid_range = f"{vid_frames['kf_n'].min()}-{vid_frames['kf_n'].max()}"
            raise ValueError(f"Keyframe {kf_n} not found in {video_id} (valid: {valid_range})")

        center_pos = vid_frames[center_mask].index[0] - start_idx  # Position within video

        # Extract window
        window_start = max(0, center_pos - window)
        window_end = min(len(vid_frames), center_pos + window + 1)
        neighbors = vid_frames.iloc[window_start:window_end]

        # Format response
        result = {
            "video_id": video_id,
            "center_kf_n": int(kf_n),
            "neighbors": [],
            "total": len(neighbors),
            "has_prev": window_start > 0,
            "has_next": window_end < len(vid_frames)
        }

        for _, row in neighbors.iterrows():
            neighbor = {
                "kf_n": int(row['kf_n']),
                "is_center": int(row['kf_n']) == kf_n
            }
            if include_metadata:
                neighbor["frame_idx"] = int(row['frame_idx'])
                neighbor["pts_time"] = float(row['pts_time'])
            result["neighbors"].append(neighbor)

        return result

    def get_video_keyframes(self, video_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all keyframes for a video (for full timeline view).

        Args:
            video_id: Video ID
            limit: Max keyframes to return (None = all)

        Returns:
            List of {kf_n, frame_idx, pts_time}

        Raises:
            ValueError: If video not found
        """
        if video_id not in self._video_ranges:
            raise ValueError(f"Video '{video_id}' not found")

        start_idx, end_idx = self._video_ranges[video_id]
        vid_frames = self.kmap.iloc[start_idx:end_idx]

        if limit:
            vid_frames = vid_frames.head(limit)

        return [
            {
                "kf_n": int(row['kf_n']),
                "frame_idx": int(row['frame_idx']),
                "pts_time": float(row['pts_time'])
            }
            for _, row in vid_frames.iterrows()
        ]

    def navigate(self, video_id: str, current_kf_n: int, direction: int) -> Optional[Dict]:
        """
        Navigate to next/previous keyframe.

        Args:
            video_id: Video ID
            current_kf_n: Current keyframe number
            direction: -1 (previous) or +1 (next)

        Returns:
            {kf_n, frame_idx, pts_time} of target keyframe, or None if at boundary
        """
        if video_id not in self._video_ranges:
            return None

        start_idx, end_idx = self._video_ranges[video_id]
        vid_frames = self.kmap.iloc[start_idx:end_idx]

        current_mask = vid_frames['kf_n'] == current_kf_n
        if not current_mask.any():
            return None

        current_pos = vid_frames[current_mask].index[0] - start_idx
        target_pos = current_pos + direction

        if target_pos < 0 or target_pos >= len(vid_frames):
            return None  # Boundary

        target_row = vid_frames.iloc[target_pos]
        return {
            "kf_n": int(target_row['kf_n']),
            "frame_idx": int(target_row['frame_idx']),
            "pts_time": float(target_row['pts_time'])
        }


# Singleton instance (lazy-loaded by FastAPI)
_navigator: Optional[TemporalNavigator] = None


def get_navigator() -> TemporalNavigator:
    """Get or create singleton TemporalNavigator instance."""
    global _navigator
    if _navigator is None:
        _navigator = TemporalNavigator()
    return _navigator
