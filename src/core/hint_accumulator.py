"""
Module 3 (L10): Hint Accumulator cho KIS interactive.

KIS thật: BTC cho 5 hint progressive (1 hint/phút). Hệ thống nhận hint mới,
tích lũy với hint cũ, re-rank top-K.

Strategy:
- Hint 1: query → top-K candidates (search space lớn)
- Hint 2: thêm thông tin → REFINE top-K (search trong subset top-K cũ HOẶC re-search nếu hint mới rộng hơn)
- Hint 3-5: tiếp tục accumulate

Concrete:
- Lưu state per-session: list[hint] + last_top_K
- Mỗi hint mới: combine với hint cũ (concat hoặc multi-query) → search → blend với last
- Output: refined top-K mỗi turn

Thuật toán: Multi-Query Score Accumulation
- Mỗi hint h_i → search → score_i per candidate
- Final score(c) = sum_i(weight_i * normalize(score_i(c)))
- weight increases với hint mới (trust ít hơn hint sớm)

Cải tiến: track hint nào cải thiện rank GT — adaptive weights (learn).
"""
import os, sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HintSession:
    """1 session KIS, tích lũy hints."""
    session_id: str
    hints: List[str] = field(default_factory=list)
    # Per-hint: list of (video_id, frame_idx, pts_time, score) sorted desc
    hint_results: List[List[Tuple]] = field(default_factory=list)
    # Last accumulated ranking
    last_combined: List[Tuple] = field(default_factory=list)


class HintAccumulator:
    """
    Mỗi search() nhận:
    - search_fn: callable(query) → list of (video_id, frame_idx, pts_time, score)
    - session: HintSession
    - new_hint: str
    Trả về top-K combined.

    Tách biệt với pipeline cụ thể — work với bất kỳ retrieval function nào.
    """

    def __init__(self, weight_decay=0.85):
        # Hint sớm vẫn quan trọng nhưng giảm trọng số khi accumulate
        # weight cho hint i (i=0 là sớm nhất) = decay^(N-1-i) where N = total hints
        self.weight_decay = weight_decay

    def add_hint(self, search_fn, session: HintSession, new_hint: str,
                 topk_per_hint=200, topk_final=50):
        """Add hint, search, re-accumulate."""
        session.hints.append(new_hint)
        # Combine hints into 1 query (concat) HOẶC search độc lập rồi blend.
        # Phương án 1: concat (đơn giản, baseline)
        # Phương án 2: search độc lập từng hint, blend score (chính xác hơn)
        # Dùng phương án 2.
        new_results = search_fn(new_hint)[:topk_per_hint]
        session.hint_results.append(new_results)

        # Build candidate pool: union of all (video_id, frame_idx) seen across hints
        cand_set = {}
        # weights: hint i (0-indexed, latest = highest weight)
        N = len(session.hint_results)
        weights = [self.weight_decay ** (N - 1 - i) for i in range(N)]
        # Normalize per-hint scores then sum * weight
        for i, results in enumerate(session.hint_results):
            if not results: continue
            scores = np.array([r[3] for r in results], dtype=np.float32)
            # Min-max normalize to [0,1]
            mn, mx = scores.min(), scores.max()
            normed = (scores - mn) / (mx - mn + 1e-8)
            for j, r in enumerate(results):
                key = (r[0], r[1])  # (video_id, frame_idx)
                if key not in cand_set:
                    cand_set[key] = {"video_id": r[0], "frame_idx": r[1],
                                     "pts_time": r[2], "score": 0.0,
                                     "hits": 0}
                cand_set[key]["score"] += weights[i] * float(normed[j])
                cand_set[key]["hits"] += 1

        # Bonus for candidates appearing across multiple hints (corroborated)
        for v in cand_set.values():
            v["score"] *= 1.0 + 0.15 * (v["hits"] - 1)  # 15% boost per extra hit

        # Sort
        ranked = sorted(cand_set.values(), key=lambda x: -x["score"])[:topk_final]
        session.last_combined = [(c["video_id"], c["frame_idx"],
                                  c["pts_time"], c["score"]) for c in ranked]
        return session.last_combined


if __name__ == "__main__":
    print("=== HintAccumulator self-test ===")

    # Mock search function
    def mock_search(query):
        # Pretend GT = ("VIDEO_A", 100, 5.0). Quality grows as query gets specific.
        if "general" in query:
            return [("VIDEO_X", 50, 2.0, 0.8),
                    ("VIDEO_Y", 30, 1.0, 0.7),
                    ("VIDEO_A", 100, 5.0, 0.6)]  # GT at #3
        elif "specific" in query:
            return [("VIDEO_A", 100, 5.0, 0.9),  # GT at #1
                    ("VIDEO_X", 50, 2.0, 0.5),
                    ("VIDEO_Z", 80, 3.0, 0.4)]
        else:
            return [("VIDEO_Y", 30, 1.0, 0.7),
                    ("VIDEO_A", 100, 5.0, 0.65),  # GT at #2
                    ("VIDEO_W", 20, 0.5, 0.5)]

    acc = HintAccumulator()
    sess = HintSession(session_id="test")

    # Hint 1: general
    r1 = acc.add_hint(mock_search, sess, "general scene", topk_per_hint=10, topk_final=5)
    print(f"\nAfter hint 1 (general):")
    for i, c in enumerate(r1[:3]): print(f"  {i+1}. {c[0]} score={c[3]:.3f}")
    rank1 = next(i for i, c in enumerate(r1) if c[0] == "VIDEO_A") + 1
    print(f"  GT VIDEO_A rank: {rank1}")

    # Hint 2: more info
    r2 = acc.add_hint(mock_search, sess, "scene with details", topk_per_hint=10, topk_final=5)
    print(f"\nAfter hint 2 (details):")
    for i, c in enumerate(r2[:3]): print(f"  {i+1}. {c[0]} score={c[3]:.3f}")
    rank2 = next(i for i, c in enumerate(r2) if c[0] == "VIDEO_A") + 1
    print(f"  GT VIDEO_A rank: {rank2}")

    # Hint 3: specific
    r3 = acc.add_hint(mock_search, sess, "specific event", topk_per_hint=10, topk_final=5)
    print(f"\nAfter hint 3 (specific):")
    for i, c in enumerate(r3[:3]): print(f"  {i+1}. {c[0]} score={c[3]:.3f}")
    rank3 = next(i for i, c in enumerate(r3) if c[0] == "VIDEO_A") + 1
    print(f"  GT VIDEO_A rank: {rank3}")

    assert rank3 == 1, f"GT should reach top-1 by hint 3, got {rank3}"
    print("\n  ✓ Progressive refinement: rank improved {rank1}→{rank2}→{rank3}".format(
        rank1=rank1, rank2=rank2, rank3=rank3))
    print(f"  ✓ Total hints stored: {len(sess.hints)}")
    print(f"  ✓ Candidates accumulated: {len(sess.last_combined)}")
    print("\n=== PASS ===")
