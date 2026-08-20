"""
DANTE — Dynamic Alignment of Narrative Temporal Events (cho TRAKE).
Thuần CPU. Theo research/trake_temporal.md (đội AIO_Owlgorithms, Outstanding TRAKE).

Bài toán: cho N sub-event queries (q_1..q_N) mô tả chuỗi khoảnh khắc liên tiếp,
tìm trong MỖI video chuỗi keyframe t_1 < t_2 < ... < t_N tối đa tổng điểm khớp,
có phạt khoảng cách thời gian λ.

DP[i,t] = S[i,t] + max_{τ<t}(DP[i-1,τ] - λ(t-τ))
Tối ưu running-max: tách λ(t-τ)=λt-λτ -> O(N*T) per video.

Output: top-k video + chuỗi keyframe (backtrack).
"""
import numpy as np


def normalize_event_scores(S, valid_mask=None):
    """Normalize each event row independently to ``[0, 1]``.

    The visual lattice uses this helper only on its opt-in path. Per-event
    normalization prevents an event with a naturally larger cosine range from
    dominating the sequence score. Statistics are computed over all finite
    frames in the row, while ``valid_mask`` only controls which entries remain
    selectable. This avoids making every top-k candidate look perfect merely
    because the lattice excluded the rest of the video. Invalid lattice
    entries are returned as ``-inf`` so they can never be selected by
    constrained DP.
    """
    scores = np.asarray(S, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("S must be a two-dimensional score matrix")
    finite_mask = np.isfinite(scores)
    if valid_mask is None:
        mask = finite_mask
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != scores.shape:
            raise ValueError("valid_mask must match S")
        mask &= finite_mask
    normalized = np.full(scores.shape, -np.inf, dtype=np.float64)
    for event_index in range(scores.shape[0]):
        finite = finite_mask[event_index]
        if not np.any(finite):
            continue
        row = scores[event_index, finite]
        mean = float(np.mean(row))
        std = float(np.std(row))
        if std <= 1e-12:
            normalized[event_index, mask[event_index]] = 0.5
        else:
            z = (scores[event_index, mask[event_index]] - mean) / std
            z = np.clip(z, -60.0, 60.0)
            normalized[event_index, mask[event_index]] = 1.0 / (1.0 + np.exp(-z))
    return normalized


def sequence_quality(normalized_scores, path, pts_times=None, *,
                     coverage_weight=0.25, coherence_weight=0.10):
    """Return quality components for an already aligned event path.

    ``coverage`` is a geometric mean of per-event normalized scores, so one
    weak event lowers sequence quality instead of being hidden by the others.
    ``coherence`` rewards a stable temporal progression while the hard
    strictly-increasing constraint remains enforced by :func:`dante_align`.
    """
    scores = np.asarray(normalized_scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("normalized_scores must be two-dimensional")
    path = [int(index) for index in path]
    if len(path) != scores.shape[0]:
        raise ValueError("path length must equal the number of events")
    if any(index < 0 or index >= scores.shape[1] for index in path):
        raise ValueError("path contains an index outside normalized_scores")
    chosen = scores[np.arange(scores.shape[0]), path]
    if not np.all(np.isfinite(chosen)):
        raise ValueError("path contains an invalid score")
    clipped = np.clip(chosen, 0.0, 1.0)
    mean_score = float(np.mean(clipped))
    coverage = float(np.exp(np.mean(np.log(np.maximum(clipped, 1e-12)))))

    if len(path) < 2:
        coherence = 1.0
    else:
        if pts_times is None:
            times = np.arange(len(path), dtype=np.float64)
        else:
            times = np.asarray(pts_times, dtype=np.float64)
            if times.shape != (len(path),):
                raise ValueError("pts_times must have one value per path step")
        gaps = np.diff(times)
        if np.any(~np.isfinite(gaps)) or np.any(gaps <= 0):
            coherence = 0.0
        else:
            mean_gap = float(np.mean(gaps))
            coefficient_of_variation = float(np.std(gaps) / max(mean_gap, 1e-12))
            coherence = float(1.0 / (1.0 + coefficient_of_variation))

    combined = (mean_score + float(coverage_weight) * coverage
                + float(coherence_weight) * coherence)
    return {
        "mean_event_score": mean_score,
        "coverage": coverage,
        "coherence": coherence,
        "score": float(combined),
    }


def dante_align(S, lam=0.005, valid_mask=None):
    """
    S: (N, T) similarity matrix cho 1 video (N sub-events, T keyframes cùng video).
    Trả: best_score, path (list N chỉ số keyframe, tăng dần).

    ``valid_mask`` is optional and is used by the opt-in visual candidate
    lattice.  With the default ``None`` the original full-frame behavior is
    preserved.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError("S must be a two-dimensional score matrix")
    N, T = S.shape
    if N < 1 or T < 1:
        return -1e9, None
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != S.shape:
            raise ValueError("valid_mask must match S")
        valid_mask &= np.isfinite(S)
    else:
        valid_mask = np.isfinite(S)
    S = np.where(valid_mask, S, -np.inf)
    if T < N:
        return -1e9, None  # không đủ keyframe cho N sub-event theo thứ tự
    DP = np.full((N, T), -np.inf, dtype=np.float64)
    BP = np.full((N, T), -1, dtype=np.int32)  # backpointer
    DP[0] = np.where(valid_mask[0], S[0], -np.inf)
    for i in range(1, N):
        # running max của (DP[i-1, τ] + λτ) cho τ < t
        best_val = -np.inf; best_tau = -1
        for t in range(T):
            # cập nhật bằng τ = t-1 trước khi dùng cho t (đảm bảo τ < t)
            if (t-1 >= 0 and np.isfinite(DP[i-1, t-1])
                    and DP[i-1, t-1] + lam*(t-1) > best_val):
                best_val = DP[i-1, t-1] + lam*(t-1); best_tau = t-1
            if best_tau >= 0 and valid_mask[i, t]:
                DP[i, t] = S[i, t] + best_val - lam*t
                BP[i, t] = best_tau
    # nghiệm tốt nhất ở hàng cuối
    if not np.any(np.isfinite(DP[N-1])):
        return -1e9, None
    t_end = int(np.argmax(DP[N-1]))
    score = float(DP[N-1, t_end])
    # backtrack
    path = [0]*N; path[N-1] = t_end
    for i in range(N-1, 0, -1):
        path[i-1] = int(BP[i, path[i]])
        if path[i-1] < 0:
            return -1e9, None
    return score, path

def dante_search(event_vecs, feats, idmap, lam=0.005, topk=10):
    """
    event_vecs: (N, d) — embedding N sub-event queries (đã normalized).
    feats: (M, d) — toàn bộ keyframe embeddings (đã normalized).
    idmap: DataFrame có cột video_id, kf_n, frame_idx, pts_time (M dòng, cùng thứ tự feats).
    Trả: list (video_id, score, [(kf_n, frame_idx, pts_time)...]) top-k.
    """
    import pandas as pd
    sims_all = feats @ event_vecs.T  # (M, N)
    results = []
    for vid, g in idmap.groupby("video_id"):
        idx = g.index.values
        # sắp theo thời gian
        order = np.argsort(g.pts_time.values)
        idx_sorted = idx[order]
        S = sims_all[idx_sorted].T  # (N, T)
        score, path = dante_align(S, lam=lam)
        if path is None:
            continue
        rows = g.iloc[order].iloc[path]
        seq = [(int(r.kf_n), int(r.frame_idx), float(r.pts_time)) for r in rows.itertuples()]
        results.append((vid, score, seq))
    results.sort(key=lambda x: -x[1])
    return results[:topk]


if __name__ == "__main__":
    # Self-test: tạo data giả kiểm tra tính đúng + ràng buộc thứ tự
    np.random.seed(0)
    N, T, d = 3, 20, 16
    feats = np.random.randn(T, d); feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    # tạo 3 query khớp lần lượt frame 3, 9, 15 (đúng thứ tự)
    ev = feats[[3, 9, 15]].copy()
    S = (feats @ ev.T).T  # (N,T)
    score, path = dante_align(S, lam=0.01)
    print(f"Self-test path={path} (kỳ vọng tăng dần, gần [3,9,15]) score={score:.3f}")
    assert path[0] < path[1] < path[2], "DANTE phải trả thứ tự tăng dần!"
    print("OK: ràng buộc thứ tự thời gian thỏa mãn.")

    # test khi query đảo thứ tự (15,9,3) -> vẫn phải tăng dần theo thời gian
    ev2 = feats[[15, 9, 3]].copy()
    S2 = (feats @ ev2.T).T
    sc2, p2 = dante_align(S2, lam=0.01)
    print(f"Reversed-query path={p2} score={sc2:.3f} (DP ép tăng dần nên score thấp hơn)")
