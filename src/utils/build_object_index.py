"""
Build inverted object index from objects-aic25 detection JSON files.

Source: OpenImages V4 detections (Faster R-CNN), provided by AIC BTC.
Schema per JSON:
    detection_scores:          list[str]  (float values as strings)
    detection_class_entities:  list[str]  (human-readable labels, e.g. "Lantern")
    detection_class_names:     list[str]  (MID codes, e.g. "/m/01jfsr")
    detection_class_labels:    list[str]  (numeric label IDs as strings)
    detection_boxes:           list[list] (normalized [y1,x1,y2,x2])

Output:
    data/index/object_index.pkl  — {label: [(video_id, kf_n, score), ...]}
    data/index/object_vocab.json — sorted list of all labels with counts
"""
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OBJECTS_DIRS = [
    str(DATA_DIR / "raw" / "objects-aic25-b1"),
    str(DATA_DIR / "raw" / "objects-aic25-b2"),
]
IDX_DIR = str(DATA_DIR / "index")
SCORE_THRESHOLD = 0.5


def _parse_video_kf(json_path: str) -> Tuple[str, int]:
    """
    Extract (video_id, kf_n) from path like:
        .../objects-aic25-b1/K01_V001/042.json  →  ('K01_V001', 42)
    """
    parts = os.path.normpath(json_path).split(os.sep)
    video_id = parts[-2]
    kf_n = int(os.path.splitext(parts[-1])[0])
    return video_id, kf_n


def _iter_video_dirs(objects_dirs: List[str]):
    """
    Yield (obj_dir, video_dir_path, video_id) for every video directory.

    Handles the nested layout: objects-aic25-b1/objects/<video_id>/*.json
    """
    for obj_dir in objects_dirs:
        if not os.path.exists(obj_dir):
            continue
        # Video dirs may sit directly under obj_dir or under an "objects/" subdir.
        for root in (obj_dir, os.path.join(obj_dir, "objects")):
            if not os.path.isdir(root):
                continue
            for entry in sorted(os.listdir(root)):
                vdir = os.path.join(root, entry)
                if os.path.isdir(vdir):
                    yield obj_dir, vdir, entry


def _scan_video_dir(
    vdir: str,
    score_threshold: float,
) -> Tuple[List[Tuple[str, str, int, float]], int, int]:
    """
    Scan one video directory's JSON files.

    Returns:
        (detections, n_files, n_skipped)
        detections: list of (label, video_id, kf_n, score)
    """
    detections: List[Tuple[str, str, int, float]] = []
    n_files = 0
    n_skipped = 0

    for fname in os.listdir(vdir):
        if not fname.endswith(".json"):
            continue
        jf = os.path.join(vdir, fname)
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            n_skipped += 1
            continue

        n_files += 1
        scores_raw = data.get("detection_scores", [])
        labels = data.get("detection_class_entities", [])
        if not scores_raw or not labels:
            continue

        try:
            video_id, kf_n = _parse_video_kf(jf)
        except (ValueError, IndexError):
            n_skipped += 1
            continue

        for score_str, label in zip(scores_raw, labels):
            try:
                score = float(score_str)
            except (ValueError, TypeError):
                continue
            if score < score_threshold or not label:
                continue
            detections.append((label, video_id, kf_n, score))

    return detections, n_files, n_skipped


def build_index(
    objects_dirs: List[str] = OBJECTS_DIRS,
    score_threshold: float = SCORE_THRESHOLD,
    verbose: bool = True,
    checkpoint_every: int = 100,
    idx_dir: str = IDX_DIR,
) -> Dict[str, List[Tuple[str, int, float]]]:
    """
    Scan all detection JSON files per video directory and build inverted index.

    Resumable: progress is checkpointed every ``checkpoint_every`` video dirs to
    ``object_index.partial.pkl`` along with the set of completed video ids. A
    re-run skips already-processed videos.

    Returns:
        {label: [(video_id, kf_n, score), ...]} sorted by score desc per label
    """
    ckpt_path = os.path.join(idx_dir, "object_index.partial.pkl")

    index: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
    done_videos: set = set()
    total_files = 0
    total_detections = 0
    skipped = 0

    # Resume from checkpoint if present
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            for label, entries in ckpt["index"].items():
                index[label].extend(entries)
            done_videos = set(ckpt["done_videos"])
            total_files = ckpt.get("total_files", 0)
            total_detections = ckpt.get("total_detections", 0)
            skipped = ckpt.get("skipped", 0)
            if verbose:
                print(f"  [resume] {len(done_videos):,} videos already processed")
        except Exception as e:
            if verbose:
                print(f"  [resume failed, starting fresh] {e}")

    video_dirs = list(_iter_video_dirs(objects_dirs))
    if verbose:
        print(f"  Found {len(video_dirs):,} video directories")

    processed_since_ckpt = 0
    for i, (_obj_dir, vdir, video_id) in enumerate(video_dirs):
        if video_id in done_videos:
            continue

        detections, n_files, n_skipped = _scan_video_dir(vdir, score_threshold)
        for label, vid, kf_n, score in detections:
            index[label].append((vid, kf_n, score))
        total_files += n_files
        total_detections += len(detections)
        skipped += n_skipped
        done_videos.add(video_id)
        processed_since_ckpt += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"    {i + 1:,}/{len(video_dirs):,} dirs · "
                  f"{total_detections:,} detections · {len(index):,} labels")

        if processed_since_ckpt >= checkpoint_every:
            _save_checkpoint(ckpt_path, index, done_videos,
                             total_files, total_detections, skipped)
            processed_since_ckpt = 0

    # Sort each label's list by score descending
    for label in index:
        index[label].sort(key=lambda x: -x[2])

    if verbose:
        print(f"\n  Indexed: {total_files:,} files, "
              f"{total_detections:,} detections, "
              f"{len(index):,} unique labels, "
              f"{skipped} skipped")

    # Remove checkpoint once complete
    if os.path.exists(ckpt_path):
        try:
            os.remove(ckpt_path)
        except OSError:
            pass

    return dict(index)


def _save_checkpoint(
    ckpt_path: str,
    index: Dict[str, List[Tuple[str, int, float]]],
    done_videos: set,
    total_files: int,
    total_detections: int,
    skipped: int,
) -> None:
    """Atomically persist build progress for resume support."""
    tmp = ckpt_path + ".tmp"
    payload = {
        "index": {k: v for k, v in index.items()},
        "done_videos": list(done_videos),
        "total_files": total_files,
        "total_detections": total_detections,
        "skipped": skipped,
    }
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, ckpt_path)


def save_index(index: dict, idx_dir: str = IDX_DIR) -> None:
    """Persist index and vocab to disk."""
    os.makedirs(idx_dir, exist_ok=True)

    index_path = os.path.join(idx_dir, "object_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: {index_path}  ({os.path.getsize(index_path) / 1e6:.1f} MB)")

    # Vocabulary with counts
    vocab = sorted(
        [{"label": k, "count": len(v)} for k, v in index.items()],
        key=lambda x: -x["count"],
    )
    vocab_path = os.path.join(idx_dir, "object_vocab_v2.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {vocab_path}  ({len(vocab)} labels)")


def load_index(idx_dir: str = IDX_DIR) -> Dict[str, List[Tuple[str, int, float]]]:
    """Load pre-built index from disk."""
    path = os.path.join(idx_dir, "object_index.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Object index not found: {path}. Run build_object_index.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Building object inverted index...")
    index = build_index(verbose=True)
    if index:
        save_index(index)
        # Show top-20 labels
        top = sorted(index.items(), key=lambda x: -len(x[1]))[:20]
        print("\nTop 20 labels:")
        for label, entries in top:
            print(f"  {label:<30} {len(entries):>6} detections")
    else:
        print("No detections found. Check objects-aic25 directories.")
