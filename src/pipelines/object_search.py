"""
Object Search for AIC HCMC 2026.

Wraps the pre-built inverted object index (build_object_index.py) into
a search interface compatible with hybrid_search.py and web_ui.py.

Index format: {label: [(video_id, kf_n, score), ...]} sorted by score desc.

Query matching strategy:
1. Exact label match (case-insensitive)
2. Substring match for compound queries ("xe đạp" matches "Bicycle")
3. Label synonym expansion via a small Vietnamese↔English table
4. Results aggregated per (video_id, kf_n) taking max score
"""
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from paths import INDEX_DIR

# Result tuple: (video_id, frame_idx, pts_time, score)
# Note: object index stores kf_n (not frame_idx); frame_idx is resolved via kmap.
# For compatibility with the rest of the pipeline we expose kf_n as frame_idx
# and set pts_time=0.0 (the web_ui _attach_kf will fix both via kmap lookup).
Result = Tuple[str, int, float, float]  # (video_id, frame_idx_or_kfn, pts_time, score)


IDX_DIR = str(INDEX_DIR)

# ---------------------------------------------------------------------------
# Vietnamese ↔ English synonym table (AIC-relevant entities from EDA)
# ---------------------------------------------------------------------------
_SYNONYMS: Dict[str, List[str]] = {
    "xe đạp": ["Bicycle", "Cycling"],
    "đua xe": ["Bicycle", "Sports equipment"],
    "chảo": ["Frying pan", "Cooking"],
    "nấu ăn": ["Food", "Cooking", "Frying pan"],
    "đèn lồng": ["Lantern"],
    "lân": ["Dragon"],
    "sư rồng": ["Dragon"],
    "bão": ["Storm", "Wind"],
    "cây": ["Tree", "Plant"],
    "núi": ["Mountain"],
    "biển": ["Sea", "Water", "Ocean"],
    "cá": ["Fish"],
    "chó": ["Dog"],
    "mèo": ["Cat"],
    "toà nhà": ["Building", "Skyscraper", "Tower"],
    "cầu": ["Bridge"],
    "người": ["Person", "Human face"],
    "quần áo": ["Clothing"],
    "áo": ["Clothing"],
    "xe hơi": ["Car", "Vehicle"],
    "ô tô": ["Car", "Vehicle"],
    "xe máy": ["Motorcycle"],
    "máy bay": ["Airplane", "Aircraft"],
    "thuyền": ["Boat", "Watercraft"],
    "thịt": ["Meat"],
    "pizza": ["Pizza"],
    "cà chua": ["Tomato"],
    "rau": ["Vegetable", "Salad"],
    "bánh": ["Bread", "Pastry"],
}


class ObjectSearcher:
    """
    Fast object-based keyframe retrieval using pre-built inverted index.

    Thread-safe (index is read-only after load).
    """

    def __init__(self, idx_dir: str = IDX_DIR):
        """
        Load pre-built object index.

        Args:
            idx_dir: Directory containing object_index.pkl

        Raises:
            FileNotFoundError: If index not built yet (run build_object_index.py)
        """
        import pickle
        index_path = os.path.join(idx_dir, "object_index.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Object index not found at {index_path}. "
                "Run: python src/utils/build_object_index.py"
            )
        with open(index_path, "rb") as f:
            self._index: Dict[str, List[Tuple[str, int, float]]] = pickle.load(f)

        # Lowercase label lookup for case-insensitive matching
        self._lower_index: Dict[str, str] = {
            label.lower(): label for label in self._index
        }
        print(f"[ObjectSearch] Loaded {len(self._index):,} labels")

    def _resolve_labels(self, query: str) -> List[str]:
        """
        Map query text → relevant object labels in the index.

        Strategy (precision-first to avoid false positives like "cup"→"Cupboard"):
        1. Synonym expansion (Vietnamese keyword → English labels) — primary path
        2. Whole-phrase label match: full label appears as a phrase in the query
        3. Whole-word token match: a query word equals a label word (no substring)
        Returns a deduplicated list of matching index labels.
        """
        q_lower = query.lower()
        q_tokens = set(re.findall(r"\w+", q_lower, flags=re.UNICODE))
        matched: set = set()

        # 1. Synonym expansion (most reliable for Vietnamese queries)
        for vn_kw, en_labels in _SYNONYMS.items():
            if vn_kw in q_lower:
                for en in en_labels:
                    if en in self._index:
                        matched.add(en)

        # 2 & 3. Direct label matching with word boundaries (no loose substrings)
        for lbl_lower, lbl_orig in self._lower_index.items():
            lbl_words = lbl_lower.split()
            # 2. Multi-word label appearing verbatim as a phrase in the query
            if len(lbl_words) > 1 and lbl_lower in q_lower:
                matched.add(lbl_orig)
                continue
            # 3. Single-word label matched as a whole query word (length-guarded)
            if len(lbl_words) == 1 and len(lbl_lower) >= 3 and lbl_lower in q_tokens:
                matched.add(lbl_orig)

        return list(matched)

    def search(self, query: str, topk: int = 50) -> List[Result]:
        """
        Search keyframes by object labels inferred from query.

        Args:
            query: Natural language query (Vietnamese or English)
            topk: Maximum results to return

        Returns:
            List of (video_id, kf_n, pts_time=0.0, score) sorted by score desc.
            pts_time is 0.0 — the caller's kmap lookup will provide the real value.
        """
        labels = self._resolve_labels(query)
        if not labels:
            return []

        # Aggregate max score per (video_id, kf_n) across all matching labels
        best: Dict[Tuple[str, int], float] = {}
        for label in labels:
            for vid, kfn, score in self._index.get(label, []):
                key = (vid, kfn)
                if score > best.get(key, 0.0):
                    best[key] = score

        ranked = sorted(best.items(), key=lambda x: -x[1])[:topk]
        return [(vid, kfn, 0.0, score) for (vid, kfn), score in ranked]

    def get_labels_for_query(self, query: str) -> List[str]:
        """Return matched index labels (for debugging / UI display)."""
        return self._resolve_labels(query)

    @property
    def vocab_size(self) -> int:
        return len(self._index)
