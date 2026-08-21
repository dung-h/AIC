"""Deterministic multi-modal evidence packaging for the Q&A answer stage.

This module deliberately does not own an ASR/OCR index.  It consumes rows
already loaded by the pipeline and turns them into a bounded, auditable packet
for the answerer.  Text evidence is never used to change the submission
contract: the caller still emits the candidate's canonical frame index.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping
import re
import unicodedata


_TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)
_NONANSWER = {
    "", "unknown", "n/a", "na", "none", "null", "evidence-only",
    "evidence only", "no answer", "not available",
}


def _repair_mojibake(value: Any) -> str:
    text = "" if value is None else str(value)
    def score(candidate: str) -> int:
        return (
            sum(candidate.count(marker) for marker in ("Ã", "Â", "â", "ð", "Ð", "Ñ", "ì", "í", "î", "ï", "ä", "å", "ç", "Æ", "æ", "ƒ", "„", "™", "»"))
            + sum(candidate.count(pair) for pair in ("â€", "â€™", "Ã", "Â", "ì", "í"))
            + 2 * sum(0x80 <= ord(char) <= 0x9F for char in candidate)
        )

    for _ in range(2):
        candidates = []
        for encoding in ("cp1252", "latin1"):
            try:
                candidate = text.encode(encoding).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if "�" not in candidate:
                candidates.append(candidate)
        if not candidates:
            break
        repaired = min(candidates, key=score)
        if score(repaired) >= score(text):
            break
        text = repaired
    return unicodedata.normalize("NFKC", text)


def normalize_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", _repair_mojibake(value)).split()).strip()


def _tokens(value: Any) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(normalize_text(value))
            if len(token) > 1}


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_value(row: Any, *names: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row[name]
        return default
    for name in names:
        try:
            value = getattr(row, name)
        except AttributeError:
            continue
        return value
    return default


def _iter_rows(rows: Iterable[Any]) -> Iterable[Any]:
    """Accept both normal iterables and pandas DataFrames without coupling."""
    if hasattr(rows, "itertuples"):
        return rows.itertuples(index=False)
    return rows


def _materialize_rows(rows: Iterable[Any] | None) -> list[Any] | None:
    """Materialize once so status classification and selection see identical rows."""
    if rows is None:
        return None
    return list(_iter_rows(rows))


def _temporal_distance(anchor: float, start: float | None, end: float | None) -> float:
    if start is None and end is None:
        return float("inf")
    start = anchor if start is None else start
    end = start if end is None else end
    if start <= anchor <= end:
        return 0.0
    return min(abs(anchor - start), abs(anchor - end))


def _select_rows(rows: Iterable[Any], *, video_id: str, anchor_time: float,
                 query_text: str, kind: str, window: float, limit: int,
                 anchor_kf_n: int | None = None, adjacent_kf_radius: int = 0,
                 adjacent_kf_window: float = 5.0) -> list[dict[str, Any]]:
    if rows is None:
        return []
    query_tokens = _tokens(query_text)
    selected: list[tuple[tuple[float, float, int], dict[str, Any]]] = []
    for index, raw in enumerate(_iter_rows(rows)):
        row_video = _row_value(raw, "video_id", "vid", "video", default=video_id)
        if str(row_video) != str(video_id):
            continue
        if kind == "asr":
            text = normalize_text(_row_value(raw, "chunk", "text", "transcript", default=""))
            start = _number(_row_value(raw, "start", "start_time", default=None))
            end = _number(_row_value(raw, "end", "end_time", default=start))
            point = start if start is not None else end
            row_kf_n = _number(_row_value(raw, "kf_n", "keyframe", default=None))
            payload: dict[str, Any] = {
                "source": "asr", "text": text, "start_time": start,
                "end_time": end, "timestamp": point,
            }
        elif kind == "ocr":
            text = normalize_text(_row_value(raw, "ocr_text", "text", default=""))
            point = _number(_row_value(raw, "pts_time", "timestamp", "time", default=None))
            start = point
            end = point
            row_kf_n = _number(_row_value(raw, "kf_n", "keyframe", default=None))
            payload = {"source": "ocr", "text": text, "start_time": start,
                       "end_time": end, "timestamp": point}
        else:
            raise ValueError(f"unsupported evidence kind: {kind}")
        if not text:
            continue
        distance = _temporal_distance(anchor_time, start, end)
        kf_distance = float("inf")
        if anchor_kf_n is not None and row_kf_n is not None:
            kf_distance = abs(float(row_kf_n) - float(anchor_kf_n))
        # Time is the primary safety boundary.  OCR is sampled sparsely, so a
        # canonical adjacent keyframe may be used as a bounded rescue only
        # when its timestamp remains close to the anchor.  This avoids
        # attaching an arbitrary same-video title from a distant scene.
        adjacent_rescue = (
            kind == "ocr"
            and adjacent_kf_radius > 0
            and kf_distance <= float(adjacent_kf_radius)
            and distance <= float(adjacent_kf_window)
        )
        if distance > window and not adjacent_rescue:
            continue
        lexical = len(query_tokens.intersection(_tokens(text)))
        # Lexical relevance is only a tie-breaker inside a time window.  It
        # cannot pull evidence from another time range or another video.
        if row_kf_n is not None:
            payload["kf_n"] = int(row_kf_n)
        row_frame_idx = _number(_row_value(raw, "frame_idx", "frame_id", default=None))
        if row_frame_idx is not None:
            payload["frame_idx"] = int(row_frame_idx)
        payload["kf_distance"] = None if kf_distance == float("inf") else kf_distance
        selected.append(((-float(lexical), distance, index), payload))
    selected.sort(key=lambda item: item[0])
    selected = selected[:max(0, int(limit))]
    selected.sort(key=lambda item: (
        float(item[1].get("timestamp") if item[1].get("timestamp") is not None else float("inf")),
        item[0][2],
    ))
    for rank, (_, item) in enumerate(selected, 1):
        item["rank"] = rank
        item["distance_s"] = _temporal_distance(
            anchor_time, item.get("start_time"), item.get("end_time"))
    return [item for _, item in selected]


def _select_rows_at_anchors(
    rows: Iterable[Any], *, video_id: str, anchor_times: Iterable[float],
    query_text: str, kind: str, window: float, limit: int,
    anchor_kf_n: int | None = None, adjacent_kf_radius: int = 0,
    adjacent_kf_window: float = 5.0,
) -> list[dict[str, Any]]:
    """Collect bounded text evidence around validated frame anchors.

    A routed candidate keeps a visual output anchor but may also carry a
    specialist ASR/OCR frame selected from the same video.  Searching text
    only around the visual anchor drops that specialist evidence whenever the
    two retrieval hits are farther apart than the normal context window.  The
    extra anchors are already materialized candidates, so this helper does
    not broaden retrieval to arbitrary timestamps or videos.
    """
    unique_times: list[float] = []
    seen_times: set[float] = set()
    for raw_time in anchor_times:
        try:
            value = float(raw_time)
        except (TypeError, ValueError):
            continue
        if value != value or value in seen_times:
            continue
        seen_times.add(value)
        unique_times.append(value)

    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for anchor_time in unique_times:
        for item in _select_rows(
            rows,
            video_id=video_id,
            anchor_time=anchor_time,
            query_text=query_text,
            kind=kind,
            window=window,
            limit=limit,
            anchor_kf_n=anchor_kf_n,
            adjacent_kf_radius=adjacent_kf_radius,
            adjacent_kf_window=adjacent_kf_window,
        ):
            key = (
                item.get("source"),
                item.get("start_time"),
                item.get("end_time"),
                item.get("timestamp"),
                item.get("text"),
            )
            merged.setdefault(key, item)

    ordered = list(merged.values())
    ordered.sort(key=lambda item: (
        float(item.get("timestamp"))
        if item.get("timestamp") is not None else float("inf"),
        str(item.get("text", "")),
    ))
    selected = ordered[:max(0, int(limit))]
    for rank, item in enumerate(selected, 1):
        item["rank"] = rank
    return selected


def build_evidence_packet(candidate: Mapping[str, Any], *, asr_rows: Iterable[Any] | None = None,
                          ocr_rows: Iterable[Any] | None = None, query: str = "",
                          question: str = "", asr_window: float = 15.0,
                          ocr_window: float = 10.0, max_asr: int = 5,
                          max_ocr: int = 5, ocr_adjacent_kf_radius: int = 2,
                          ocr_adjacent_kf_window: float = 5.0,
                          modality_status: Mapping[str, Mapping[str, Any]] | None = None,
                          canonical_map: Mapping[tuple[str, int], tuple[int, float]] | None = None) -> dict[str, Any]:
    """Build a bounded packet anchored to one canonical candidate frame.

    The packet is intentionally JSON-compatible so it can be recorded in an
    answer trace.  ASR/OCR evidence may support the answer even if the anchor
    image does not visibly contain the answer (e.g. spoken weather data).
    """
    video_id = str(candidate.get("video_id", ""))
    if not video_id:
        raise ValueError("evidence candidate requires video_id")
    frame_idx = candidate.get("frame_idx", candidate.get("frame_id"))
    kf_n = candidate.get("kf_n")
    pts_time = _number(candidate.get("pts_time"), 0.0)
    if frame_idx is None or kf_n is None:
        raise ValueError("evidence candidate requires canonical frame_idx and kf_n")
    if pts_time is None:
        pts_time = 0.0
    if canonical_map is not None:
        expected = canonical_map.get((video_id, int(kf_n)))
        if expected is None:
            expected = canonical_map.get((video_id.upper(), int(kf_n)))
        if expected is None:
            raise ValueError(f"candidate frame is outside canonical map: {(video_id, int(kf_n))}")
        expected_frame, expected_time = int(expected[0]), float(expected[1])
        if int(frame_idx) != expected_frame or abs(float(pts_time) - expected_time) > 1e-2:
            raise ValueError(
                "candidate canonical mapping mismatch: "
                f"{(video_id, int(kf_n))} -> {(frame_idx, pts_time)}, "
                f"expected {(expected_frame, expected_time)}"
            )
    query_text = f"{normalize_text(query)}\n{normalize_text(question)}".strip()
    asr_rows = _materialize_rows(asr_rows)
    ocr_rows = _materialize_rows(ocr_rows)
    asr_anchor_times = [float(pts_time)]
    ocr_anchor_times = [float(pts_time)]
    for evidence_frame in candidate.get("evidence_frames", ()) or ():
        if not isinstance(evidence_frame, Mapping):
            continue
        modality = str(
            evidence_frame.get("modality")
            or evidence_frame.get("source")
            or ""
        ).strip().lower()
        evidence_time = _number(evidence_frame.get("pts_time"))
        if evidence_time is None:
            continue
        relation = str(evidence_frame.get("relation", "")).strip().lower()
        if not relation and abs(float(evidence_time) - float(pts_time)) <= 20.0:
            # Backwards-compatible input seam for already-bounded legacy
            # candidates. New production allocators always annotate relation.
            relation = "bounded_temporal_neighbor"
        if relation not in {"same_frame", "bounded_temporal_neighbor"}:
            continue
        if modality == "asr":
            asr_anchor_times.append(evidence_time)
        elif modality == "ocr":
            ocr_anchor_times.append(evidence_time)
    asr = _select_rows_at_anchors(
        asr_rows, video_id=video_id, anchor_times=asr_anchor_times,
        query_text=query_text, kind="asr", window=float(asr_window), limit=max_asr,
    )
    ocr = _select_rows_at_anchors(
        ocr_rows, video_id=video_id, anchor_times=ocr_anchor_times,
        query_text=query_text, kind="ocr", window=float(ocr_window), limit=max_ocr,
        anchor_kf_n=int(kf_n), adjacent_kf_radius=int(ocr_adjacent_kf_radius),
        adjacent_kf_window=float(ocr_adjacent_kf_window),
    )
    frames = [{
        "video_id": video_id, "frame_idx": int(frame_idx), "kf_n": int(kf_n),
        "pts_time": float(pts_time), "frame_path": candidate.get("frame_path"),
        "role": "anchor",
    }]
    # Callers may provide already validated temporal frames.  Keep the helper
    # defensive: never allow another video or invalid canonical fields in the
    # packet.
    for item in candidate.get("evidence_frames", ()) or ():
        item_video = str(item.get("video_id", video_id))
        item_frame = item.get("frame_idx", item.get("frame_id"))
        item_kf = item.get("kf_n")
        if item_video != video_id or item_frame is None or item_kf is None:
            continue
        relation = str(item.get("relation", "")).strip().lower()
        item_time = _number(item.get("pts_time"), pts_time)
        if not relation and item_time is not None and abs(float(item_time) - float(pts_time)) <= 20.0:
            relation = "bounded_temporal_neighbor"
        if relation not in {"same_frame", "bounded_temporal_neighbor"}:
            continue
        if canonical_map is not None:
            expected = canonical_map.get((item_video, int(item_kf)))
            if expected is None:
                expected = canonical_map.get((item_video.upper(), int(item_kf)))
            if expected is None or int(item_frame) != int(expected[0]):
                raise ValueError(
                    "evidence frame is outside or mismatched with canonical map: "
                    f"{(item_video, int(item_kf))} -> {item_frame}"
                )
        frames.append({
            "video_id": video_id, "frame_idx": int(item_frame), "kf_n": int(item_kf),
            "pts_time": float(_number(item.get("pts_time"), pts_time) or pts_time),
            "frame_path": item.get("frame_path"), "role": item.get("role", "neighbor"),
            "relation": relation,
        })
    sources = ["visual"]
    if asr:
        sources.append("asr")
    if ocr:
        sources.append("ocr")
    timestamps = [{"source": "visual", "start_time": float(pts_time),
                   "end_time": float(pts_time), "frame_idx": int(frame_idx)}]
    timestamps.extend({"source": item["source"], "start_time": item["start_time"],
                       "end_time": item["end_time"], "text": item["text"]}
                      for item in [*asr, *ocr])

    def _classify_status(kind: str, raw_rows: list[Any] | None,
                         selected: list[dict[str, Any]]) -> dict[str, Any]:
        supplied = (modality_status or {}).get(kind, {})
        if supplied:
            result = {str(key): value for key, value in supplied.items()}
            result.setdefault("status", "unknown")
            result.setdefault("row_count", len(raw_rows or ()))
            result.setdefault("usable_row_count", len(selected))
            result.setdefault("matched_row_count", len(selected))
            return result
        if raw_rows is None:
            status = "index_missing"
            usable_count = 0
        else:
            text_values = [
                normalize_text(_row_value(row, "chunk", "text", "transcript", "ocr_text", default=""))
                for row in raw_rows
            ]
            usable_count = sum(bool(value) for value in text_values)
            if not raw_rows:
                status = "coverage_missing"
            elif not usable_count:
                status = "no_speech" if kind == "asr" else "no_text"
            elif not selected:
                status = "no_match"
            else:
                status = "matched"
        return {
            "status": status,
            "row_count": len(raw_rows or ()),
            "usable_row_count": int(usable_count),
            "matched_row_count": len(selected),
            "available": status not in {"index_missing", "coverage_missing", "no_speech", "no_text"},
        }

    asr_status = _classify_status("asr", asr_rows, asr)
    ocr_status = _classify_status("ocr", ocr_rows, ocr)
    return {
        "video_id": video_id,
        "anchor": {"frame_idx": int(frame_idx), "kf_n": int(kf_n),
                    "pts_time": float(pts_time)},
        "frames": frames,
        "asr_chunks": asr,
        "ocr_text": ocr,
        "timestamps": timestamps,
        "sources": sources,
        "has_spoken_evidence": bool(asr),
        "has_screen_text_evidence": bool(ocr),
        "modality_status": {"asr": asr_status, "ocr": ocr_status},
        "canonical_mapping": {
            "status": "validated" if canonical_map is not None else "candidate_supplied",
            "video_id": video_id, "kf_n": int(kf_n), "frame_idx": int(frame_idx),
        },
    }


def answer_is_submission_safe(answer: Any) -> bool:
    text = normalize_text(answer).casefold()
    if text in _NONANSWER:
        return False
    return bool(text) and not any(marker in text for marker in (
        "cannot determine", "cannot answer", "unable to answer",
        "insufficient evidence", "not enough information", "no information",
        "không đủ thông tin", "không thể trả lời", "không xác định",
    ))


def evidence_support_score(answer: Any, packet: Mapping[str, Any]) -> float:
    """Cheap auditable support feature; VLM confidence remains authoritative.

    Exact token overlap is useful for spoken numeric facts and OCR labels.  A
    non-empty text evidence packet receives a small prior for paraphrases, but
    this function never manufactures an answer or rejects visual-only answers.
    """
    if not answer_is_submission_safe(answer):
        return 0.0
    answer_tokens = _tokens(answer)
    text_rows = [*packet.get("asr_chunks", []), *packet.get("ocr_text", [])]
    if not text_rows:
        return 0.0
    evidence_tokens = set().union(*(_tokens(row.get("text", "")) for row in text_rows))
    if answer_tokens and answer_tokens.intersection(evidence_tokens):
        return 1.0
    return 0.25


def render_evidence_prompt(packet: Mapping[str, Any]) -> str:
    """Render bounded, timestamped evidence without exposing internal JSON."""
    lines = [
        "Evidence policy: the answer may be supported by speech or screen text even when it is not visible in the anchor frame.",
        "Use only the evidence below. Do not claim that text is visible in the image unless the image shows it.",
    ]
    asr = packet.get("asr_chunks", [])
    if asr:
        lines.append("ASR transcript evidence:")
        for item in asr:
            start = item.get("start_time")
            end = item.get("end_time")
            start_text = "?" if start is None else f"{float(start):.2f}"
            end_text = "?" if end is None else f"{float(end):.2f}"
            lines.append(f"- [{start_text}s-{end_text}s] {item.get('text', '')}")
    else:
        lines.append("ASR transcript evidence: (none)")
    ocr = packet.get("ocr_text", [])
    if ocr:
        lines.append("OCR screen-text evidence:")
        for item in ocr:
            point = item.get("timestamp")
            point_text = "?" if point is None else f"{float(point):.2f}"
            lines.append(f"- [{point_text}s] {item.get('text', '')}")
    else:
        lines.append("OCR screen-text evidence: (none)")
    claim_evidence = packet.get("claim_evidence", [])
    if claim_evidence:
        lines.append("Query-constraint evidence (identifies the same entity/condition; not answer evidence by itself):")
        for item in claim_evidence:
            if not isinstance(item, Mapping):
                continue
            source = str(item.get("source", "text")).upper()
            claim = item.get("claim", {})
            claim_text = claim.get("text", "") if isinstance(claim, Mapping) else ""
            lines.append(f"- [{source}] claim={claim_text!s}: {item.get('text', '')}")
    lines.append("Anchor timestamp: {:.2f}s; anchor frame is the output grounding frame.".format(
        float(packet.get("anchor", {}).get("pts_time", 0.0))))
    return "\n".join(lines)
