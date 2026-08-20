"""Conservative query-to-evidence routing.

Routing selects candidate branches; it does not fuse their uncalibrated
scores. The visual branch is always retained as the production anchor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialistRoute:
    branches: tuple[str, ...]
    reasons: tuple[str, ...]


_OBJECT_TERMS = ("chảo", "xe đạp", "bicycle", "pizza", "cà chua", "chó", "mèo", "cá", "lân", "rồng")
_OCR_PATTERNS = (r"\b[A-ZĐ][a-zà-ỹ]+(?:\s+[A-ZĐ][a-zà-ỹ]+)+", r"\b[A-Z]{2,}\b", r"\d{4}", r"cấp\s*\d+")
_ASR_TERMS = ("nói", "phát biểu", "lời", "giọng", "audio", "nghe", "nhắc đến")


def route_specialists(query: str) -> SpecialistRoute:
    text = str(query).strip()
    lower = text.lower()
    branches = ["visual"]
    reasons = ["visual_anchor"]
    if any(term in lower for term in _OBJECT_TERMS):
        branches.append("object")
        reasons.append("object_mention")
    if any(re.search(pattern, text) for pattern in _OCR_PATTERNS):
        branches.append("ocr")
        reasons.append("screen_text_or_entity")
    if any(term in lower for term in _ASR_TERMS):
        branches.append("asr")
        reasons.append("spoken_content")
    return SpecialistRoute(tuple(branches), tuple(reasons))
