"""Experimental OCR specialist routing for candidate generation."""
from __future__ import annotations

import re


_OCR_TERMS = (
    "chữ", "viết", "ghi", "dòng chữ", "biển hiệu", "logo", "tên chương trình",
    "tiêu đề", "phụ đề", "subtitles", "subtitle", "text", "written", "sign",
    "caption", "ticker", "banner", "headline", "what does it say",
)


def ocr_route(query: str) -> bool:
    """Return whether the query explicitly asks for visible screen text."""
    text = re.sub(r"\s+", " ", str(query).lower()).strip()
    return any(term in text for term in _OCR_TERMS)


class OCRSpecialist:
    """Policy-only specialist; candidate retrieval remains outside this class."""

    def __init__(self, budget: int = 100):
        self.budget = int(budget)

    def enabled(self, query: str) -> bool:
        return ocr_route(query)

    def policy(self, query: str) -> dict:
        return {"enabled": self.enabled(query), "budget": self.budget,
                "reason": "screen-text signal" if self.enabled(query) else "visual-default"}
