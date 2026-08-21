"""Contract tests for bounded external-fact retrieval views."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from src.vqa.grounding import DuckDuckGoGroundingResolver, GroundingRequest


def test_grounding_request_prioritizes_exact_quote_then_entity_then_scene():
    request = GroundingRequest(
        "Phóng sự về FANA (Nắng Ấm Yêu Thương) tại Khánh Hòa.",
        'Câu thơ "Hỏa hồng Nhựt Tảo oanh thiên địa" được đọc là gì?',
    )

    views = request.search_queries()

    assert views[:3] == (
        "Hỏa hồng Nhựt Tảo oanh thiên địa",
        "FANA Nắng Ấm Yêu Thương",
        "Phóng sự về FANA (Nắng Ấm Yêu Thương) tại Khánh Hòa.",
    )
    assert len(views) <= 4


def test_grounding_request_uses_planner_fact_view_before_broad_scene():
    request = GroundingRequest(
        "Phóng sự về câu lạc bộ thiện nguyện FANA tại Khánh Hòa.",
        "Tên xã được nhắc đến là gì?",
        hypothesis_views=("câu lạc bộ thiện nguyện FANA",),
    )

    assert request.search_queries()[:2] == (
        "câu lạc bộ thiện nguyện FANA",
        "Phóng sự về câu lạc bộ thiện nguyện FANA tại Khánh Hòa.",
    )


def test_grounding_request_reserves_source_query_when_planner_fills_budget():
    request = GroundingRequest(
        "Phóng sự về Nguyễn Trung Trực tại Kiên Giang.",
        "Hai câu thơ được đọc là gì?",
        hypothesis_views=(
            "Nguyễn Trung Trực thơ Kiên Giang",
            "hai câu thơ Nguyễn Trung Trực",
            "Nguyễn Trung Trực documentary poetry",
            "Nguyễn Trung Trực phóng sự trích thơ",
        ),
    )

    views = request.search_queries()

    assert len(views) == 4
    assert request.query in views
    assert views[:3] == request.hypothesis_views[:3]


def test_ddg_continues_after_zero_hit_view_and_records_winning_search_view(monkeypatch):
    calls: list[str] = []

    class _FakeDDGS:
        def __init__(self, *, timeout):
            assert timeout == 3.0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def text(self, query, *, max_results):
            assert max_results == 20
            calls.append(query)
            if query == "FANA Nắng Ấm Yêu Thương":
                raise RuntimeError("No results found")
            if query == "Phóng sự về FANA (Nắng Ấm Yêu Thương) tại Khánh Hòa.":
                return [{
                    "href": "https://example.org/fana-giang-ly",
                    "title": "FANA đến xã Giang Ly",
                    "body": "Hoạt động tại Khánh Hòa.",
                }]
            return []

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=_FakeDDGS))
    resolver = DuckDuckGoGroundingResolver(
        allowed_domains=("example.org",), timeout_seconds=3.0,
    )

    evidence = resolver.resolve(GroundingRequest(
        "Phóng sự về FANA (Nắng Ấm Yêu Thương) tại Khánh Hòa.",
        "Tên xã được nhắc đến là gì?",
    ))

    assert tuple(calls[:2]) == (
        "FANA Nắng Ấm Yêu Thương",
        "Phóng sự về FANA (Nắng Ấm Yêu Thương) tại Khánh Hòa.",
    )
    assert len(evidence) == 1
    assert evidence[0].search_query == calls[1]
    assert evidence[0].source_url == "https://example.org/fana-giang-ly"
