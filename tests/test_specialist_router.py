from src.core.specialist_router import route_specialists


def test_visual_is_always_anchor():
    route = route_specialists("một người đi bộ trong công viên")
    assert route.branches == ("visual",)


def test_routes_explicit_evidence_without_replacing_visual():
    route = route_specialists("HTV phát biểu về bão cấp 16")
    assert route.branches[0] == "visual"
    assert "ocr" in route.branches
    assert "asr" in route.branches
    assert "screen_text_or_entity" in route.reasons
