"""
Module 2 (L10): Query Rewriter — đa ngôn ngữ + multi-style query expansion.

Tạo N variants của query để feed nhiều signal:
- VN original (cho ASR/OCR Vietnamese matching)
- EN translated (cho SigLIP visual matching)
- VN expanded with synonyms (cho BM25 OCR)
- "Visual-style" caption (cho SigLIP visual)

Cache mạnh để không gọi LLM lặp.
"""
import os, sys, json, urllib.request, urllib.error, time, re
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "..", "utils"))
from cache import get_cache
from src.core.providers import provider_for

QUERY_CACHE = get_cache("query_rewrites", version="v1")


def _llm(messages, model=None, max_tokens=200, temperature=0.3):
    """Call the explicitly configured text provider.

    Rewriting is an opt-in remote enhancement.  It must not borrow the VLM
    endpoint or guess a DigitalOcean default when no text provider exists.
    """
    provider = provider_for("text")
    if not provider.configured:
        raise RuntimeError(
            "text rewriting requires TEXT_BASE_URL, TEXT_API_KEY and TEXT_MODEL"
        )
    pl = {"model": model or provider.model, "messages": messages,
          "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(provider.base_url + "/chat/completions",
        data=json.dumps(pl).encode(),
        headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"})
    for a in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            return d["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(8 * (a + 1)); continue
            raise
        except Exception:
            if a == 4: raise
            time.sleep(3 * (a + 1))


def detect_lang(q):
    """Heuristic VN vs EN detect dựa vào tỉ lệ ký tự ASCII và dấu tiếng Việt."""
    if not q: return "unknown"
    has_diacritics = bool(re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
                                      q.lower()))
    if has_diacritics:
        return "vi"
    # Tiếng Việt không dấu vẫn nhiều âm tiết ngắn
    words = q.split()
    if len(words) >= 3 and all(len(w) <= 4 for w in words[:5]):
        # Có thể là VN không dấu (ngắn từ)
        ascii_only = all(ord(c) < 128 for w in words for c in w)
        if ascii_only:
            return "vi_no_diacritic"
    return "en"


class QueryRewriter:
    """Rewriter đa-style. Cache cứng mỗi rewrite."""

    @QUERY_CACHE.cached
    def to_vn(self, q):
        """Translate to natural Vietnamese."""
        lang = detect_lang(q)
        if lang == "vi": return q
        result = _llm([
            {"role": "system", "content": "Translate to natural Vietnamese as a search query. "
                                          "Output only the Vietnamese, no extra text."},
            {"role": "user", "content": q},
        ], max_tokens=80)
        return result.strip()

    @QUERY_CACHE.cached
    def to_en(self, q):
        """Translate to natural English caption-style."""
        lang = detect_lang(q)
        if lang == "en": return q
        result = _llm([
            {"role": "system", "content": "Translate the Vietnamese query to a natural English "
                                          "caption-style description (1 sentence, 10-20 words). "
                                          "Output only English, no extra text."},
            {"role": "user", "content": q},
        ], max_tokens=80)
        return result.strip()

    @QUERY_CACHE.cached
    def visual_caption(self, q):
        """Convert query to visual caption (what would the frame look like)."""
        result = _llm([
            {"role": "system", "content": ("Imagine you are looking at the frame matching this query. "
                                            "Describe what you SEE in 1 English sentence (visual scene, "
                                            "objects, action, colors). 10-20 words. No 'a video' / "
                                            "'a frame' framing. Just the visual description.")},
            {"role": "user", "content": q},
        ], max_tokens=80)
        return result.strip()

    @QUERY_CACHE.cached
    def expand_vn(self, q):
        """Expand VN query with synonyms cho BM25 OCR matching."""
        result = _llm([
            {"role": "system", "content": ("Mở rộng truy vấn tiếng Việt với từ đồng nghĩa và biến thể "
                                            "để tìm kiếm tốt hơn. Giữ ý gốc. Output chỉ chuỗi mở rộng, "
                                            "không giải thích.")},
            {"role": "user", "content": q},
        ], max_tokens=120)
        return result.strip()

    def rewrite_all(self, q, with_visual=True, with_expand=False):
        """Trả về dict các variants."""
        out = {"original": q, "lang": detect_lang(q)}
        try: out["vn"] = self.to_vn(q)
        except Exception as e: out["vn"] = q; out["err_vn"] = str(e)[:100]
        try: out["en"] = self.to_en(q)
        except Exception as e: out["en"] = q; out["err_en"] = str(e)[:100]
        if with_visual:
            try: out["visual_caption"] = self.visual_caption(q)
            except Exception as e: out["visual_caption"] = q; out["err_vc"] = str(e)[:100]
        if with_expand:
            try: out["vn_expanded"] = self.expand_vn(q)
            except Exception as e: out["vn_expanded"] = q
        return out


if __name__ == "__main__":
    print("=== QueryRewriter self-test ===")
    qr = QueryRewriter()
    tests = [
        "siêu bão Biển Đông cấp 16",
        "People eating traditional German food",
        "lũ quét Nghệ An nhà sập",
    ]
    for q in tests:
        print(f"\n[{q}]")
        out = qr.rewrite_all(q, with_visual=True, with_expand=False)
        for k, v in out.items():
            if k != "original":
                print(f"  {k}: {v}")
    print("\n=== PASS ===")
