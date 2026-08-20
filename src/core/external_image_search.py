"""
QUEST-style External Image Search (Module L10).

Cho out-of-knowledge (OOK) queries — entity lạ mà SigLIP/text embed không biết
(vd: nhân vật mới, sự kiện thời sự, sản phẩm trend như "Labubu").

Flow (theo đội AIO_Owlgorithms 2025):
  1. Query có entity lạ → text retrieval confidence thấp
  2. Search ảnh ngoài cho entity (DuckDuckGo Images / OpenSERP)
  3. Encode ảnh tham chiếu bằng SigLIP2 → image embedding
  4. Image-to-image search trong global index (dùng VKIS pipeline đã có)

Backend pluggable:
  - "ddg": duckduckgo_search (free, no key) — default
  - "openserp": self-host Docker (offline-capable cho chung kết)

Cache aggressive: query → reference images bytes.
"""
import os, sys, io, time, hashlib, tempfile, urllib.request
from urllib.parse import quote
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from cache import get_cache

try:
    from paths import CACHE_DIR
    IMG_CACHE_DIR = os.path.join(str(CACHE_DIR), "ext_images")
except Exception:
    IMG_CACHE_DIR = os.path.join(ROOT, "..", "..", "data", "cache", "ext_images")
IMG_CACHE_DIR = os.path.abspath(IMG_CACHE_DIR)
os.makedirs(IMG_CACHE_DIR, exist_ok=True)
QUEST_CACHE = get_cache("quest_urls", version="v1")


def search_images_ddg(query, max_results=5):
    """DuckDuckGo image search → list of image URLs. Free, no key.
    Thử cả 2 package: ddgs (mới) và duckduckgo_search (cũ)."""
    DDGS = None
    try:
        from ddgs import DDGS  # newer package name
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("[QUEST] no ddg package"); return []
    urls = []
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                # ddgs new API: .images(query, ...) ; older: same
                for r in ddgs.images(query, max_results=max_results):
                    u = r.get("image") or r.get("thumbnail") or r.get("url")
                    if u: urls.append(u)
            if urls: break
        except Exception as e:
            if attempt < 2:
                time.sleep(5 * (attempt + 1)); continue
            print(f"[QUEST] DDG search err: {str(e)[:100]}")
    return urls[:max_results]


def search_images_openserp(query, max_results=5, base="http://localhost:7000"):
    """OpenSERP self-host. Offline-capable cho chung kết (Docker)."""
    import json
    url = f"{base}/google/image?text={quote(query)}&limit={max_results}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        return [item.get("image_url") or item.get("url") for item in data][:max_results]
    except Exception as e:
        print(f"[QUEST] OpenSERP err: {e}")
        return []


def download_image(url, timeout=15):
    """Download 1 image → PIL Image. Cache by URL hash."""
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    fp = os.path.join(IMG_CACHE_DIR, f"{h}.jpg")
    if os.path.exists(fp):
        try:
            from PIL import Image
            return Image.open(fp).convert("RGB")
        except Exception:
            pass
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.save(fp, "JPEG")
        return img
    except Exception as e:
        return None


class ExternalImageSearch:
    """QUEST: query → reference images → image embeddings (SigLIP2)."""

    def __init__(self, vkis_pipeline=None, backend="ddg"):
        """vkis_pipeline: VKISPipeline instance (cho encode_image + search).
        Nếu None, lazy-load.
        """
        self.backend = backend
        self._vkis = vkis_pipeline

    def _ensure_vkis(self):
        if self._vkis is None:
            sys.path.insert(0, os.path.join(ROOT, "..", "pipelines"))
            from vkis_pipeline import VKISPipeline
            self._vkis = VKISPipeline()
        return self._vkis

    def get_reference_images(self, query, max_images=5):
        """Search + download reference images cho entity."""
        cached_urls = QUEST_CACHE.get("urls", query, self.backend)
        if cached_urls is None:
            if self.backend == "openserp":
                cached_urls = search_images_openserp(query, max_results=max_images)
            else:
                cached_urls = search_images_ddg(query, max_results=max_images)
            QUEST_CACHE.set(cached_urls, "urls", query, self.backend)
        imgs = []
        for u in cached_urls:
            img = download_image(u)
            if img is not None:
                imgs.append((u, img))
        return imgs

    def search_by_entity(self, query, topk=10, max_ref_images=5, agg="mean"):
        """
        OOK pipeline: query (entity lạ) → external images → image-to-image search.
        agg: cách pool nhiều reference image embeddings.
        Returns: list of (video_id, frame_idx, pts_time, score).
        """
        vkis = self._ensure_vkis()
        refs = self.get_reference_images(query, max_images=max_ref_images)
        if not refs:
            return [], {"error": "no reference images found"}

        # Encode reference images
        ref_embs = []
        for url, img in refs:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            try:
                img.save(tmp_path, "JPEG")
                ref_embs.append(vkis.encode_image(tmp_path))
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        if not ref_embs:
            return [], {"error": "encode failed"}
        ref_embs = np.array(ref_embs)

        # Aggregate query vector
        if agg == "mean":
            q = ref_embs.mean(0); q = q / np.linalg.norm(q)
            scores = vkis.vfeat @ q
        else:  # max over reference images
            sims = vkis.vfeat @ ref_embs.T
            scores = sims.max(axis=1)

        order = np.argsort(-scores)[:topk]
        results = []
        for j in order:
            row = vkis.vmap.iloc[j]
            results.append((row.video_id, int(row.frame_idx),
                            float(row.pts_time), float(scores[j])))
        return results, {"n_ref_images": len(ref_embs),
                         "ref_urls": [r[0] for r in refs]}


if __name__ == "__main__":
    print("=== External Image Search (QUEST) self-test ===\n")

    # Test 1: search URLs only (no download)
    print("[1] DDG image search for 'bão Kong-Rey'")
    urls = search_images_ddg("bão Kong-Rey Biển Đông", max_results=3)
    print(f"  Found {len(urls)} URLs")
    for u in urls[:3]: print(f"    {u[:80]}")

    if not urls:
        print("\n  (No URLs — DDG may be rate-limited or offline)")
        print("  Module structure OK, backend swap available (openserp for finals)")
    else:
        print("\n[2] Full entity search (download + encode + retrieve)")
        q = ExternalImageSearch(backend="ddg")
        results, meta = q.search_by_entity("bão Kong-Rey Biển Đông", topk=5)
        print(f"  meta: {meta.get('n_ref_images', 0)} ref images")
        for v, f, t, sc in results[:5]:
            print(f"    {v} fidx={f} t={t:.1f}s sc={sc:.3f}")
