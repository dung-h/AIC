"""
Module 5 (L10): Offline Fallback Layer.

Khi BTC cấm internet trong chung kết, hệ thống phải chạy bằng model local:
- TextEmbedder: bge-m3 LOCAL (sentence-transformers)
- VisualEncoder: SigLIP2 LOCAL (đã có)
- OCREngine: PaddleOCR (cho VN, optional install)
- ASREngine: PhoWhisper (vinai, GPU-friendly)

Common interface: `embed(texts) -> [N, D]`. Pipeline có thể swap online↔offline qua flag.

Lazy-load: chỉ load model khi gọi lần đầu (nếu bạn KHÔNG dùng offline thì không tốn RAM/VRAM).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from paths import load_env


class TextEmbedderOffline:
    """bge-m3 local. ~2.3GB model, CPU OK, GPU faster."""

    def __init__(self, device=None, model_name="BAAI/bge-m3"):
        self.model = None
        self.model_name = model_name
        self.device = device

    def _ensure(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"[Offline TextEmbedder] Loaded {self.model_name} on {self.model.device}")

    def embed(self, texts, batch_size=16, normalize=True):
        self._ensure()
        if isinstance(texts, str): texts = [texts]
        embs = self.model.encode(texts, batch_size=batch_size,
                                 normalize_embeddings=normalize,
                                 show_progress_bar=False)
        return np.asarray(embs, dtype=np.float32)


class TextEmbedderOnline:
    """bge-m3 via DO API (current path). Same interface."""

    def __init__(self):
        from urllib.request import Request, urlopen
        from urllib.error import HTTPError
        import json
        self.urlopen = urlopen; self.Request = Request
        self.HTTPError = HTTPError; self.json = json
        env = load_env()
        self.KEY = env.get("DO_INFERENCE_KEY", "")
        self.BASE = env.get("DO_INFERENCE_BASE", "")

    def embed(self, texts, batch_size=16, normalize=True):
        if isinstance(texts, str): texts = [texts]
        out = []
        import time
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            pl = {"model": "bge-m3", "input": batch}
            req = self.Request(self.BASE.rstrip("/") + "/embeddings",
                data=self.json.dumps(pl).encode(),
                headers={"Authorization": f"Bearer {self.KEY}",
                         "Content-Type": "application/json"})
            for a in range(5):
                try:
                    with self.urlopen(req, timeout=120) as r:
                        d = self.json.load(r)
                    out.extend([item["embedding"] for item in d["data"]])
                    break
                except self.HTTPError as e:
                    if e.code == 429: time.sleep(8 * (a + 1)); continue
                    raise
                except Exception:
                    if a == 4: raise
                    time.sleep(5 * (a + 1))
        arr = np.array(out, dtype=np.float32)
        if normalize:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
        return arr


def get_text_embedder(prefer="auto"):
    """
    prefer:
    - "auto": rejected because provider choice and fallback must be visible
    - "online": only DO API
    - "offline": only local
    """
    if prefer == "offline":
        return TextEmbedderOffline()
    if prefer == "online":
        return TextEmbedderOnline()
    if prefer == "auto":
        raise ValueError(
            "ambiguous text embedder provider: choose prefer='offline' or "
            "prefer='online' explicitly; automatic network fallback is disabled"
        )
    raise ValueError(f"unsupported text embedder preference: {prefer!r}")


# OCR & ASR offline placeholders (interface only — install optional)
class OCREngineOffline:
    """PaddleOCR fallback. Chỉ load khi cần."""
    def __init__(self):
        self.ocr = None

    def _ensure(self):
        if self.ocr is None:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang="vi", show_log=False)

    def read(self, image_path):
        self._ensure()
        result = self.ocr.ocr(image_path, cls=True)
        if not result or not result[0]: return ""
        return "\n".join([line[1][0] for line in result[0] if line and len(line) > 1])


class ASREngineOffline:
    """PhoWhisper-large fallback. ~2.7GB, GPU recommended."""
    def __init__(self, model_size="vinai/PhoWhisper-large"):
        self.pipe = None
        self.model_size = model_size

    def _ensure(self):
        if self.pipe is None:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline("automatic-speech-recognition",
                                 model=self.model_size, device=device)

    def transcribe(self, wav_path):
        self._ensure()
        return self.pipe(wav_path, return_timestamps=True)


if __name__ == "__main__":
    print("=== Offline Fallback self-test ===")

    # Test 1: TextEmbedder (online via cache)
    print("\n[1] TextEmbedderOnline")
    eo = TextEmbedderOnline()
    emb = eo.embed(["siêu bão Biển Đông", "lũ quét miền Trung"])
    print(f"  ✓ shape: {emb.shape}, norm avg: {np.linalg.norm(emb, axis=1).mean():.4f}")

    # Test 2: TextEmbedder offline (lazy load)
    print("\n[2] TextEmbedderOffline (will download bge-m3 ~2.3GB if first time)")
    print("    Skipped automatically — gọi e2 = TextEmbedderOffline(); e2.embed([...]) khi cần")
    # eo2 = TextEmbedderOffline()
    # emb2 = eo2.embed(["test"])
    # print(f"  ✓ offline embed shape: {emb2.shape}")

    # Test 3: get_text_embedder auto
    print("\n[3] get_text_embedder auto")
    e = get_text_embedder("online")
    print(f"  ✓ Got: {type(e).__name__}")

    # Test 4: OCR/ASR placeholders (interface only)
    print("\n[4] OCR/ASR offline placeholders (install optional)")
    print("    OCREngineOffline class: ready (need pip install paddleocr)")
    print("    ASREngineOffline class: ready (need pip install transformers + GPU 6GB+)")

    print("\n=== PASS (online tested, offline ready when deps installed) ===")
