"""
KIS Retriever — TASK-AWARE, visual-first (kiến trúc đúng sau Exp 101-107).

Khác router cũ: KHÔNG đoán query-type. KIS = visual description → visual-first.
Đòn bẩy A/B-proven trên VBS unbiased set (n=90):
  - ViT-L-16-SigLIP2-256 visual (Exp 093)
  - VN→EN translate query (Exp 104, +8pp)
  - CSLS text-anchor rerank (Exp 106, +3pp)
  - [optional] VLM rerank top-K (Exp 107, +6.7pp partial) — bật khi API ổn

Tiến trình R@1: router cũ 0.056 → 0.478 (visual+EN+CSLS) → ~0.55 (kỳ vọng +VLM).

API-free mode (translate off) vẫn chạy: dùng VN trực tiếp (0.367) khi offline/no-API.
"""
import os, sys, json, urllib.request, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "router"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
from paths import INDEX_DIR, KEYFRAMES_DIR, load_env
from src.utils.open_clip_local import get_tokenizer as get_local_tokenizer


IDX = str(INDEX_DIR)
KF_ROOT = str(KEYFRAMES_DIR)


class KISRetriever:
    def __init__(self, translate=True, use_csls=False, csls_beta=1.5, translate_cache=None):
        # CSLS default OFF: Exp 108 cho thấy không ổn định (VBS +2pp nhưng visual_large -4pp = noise).
        # EN-translate là lever đáng tin duy nhất. csls có thể bật để thử nghiệm.
        import torch, open_clip
        self.torch = torch
        self.translate_on = translate
        self.use_csls = use_csls
        self.csls_beta = csls_beta
        self.env = load_env()
        # Pre-translated cache: {vn_text: en_text} — avoids per-query API call
        self._translate_cache: dict = translate_cache or {}

        self.F = np.load(os.path.join(IDX, "global_siglip_vitl.npy")).astype(np.float32)
        self.km = pd.read_parquet(os.path.join(IDX, "global_keyframes_vitl.parquet"))
        self.vid_arr = self.km.video_id.values
        self.all_vids = sorted(set(self.vid_arr))
        self._vidx = {v: i for i, v in enumerate(self.all_vids)}

        self.m, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-16-SigLIP2-256", pretrained="webli")
        self.m = self.m.eval()
        self.tk = get_local_tokenizer(open_clip, "ViT-L-16-SigLIP2-256")
        self._rf = None  # CSLS frame-hubness term (lazy, needs anchor queries)
        print(f"[KIS] ViT-L index {self.F.shape}, {len(self.all_vids)} videos, "
              f"translate={translate}, csls={use_csls}")

    def sig_text(self, q):
        with self.torch.no_grad():
            f = self.m.encode_text(self.tk([q]))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()[0].astype(np.float32)

    def translate(self, text):
        if not self.translate_on:
            return text
        if text in self._translate_cache:
            return self._translate_cache[text]
        key = self.env.get("DO_INFERENCE_KEY"); base = self.env.get("DO_INFERENCE_BASE")
        if not key:
            return text
        pl = {"model": "llama-4-maverick",
              "messages": [{"role": "user", "content":
                  "Translate this Vietnamese visual scene description to a concise English "
                  "image caption (keep all visual details). Output ONLY the caption:\n\n" + text}],
              "max_tokens": 150, "temperature": 0.0}
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
            data=json.dumps(pl).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["choices"][0]["message"]["content"].strip()
        except Exception:
            return text  # graceful fallback to VN

    def set_csls_anchors(self, anchor_texts):
        """Precompute frame-hubness r_f vs a set of anchor query embeds (improves CSLS)."""
        Q = np.stack([self.sig_text(t) for t in anchor_texts])
        k = min(10, len(Q))
        self._rf = np.sort(self.F @ Q.T, axis=1)[:, -k:].mean(axis=1)

    def _maxvec(self, frame_sc):
        out = np.full(len(self.all_vids), -9.0, np.float32)
        np.maximum.at(out, [self._vidx[v] for v in self.vid_arr], frame_sc)
        return out

    def search(self, query_vn, topk=20):
        """KIS search: VN desc → (translate) → ViT-L visual (+CSLS) → top-K videos+frames."""
        q = self.translate(query_vn)
        qv = self.sig_text(q)
        frame_sc = self.F @ qv
        if self.use_csls and self._rf is not None:
            frame_sc = 2 * frame_sc - self.csls_beta * self._rf
        vid_sc = self._maxvec(frame_sc)
        order = np.argsort(-vid_sc)[:topk]
        results = []
        for j in order:
            v = self.all_vids[j]
            sub = self.km[self.km.video_id == v]
            gidx = sub.g.values
            best = sub.iloc[int(np.argmax(frame_sc[gidx]))]
            results.append((v, int(best.frame_idx), int(best.kf_n), float(vid_sc[j])))
        return results

    def vlm_score(self, desc_vn, img_path, model="llama-4-maverick"):
        """Chấm điểm khớp (desc, keyframe) bằng VLM. maverick: nhanh + đọc ảnh VN tốt
        (gemma-4-31B throttle nặng nên dùng maverick). Exp 112."""
        import base64, re
        key = self.env.get("DO_INFERENCE_KEY"); base = self.env.get("DO_INFERENCE_BASE")
        if not key:
            return 0.0
        try:
            b64 = base64.b64encode(open(img_path, "rb").read()).decode()
        except Exception:
            return 0.0
        prompt = (f"Mô tả cần tìm: \"{desc_vn}\"\n\nKhung hình này khớp mức nào? "
                  "Chỉ 1 số 0-10 (10=khớp hoàn hảo). CHỈ số.")
        pl = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            "max_tokens": 10, "temperature": 0.0}
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
            data=json.dumps(pl).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as rr:
                    c = json.load(rr)["choices"][0]["message"]["content"].strip()
                m = re.search(r"\b(10|[0-9])(\.\d+)?\b", c)
                return float(m.group(0)) if m else 0.0
            except Exception:
                time.sleep(2)
        return 0.0

    def search_rerank(self, query_vn, topk=20, rerank_k=5):
        """KIS + VLM rerank top-K (Exp 112: +6.6pp). Cần API (maverick).
        Rerank rerank_k candidate đầu bằng VLM, giữ phần đuôi."""
        from concurrent.futures import ThreadPoolExecutor
        res = self.search(query_vn, topk=topk)
        if not res:
            return res
        head = res[:rerank_k]
        imgs = [os.path.join(KF_ROOT,
                             v, f"{kf:03d}.jpg") for v, fidx, kf, sc in head]
        with ThreadPoolExecutor(max_workers=5) as ex:
            vsc = list(ex.map(lambda im: self.vlm_score(query_vn, im), imgs))
        order = sorted(range(len(head)), key=lambda i: (-vsc[i], i))
        reranked = [head[i] for i in order]
        return reranked + res[rerank_k:]


if __name__ == "__main__":
    r = KISRetriever(translate=False)
    res = r.search("Một nữ MC mặc áo blazer nâu đứng trong studio", topk=5)
    for v, fidx, kf, sc in res:
        print(f"  {v} frame={fidx} kf={kf} score={sc:.3f}")
