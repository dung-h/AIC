"""
VKIS Pipeline (Video Known-Item Search): cho 1 clip ngắn (5-30s) hoặc 1 ảnh,
tìm video + keyframe gần nhất trong global SigLIP2 index.

Strategy L1:
- Input: video clip (mp4) hoặc 1 ảnh truy vấn (jpg)
- Encode SigLIP2 (image encoder local), pool nhiều frame
- Search top-K trong global_siglip.npy (cosine)
- Output: (video_id, frame_idx, pts_time) top-N

Tiết kiệm: dùng cùng encoder open_clip ViT-B-16-SigLIP2 đã load.
"""
import os
import numpy as np, pandas as pd
import sys
from pathlib import Path
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "utils"))
from paths import INDEX_DIR, KEYFRAMES_DIR
from src.pipelines.kis_fusion_retriever import _normalise_active_video_prefixes

IDX = str(INDEX_DIR)

class VKISPipeline:
    """VKIS: video clip / image query → top-K (video_id, frame_idx)."""

    def __init__(self, model_name="ViT-L-16-SigLIP2-256",
                 index_npy="global_siglip_vitl.npy",
                 keyframes_parquet="global_keyframes_vitl.parquet"):
        # Default: ViT-L index (unified with KIS production stack, Exp 093+).
        # Pass the ViT-B/16 pair for the legacy 768-dim index.
        self.vfeat = np.load(os.path.join(IDX, index_npy), mmap_mode="r")
        self.vmap = pd.read_parquet(os.path.join(IDX, keyframes_parquet)).reset_index(drop=True)
        self.active_video_prefixes = _normalise_active_video_prefixes(None)
        active_mask = np.ones(len(self.vmap), dtype=bool)
        if self.active_video_prefixes:
            active_mask = self.vmap["video_id"].astype(str).str.upper().str.startswith(
                self.active_video_prefixes
            ).to_numpy()
            if not bool(active_mask.any()):
                raise ValueError(
                    "active video prefixes matched no VKIS index rows: "
                    + ", ".join(self.active_video_prefixes)
                )
        self._active_indices = np.flatnonzero(active_mask)
        self.video_groups = [
            (str(video), group.sort_values("pts_time").index.to_numpy(dtype=np.int64))
            for video, group in self.vmap.groupby("video_id", sort=False)
            if bool(active_mask[group.index].any())
        ]
        self.video_group_ids = {video: ids for video, ids in self.video_groups}
        import torch, open_clip
        self.torch = torch
        self.m, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="webli")
        self.m = self.m.eval()
        if torch.cuda.is_available():
            self.m = self.m.cuda()
            self.device = "cuda"
        else:
            self.device = "cpu"
        print(f"[VKIS] Loaded {model_name} index: {self.vfeat.shape} on {self.device}")

    def encode_image(self, img_path):
        """Encode 1 ảnh thành SigLIP2 embedding (768-dim, normalized)."""
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            f = self.m.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy()[0].astype(np.float32)

    def encode_clip_frames(self, mp4_path, n_frames=8, fps_sample=None):
        """Encode N evenly-spaced frames của 1 video clip → [N, 768] embeddings."""
        try:
            import cv2
        except ImportError:
            print("opencv-python missing. pip install opencv-python")
            return None
        from PIL import Image
        cap = cv2.VideoCapture(mp4_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps_sample is not None:
            n_frames = max(1, int(total / fps * fps_sample))
        if not cap.isOpened() or total <= 0:
            cap.release()
            return self._encode_clip_frames_ffmpeg(mp4_path, n_frames)
        idxs = np.linspace(0, total - 1, n_frames).astype(int)
        embs = []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok: continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                f = self.m.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            embs.append(f.cpu().numpy()[0].astype(np.float32))
        cap.release()
        return np.array(embs)

    def _encode_clip_frames_ffmpeg(self, mp4_path, n_frames):
        """Decode clips when OpenCV lacks a usable H.264 backend."""
        import glob
        import subprocess
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            pattern = str(Path(tmp) / "%03d.jpg")
            ffmpeg = shutil.which("ffmpeg") or str(Path(__file__).resolve().parents[2] / ".venv/bin/ffmpeg")
            command = [ffmpeg, "-y", "-i", str(mp4_path), "-vf",
                       f"fps={max(1, n_frames / 20):.6f}",
                       "-frames:v", str(n_frames), "-q:v", "3", pattern]
            result = subprocess.run(command, capture_output=True, text=True)
            paths = sorted(glob.glob(str(Path(tmp) / "*.jpg")))
            if result.returncode != 0 or not paths:
                return np.empty((0, 0), dtype=np.float32)
            embs = []
            for path in paths[:n_frames]:
                x = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
                with self.torch.no_grad():
                    f = self.m.encode_image(x)
                    f = f / f.norm(dim=-1, keepdim=True)
                embs.append(f.cpu().numpy()[0].astype(np.float32))
            return np.asarray(embs)

    def search_image(self, img_path, topk=10):
        """Search 1 ảnh query."""
        q = self.encode_image(img_path)
        scores = self.vfeat @ q
        order = self._active_indices[np.argsort(-scores[self._active_indices])[:topk]]
        out = []
        for j in order:
            row = self.vmap.iloc[j]
            out.append((row.video_id, int(row.frame_idx), float(row.pts_time), float(scores[j])))
        return out

    def search_clip(self, mp4_path, n_frames=8, topk=10, agg="hybrid0.5"):
        """Search 1 clip with global or local-temporal score aggregation.

        ``hybrid0.7`` protects global top3 video evidence while smoothing only
        the top candidate videos; it is useful when timestamp precision matters.
        Default hybrid0.5: selected from the fixed holdout because it
        improves frame timestamp localization while retaining strong video
        recall. Use top3 for the historical video-recall baseline.
        - max: max cosine across clip frames
        - mean: pool clip embeddings rồi search
        - top3: average top-3 cosines (best balance)
        """
        embs = self.encode_clip_frames(mp4_path, n_frames=n_frames)
        if embs is None or len(embs) == 0:
            return []
        if agg == "mean":
            q = embs.mean(0); q = q / np.linalg.norm(q)
            scores = self.vfeat @ q
        else:
            sims = self.vfeat @ embs.T  # [N_kf, N_clip]
            if agg == "max":
                scores = sims.max(axis=1)
            elif agg == "top3":
                if sims.shape[1] >= 3:
                    scores = np.sort(sims, axis=1)[:, -3:].mean(axis=1)
                else:
                    scores = sims.mean(axis=1)
            elif agg in {"smooth3", "smooth5"}:
                # Smooth each video's keyframe evidence in time to reduce
                # isolated near-duplicate peaks from the sampled clip.
                radius = int(agg[-1]) // 2
                scores = sims.max(axis=1).copy()
                for _, ids in self.video_groups:
                    local = scores[ids]
                    padded = np.pad(local, (radius, radius), mode="edge")
                    scores[ids] = np.convolve(
                        padded, np.ones(2 * radius + 1) / (2 * radius + 1), mode="valid"
                    )
            elif agg.startswith("hybrid"):
                alpha = float(agg.removeprefix("hybrid"))
                top3 = np.sort(sims, axis=1)[:, -min(3, sims.shape[1]):].mean(axis=1)
                groups = self.video_groups
                prior = {video: float(top3[ids].max()) for video, ids in groups}
                selected = set(sorted(prior, key=prior.get, reverse=True)[:20])
                local = sims.max(axis=1).copy()
                for video, ids in groups:
                    if video not in selected:
                        continue
                    padded = np.pad(local[ids], (2, 2), mode="edge")
                    local[ids] = np.convolve(padded, np.ones(5) / 5, mode="valid")
                scores = alpha * top3 + (1.0 - alpha) * local
                scores[[str(v) not in selected for v in self.vmap.video_id]] = -np.inf
            else:
                scores = sims.mean(axis=1)
        order = self._active_indices[np.argsort(-scores[self._active_indices])[:topk]]
        out = []
        for j in order:
            row = self.vmap.iloc[j]
            out.append((row.video_id, int(row.frame_idx), float(row.pts_time), float(scores[j])))
        return out

if __name__ == "__main__":
    p = VKISPipeline()
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            res = p.search_image(path, topk=5)
        else:
            res = p.search_clip(path, topk=5)
        print(f"\nQuery: {path}")
        for k, (v, f, t, sc) in enumerate(res):
            print(f"  {k+1}. {v} frame_idx={f} t={t:.1f}s score={sc:.4f}")
    else:
        # Self-test: lấy 1 keyframe biết → search xem có ra đúng không
        kfp = os.path.join(str(KEYFRAMES_DIR), "K01_V001", "010.jpg")
        if os.path.exists(kfp):
            res = p.search_image(kfp, topk=3)
            print(f"\nSelf-test keyframe K01_V001/010.jpg:")
            for k, (v, f, t, sc) in enumerate(res):
                print(f"  {k+1}. {v} frame_idx={f} t={t:.1f}s score={sc:.4f}")
            ok = res[0][0] == "K01_V001"
            print(f"\n{'PASS' if ok else 'FAIL'}: top-1 video {'==' if ok else '!='} K01_V001")
        else:
            print("Provide a path to clip or image")
