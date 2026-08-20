"""
Encode SigLIP2-B/16 cho TOÀN BỘ keyframe (L-series mặc định).
- Checkpoint THEO PACK: pack xong lưu ngay -> resume được nếu gián đoạn.
- tqdm progress bar (tổng + từng pack).
- Batch lớn + đa luồng đọc ảnh (giảm bottleneck I/O).
- Output: data/index/siglip_full/<pack>.npy + <pack>_idmap.parquet
  Sau đó gộp bằng --merge.

Usage:
  # encode L-series (mặc định)
  python encode_siglip_full.py
  # encode K-series
  python encode_siglip_full.py --series K
  # gộp tất cả pack đã encode thành 1 file
  python encode_siglip_full.py --merge
  # tùy chỉnh
  python encode_siglip_full.py --batch 128 --workers 8
"""
import os, sys, glob, time, argparse, threading, queue
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from paths import INDEX_DIR, KEYFRAMES_DIR

IDX = str(INDEX_DIR)
KF  = str(KEYFRAMES_DIR)
OUT = os.path.join(IDX, "siglip_full")
os.makedirs(OUT, exist_ok=True)

def kf_path(vid, n): return os.path.join(KF, vid, f"{int(n):03d}.jpg")

def encode_series(series="L", model_name="ViT-B-16-SigLIP2", pretrained="webli",
                  batch=128, workers=8, idmap_file=None, out_dir=None):
    global OUT
    if out_dir:
        OUT = out_dir
        os.makedirs(OUT, exist_ok=True)
    import torch, open_clip
    from tqdm import tqdm
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev} | Model: {model_name} | batch={batch} workers={workers}")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(dev).eval()
    if dev == "cuda":
        model = model.half()  # fp16 cho nhanh + nhẹ VRAM

    # chọn id-map: L dùng clip_id_map, K dùng k_id_map (hoặc file chỉ định)
    if idmap_file is None:
        idmap_file = "k_id_map.parquet" if series == "K" else "clip_id_map.parquet"
    idmap = pd.read_parquet(os.path.join(IDX, idmap_file)).reset_index(drop=True)
    if "g" not in idmap.columns:
        idmap["g"] = idmap.index
    if "pack" not in idmap.columns:
        idmap["pack"] = idmap.video_id.str.split("_").str[0]
    # lọc đúng series
    idmap = idmap[idmap.pack.str.startswith(series)].copy()
    junk_fp = os.path.join(IDX, "junk_flags.parquet")
    if os.path.exists(junk_fp):
        junk = pd.read_parquet(junk_fp)
        junk_set = set(junk[junk.is_junk].g.values)
        before = len(idmap)
        idmap = idmap[~idmap.g.isin(junk_set)]
        print(f"Loại {before-len(idmap)} junk frame (black/blank)")

    packs = sorted(idmap.pack.unique())
    print(f"Series {series}: {len(packs)} packs, {len(idmap):,} keyframes")

    for pack in packs:
        outfp = os.path.join(OUT, f"{pack}.npy")
        if os.path.exists(outfp):
            print(f"SKIP {pack} (done)")
            continue
        gp = idmap[idmap.pack == pack].reset_index(drop=True)
        n = len(gp)
        t0 = time.time()

        # đa luồng đọc + preprocess ảnh -> queue
        q = queue.Queue(maxsize=batch*4)
        def loader(rows):
            for r in rows:
                p = kf_path(r.video_id, r.kf_n)
                try:
                    img = Image.open(p).convert("RGB")
                    q.put((r.g, preprocess(img)))
                except Exception:
                    q.put((r.g, None))
            q.put(None)  # sentinel
        rows = list(gp.itertuples())
        # chia rows cho workers
        threads = []
        chunks = [rows[i::workers] for i in range(workers)]
        for ch in chunks:
            t = threading.Thread(target=loader, args=(ch,), daemon=True)
            t.start(); threads.append(t)

        results = {}
        sentinels = 0
        buf_g, buf_t = [], []
        pbar = tqdm(total=n, desc=f"{pack}", unit="img")

        def flush():
            if not buf_t: return
            x = torch.stack(buf_t).to(dev)
            if dev == "cuda": x = x.half()
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            f = f.float().cpu().numpy().astype(np.float16)
            for gg, vec in zip(buf_g, f):
                results[gg] = vec
            pbar.update(len(buf_t))
            buf_g.clear(); buf_t.clear()

        while sentinels < workers:
            item = q.get()
            if item is None:
                sentinels += 1; continue
            g, t = item
            if t is None:
                results[g] = None; pbar.update(1); continue
            buf_g.append(g); buf_t.append(t)
            if len(buf_t) >= batch:
                flush()
        flush()
        pbar.close()

        valid = [(g, v) for g, v in results.items() if v is not None]
        valid.sort(key=lambda x: x[0])
        feats = np.stack([v for _, v in valid]).astype(np.float16)
        gs = np.array([g for g, _ in valid])
        np.save(outfp, feats)
        pd.DataFrame({"g": gs}).merge(
            idmap[["g","video_id","kf_n","frame_idx","pts_time"]], on="g"
        ).to_parquet(os.path.join(OUT, f"{pack}_idmap.parquet"))
        rate = n/(time.time()-t0)
        print(f"DONE {pack}: {len(valid)}/{n} kf | {rate:.1f} img/s | {time.time()-t0:.0f}s")
    print("ALL PACKS DONE for series", series)

def merge():
    feats=[]; maps=[]
    for fp in sorted(glob.glob(os.path.join(OUT, "*.npy"))):
        pack = os.path.basename(fp)[:-4]
        feats.append(np.load(fp))
        maps.append(pd.read_parquet(os.path.join(OUT, f"{pack}_idmap.parquet")))
    F = np.concatenate(feats); M = pd.concat(maps, ignore_index=True)
    np.save(os.path.join(IDX, "siglip_features.npy"), F)
    M.to_parquet(os.path.join(IDX, "siglip_id_map.parquet"))
    print(f"Merged: {F.shape} -> siglip_features.npy + siglip_id_map.parquet")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", default="L", help="L hoặc K")
    ap.add_argument("--model", default="ViT-B-16-SigLIP2")
    ap.add_argument("--pretrained", default="webli")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--idmap", default=None, help="id-map parquet (mặc định theo series)")
    ap.add_argument("--out", default=None, help="output dir (mặc định siglip_full)")
    a = ap.parse_args()
    if a.merge: merge()
    else: encode_series(a.series, a.model, a.pretrained, a.batch, a.workers, a.idmap, a.out)
