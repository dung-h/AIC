"""
Encode toàn bộ 379k keyframes với bất kỳ open_clip model nào.

- Idmap: global_keyframes_vitl.parquet (379765 rows, L+K series)
- Checkpoint theo pack — resume-safe nếu bị gián đoạn
- Output: data/index/<out_name>/<pack>.npy + <pack>_idmap.parquet
  Sau đó merge bằng --merge thành 1 file index + 1 file kmap

Usage:
  # Full encode SO400M-384 (recommended winner từ A/B Exp123)
  python src/encode/encode_full_index.py --model ViT-SO400M-16-SigLIP2-384 --out-name so400m384

  # Merge sau khi encode xong
  python src/encode/encode_full_index.py --merge --out-name so400m384

  # Tùy chỉnh
  python src/encode/encode_full_index.py --model ViT-SO400M-16-SigLIP2-384 \\
      --out-name so400m384 --batch 128 --workers 8
"""
import os, sys, glob, time, argparse, threading, queue, hashlib, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
from paths import INDEX_DIR, KEYFRAMES_DIR

IDX = str(INDEX_DIR)
KF  = str(KEYFRAMES_DIR)


def _runtime_artifact_names(out_name):
    """Return the stable runtime filenames for a visual encoder variant."""
    aliases = {
        "vitl": ("global_siglip_vitl.npy", "global_keyframes_vitl.parquet"),
        "so400m384": ("global_so400m384.npy", "global_keyframes_so400m384.parquet"),
    }
    return aliases.get(out_name, (f"global_{out_name}.npy", f"global_keyframes_{out_name}.parquet"))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npy(path, array):
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _atomic_parquet(path, frame):
    temporary = f"{path}.tmp"
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path, payload):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def kf_path(vid, n):
    return os.path.join(KF, vid, f"{int(n):03d}.jpg")


def encode_all(model_name="ViT-SO400M-16-SigLIP2-384", pretrained="webli",
               out_name="so400m384", batch=128, workers=8,
               canonical_path=None):
    import torch, open_clip

    out_dir = os.path.join(IDX, out_name)
    os.makedirs(out_dir, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev} | Model: {model_name} | batch={batch} workers={workers}")
    print(f"Output: {out_dir}")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(dev).eval()
    if dev == "cuda":
        model = model.half()

    # The canonical map is an explicit input.  A raw-map rebuild starts from
    # global_keyframes.parquet; a production re-encode may use the stable
    # ViT-L runtime map.  Neither case depends on a workstation path.
    if canonical_path is None:
        canonical_path = os.path.join(IDX, "global_keyframes_vitl.parquet")
    idmap = pd.read_parquet(canonical_path).reset_index(drop=True)
    required_columns = {"video_id", "pack", "kf_n", "frame_idx", "pts_time"}
    missing_columns = sorted(required_columns - set(idmap.columns))
    if missing_columns:
        raise RuntimeError(f"canonical map missing columns: {missing_columns}: {canonical_path}")
    print(f"Idmap: {len(idmap):,} keyframes, {idmap['video_id'].nunique()} videos")

    # Probe output dim
    probe_row = idmap.iloc[0]
    probe_p = kf_path(probe_row.video_id, probe_row.kf_n)
    with torch.no_grad():
        _img = preprocess(Image.open(probe_p).convert("RGB")).unsqueeze(0).to(dev)
        if dev == "cuda": _img = _img.half()
        _f = model.encode_image(_img)
        out_dim = _f.shape[1]
    print(f"Output dim: {out_dim}")

    packs = sorted(idmap["pack"].unique())
    print(f"Packs: {len(packs)} ({packs[0]}..{packs[-1]})\n")

    for pack in packs:
        out_npy = os.path.join(out_dir, f"{pack}.npy")
        out_idmap = os.path.join(out_dir, f"{pack}_idmap.parquet")
        if os.path.exists(out_npy) and os.path.exists(out_idmap):
            print(f"SKIP {pack} (done)")
            continue

        gp = idmap[idmap["pack"] == pack].reset_index(drop=True)
        n = len(gp)
        t0 = time.time()

        feats = np.zeros((n, out_dim), np.float32)
        rows = list(gp.itertuples())

        q = queue.Queue(maxsize=batch * 4)

        def loader(chunk):
            for r in chunk:
                p = kf_path(r.video_id, r.kf_n)
                try:
                    img = preprocess(Image.open(p).convert("RGB"))
                    q.put((r.Index, img))
                except Exception:
                    q.put((r.Index, None))
            q.put(None)  # sentinel

        chunks = [rows[i::workers] for i in range(workers)]
        threads = [threading.Thread(target=loader, args=(ch,), daemon=True) for ch in chunks]
        for t in threads:
            t.start()

        sentinels = 0
        buf_idx, buf_t = [], []
        miss = 0
        done = 0
        report_every = max(1000, batch * 8)

        def flush():
            if not buf_t:
                return
            x = torch.stack(buf_t).to(dev)
            if dev == "cuda": x = x.half()
            with torch.no_grad():
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
            feats[buf_idx] = f.float().cpu().numpy()
            buf_idx.clear(); buf_t.clear()

        while sentinels < workers:
            item = q.get()
            if item is None:
                sentinels += 1
                continue
            idx, tensor = item
            done += 1
            if tensor is None:
                miss += 1
                continue
            buf_idx.append(idx)
            buf_t.append(tensor)
            if len(buf_t) >= batch:
                flush()
            if done % report_every == 0:
                print(f"  {pack}: {done}/{n} ({done*100//n}%)", flush=True)
        flush()

        if miss:
            raise RuntimeError(
                f"{pack}: {miss} canonical keyframe images could not be read; "
                "refusing to publish a zero-vector visual shard"
            )
        # A pack is published only after its matrix and map are complete.
        _atomic_npy(out_npy, feats.astype(np.float16))
        _atomic_parquet(out_idmap, gp[["g", "video_id", "kf_n", "frame_idx", "pts_time"]])

        rate = n / (time.time() - t0)
        print(f"DONE {pack}: {n-miss}/{n} kf | {rate:.1f} img/s | {time.time()-t0:.0f}s")

    print(f"\nAll packs done. Run with --merge to combine into single index.")


def merge(out_name="so400m384", model_name="ViT-SO400M-16-SigLIP2-384", pretrained="webli",
          canonical_path=None):
    out_dir = os.path.join(IDX, out_name)
    npy_files = sorted(glob.glob(os.path.join(out_dir, "*.npy")))
    if not npy_files:
        print(f"No .npy files found in {out_dir}")
        return

    feats, maps = [], []
    for fp in npy_files:
        pack = os.path.basename(fp)[:-4]
        idmap_fp = os.path.join(out_dir, f"{pack}_idmap.parquet")
        if not os.path.exists(idmap_fp):
            print(f"WARNING: missing idmap for {pack}, skipping")
            continue
        feats.append(np.load(fp))
        maps.append(pd.read_parquet(idmap_fp))

    F = np.concatenate(feats, axis=0)
    M = pd.concat(maps, ignore_index=True)

    out_feature_name, out_map_name = _runtime_artifact_names(out_name)
    out_npy = os.path.join(IDX, out_feature_name)
    out_kmap = os.path.join(IDX, out_map_name)
    _atomic_npy(out_npy, F)
    _atomic_parquet(out_kmap, M)
    manifest = {
        "schema_version": "hcmai.visual_index.v1",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "status": "ready",
        "variant": out_name,
        "model": model_name,
        "pretrained": pretrained,
        "canonical_input": str(canonical_path or os.path.join(IDX, "global_keyframes_vitl.parquet")),
        "rows": int(len(M)),
        "videos": int(M["video_id"].nunique()),
        "embedding": {"shape": list(F.shape), "dtype": str(F.dtype), "sha256": _sha256(out_npy)},
        "metadata": {"path": out_kmap, "sha256": _sha256(out_kmap)},
    }
    _atomic_json(os.path.join(IDX, f"{out_name}_visual_index_manifest.json"), manifest)
    print(f"Merged: {F.shape} float16 -> {out_npy}")
    print(f"Kmap:   {len(M)} rows     -> {out_kmap}")
    size_mb = os.path.getsize(out_npy) / 1024**2
    print(f"Index size: {size_mb:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Full keyframe encoder for any open_clip model")
    ap.add_argument("--model",    default="ViT-SO400M-16-SigLIP2-384", help="open_clip model name")
    ap.add_argument("--pretrained", default="webli")
    ap.add_argument("--out-name", default="so400m384", help="subdir name under data/index/")
    ap.add_argument("--batch",    type=int, default=128)
    ap.add_argument("--workers",  type=int, default=8)
    ap.add_argument("--merge",    action="store_true", help="Merge pack files into single index")
    ap.add_argument("--canonical", default=None,
                    help="Canonical parquet; use global_keyframes.parquet for a raw-map rebuild")
    a = ap.parse_args()
    if a.merge:
        merge(a.out_name, a.model, a.pretrained, a.canonical)
    else:
        encode_all(a.model, a.pretrained, a.out_name, a.batch, a.workers, a.canonical)
