"""Tóm tắt index đã build, dùng để kiểm tra trạng thái."""
import os, glob
import numpy as np, pandas as pd

IDX = r"D:\HCMAI\data\index"

print("=== INDEX SUMMARY ===\n")

g = np.load(os.path.join(IDX, "global_siglip.npy"))
m = pd.read_parquet(os.path.join(IDX, "global_keyframes.parquet"))
print(f"Global SigLIP2: {g.shape}, {m.video_id.nunique()} videos, {len(m)} kf")
print(f"  L kf: {m.pack.str.startswith('L').sum()}, "
      f"K kf: {m.pack.str.startswith('K').sum()}")
print()

ac_files = sorted(glob.glob(os.path.join(IDX, "asr_chunks_*_ts.parquet")))
total_asr = 0
print("=== ASR ===")
for f in ac_files:
    pack = os.path.basename(f).replace("asr_chunks_", "").replace("_ts.parquet", "")
    n = len(pd.read_parquet(f))
    emb_fp = os.path.join(IDX, f"emb_cache_asr_{pack}_chunks.npy")
    e = np.load(emb_fp).shape if os.path.exists(emb_fp) else "NONE"
    print(f"  {pack}: {n} chunks, embed {e}")
    total_asr += n
print(f"  TOTAL ASR: {total_asr}\n")

ocr_files = [f for f in sorted(glob.glob(os.path.join(IDX, "ocr_*.parquet")))
             if all(x not in f for x in ["_partial", "_compare", "_gt", "ocr_query"])]
total_ocr = 0
print("=== OCR ===")
for f in ocr_files:
    n = len(pd.read_parquet(f))
    print(f"  {os.path.basename(f)}: {n} text frames")
    total_ocr += n
print(f"  TOTAL OCR: {total_ocr}\n")

q_files = sorted(glob.glob(os.path.join(IDX, "*queryset*.parquet")))
print("=== QUERY SETS ===")
for f in q_files:
    df = pd.read_parquet(f)
    print(f"  {os.path.basename(f)}: {len(df)} queries, "
          f"{df.gt_video.nunique() if 'gt_video' in df.columns else '-'} GT videos")
