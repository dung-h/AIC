"""
GĐ1: Embed OCR text bằng bge-m3 LOCAL → semantic OCR retrieval (thay BM25 lexical).

Cho mỗi pack có ocr_<pack>.parquet → emb_cache_ocr_<pack>.npy (N, 1024).
Idempotent. Dùng offline bge-m3 (22x nhanh, no rate-limit).

Cách dùng:
  python src/utils/embed_ocr_pack.py            # all OCR packs
  python src/utils/embed_ocr_pack.py k01 l21    # specific
"""
import os, sys, glob
import numpy as np, pandas as pd

IDX = r"D:\HCMAI\data\index"


def main(packs=None):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
    from offline_fallback import TextEmbedderOffline

    # Discover OCR final files
    ocr_files = [f for f in sorted(glob.glob(os.path.join(IDX, "ocr_*.parquet")))
                 if all(x not in f for x in ["_partial", "_compare", "_gt", "ocr_query"])]
    if packs:
        packs_l = [p.lower() for p in packs]
        ocr_files = [f for f in ocr_files
                     if os.path.basename(f).replace("ocr_", "").replace(".parquet", "") in packs_l]

    print(f"Embedding OCR for {len(ocr_files)} packs (bge-m3 local)")
    embedder = TextEmbedderOffline()
    embedder.embed(["warmup"])

    for fp in ocr_files:
        pack = os.path.basename(fp).replace("ocr_", "").replace(".parquet", "")
        out_fp = os.path.join(IDX, f"emb_cache_ocr_{pack}.npy")
        df = pd.read_parquet(fp)
        if os.path.exists(out_fp):
            prev = np.load(out_fp)
            if len(prev) == len(df):
                print(f"  {pack}: already done ({len(prev)})"); continue
        texts = df.ocr_text.fillna("").astype(str).tolist()
        embs = embedder.embed(texts, batch_size=64)
        np.save(out_fp, embs)
        print(f"  {pack}: {embs.shape} -> {out_fp}")
    print("DONE")


if __name__ == "__main__":
    packs = sys.argv[1:] if len(sys.argv) > 1 else None
    main(packs)
