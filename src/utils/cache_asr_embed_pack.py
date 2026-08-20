"""
Embed các ASR chunk của 1 pack với bge-m3 (DO API). Có retry 429, batch 32, checkpoint.
Output: data/index/emb_cache_asr_<pack_lower>_chunks.npy (N, 1024)
"""
import os, sys, json, time, urllib.request, urllib.error
import numpy as np, pandas as pd

IDX = r"D:\HCMAI\data\index"

def load_env(p=r"D:\HCMAI\.env"):
    e = {}
    for l in open(p):
        l = l.strip()
        if l and "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1); e[k.strip()] = v.strip()
    return e

ENV = load_env(); KEY = ENV["DO_INFERENCE_KEY"]; BASE = ENV["DO_INFERENCE_BASE"]

def embed_batch(texts, retries=6):
    pl = {"model": "bge-m3", "input": texts}
    req = urllib.request.Request(
        BASE.rstrip("/") + "/embeddings", data=json.dumps(pl).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return [item["embedding"] for item in d["data"]]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 8 * (a + 1)
                print(f"  429 rate-limit, sleep {wait}s ...")
                time.sleep(wait); continue
            raise
        except Exception as e:
            print(f"  err {e}, retry"); time.sleep(5); continue
    raise RuntimeError("embed failed all retries")

def main(pack, batch=16):
    pack_l = pack.lower()
    chunks_fp = os.path.join(IDX, f"asr_chunks_{pack_l}_ts.parquet")
    out_fp = os.path.join(IDX, f"emb_cache_asr_{pack_l}_chunks.npy")
    if not os.path.exists(chunks_fp):
        print(f"Không có {chunks_fp}. Chạy rebuild_asr_chunks_pack trước."); return
    df = pd.read_parquet(chunks_fp)
    n = len(df)
    print(f"Pack {pack}: embed {n} chunks (batch={batch})")

    # Resume nếu có file partial
    embs = None
    if os.path.exists(out_fp):
        prev = np.load(out_fp)
        if len(prev) == n:
            print(f"Đã có {out_fp} với {len(prev)} dòng. Skip.")
            return
        if len(prev) < n:
            embs = prev.tolist()
            print(f"Resume từ {len(prev)}/{n}")
    if embs is None:
        embs = []
    start = len(embs)
    t0 = time.time()
    for i in range(start, n, batch):
        texts = df.chunk.iloc[i:i+batch].tolist()
        out = embed_batch(texts)
        embs.extend(out)
        if (i // batch) % 5 == 0:
            arr = np.array(embs, dtype=np.float32)
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
            np.save(out_fp, arr)
            print(f"  {len(embs)}/{n} | {time.time()-t0:.0f}s")
    arr = np.array(embs, dtype=np.float32)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-8)
    np.save(out_fp, arr)
    print(f"DONE: {arr.shape} -> {out_fp}")

if __name__ == "__main__":
    pack = sys.argv[1] if len(sys.argv) > 1 else "K02"
    main(pack)
