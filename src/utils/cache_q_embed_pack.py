"""
Embed (bge-m3) một query set parquet thành emb_cache_<name>.npy.
Idempotent. Resume.
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

def embed_batch(texts):
    pl = {"model": "bge-m3", "input": texts}
    req = urllib.request.Request(
        BASE.rstrip("/") + "/embeddings", data=json.dumps(pl).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for a in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            return [item["embedding"] for item in d["data"]]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(8 * (a + 1)); continue
            raise
        except Exception:
            time.sleep(5); continue
    raise RuntimeError("embed failed")

def main(qs_fp, out_name, col="query", batch=16):
    qs_path = qs_fp if os.path.isabs(qs_fp) else os.path.join(IDX, qs_fp)
    out_fp = os.path.join(IDX, f"emb_cache_{out_name}.npy")
    df = pd.read_parquet(qs_path)
    n = len(df)
    print(f"Embed {n} {col} from {qs_path}")

    embs = []
    if os.path.exists(out_fp):
        prev = np.load(out_fp)
        if len(prev) == n:
            print(f"Đã có {out_fp}, skip"); return
        if len(prev) < n:
            embs = prev.tolist(); print(f"Resume từ {len(prev)}/{n}")

    t0 = time.time()
    for i in range(len(embs), n, batch):
        out = embed_batch(df[col].iloc[i:i+batch].tolist())
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
    if len(sys.argv) < 3:
        print("Usage: python cache_q_embed_pack.py <queryset_parquet> <out_name> [col=query]")
        sys.exit(1)
    qs_fp = sys.argv[1]
    out_name = sys.argv[2]
    col = sys.argv[3] if len(sys.argv) > 3 else "query"
    main(qs_fp, out_name, col)
