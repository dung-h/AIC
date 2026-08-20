"""
OCR cho 1 pack (K01, K02, ...) — gemma API, thưa 1 frame mỗi ~10s.
Resume từ partial. Idempotent. Khi đủ tất cả video của pack -> ghi final.

Cách dùng:
  python src/ocr_pack.py K01
  python src/ocr_pack.py K02
"""
import os, sys, json, base64, time, urllib.request
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

IDX = r"D:\HCMAI\data\index"
KF = r"D:\HCMAI\data\keyframes\keyframes"

def load_env(p=r"D:\HCMAI\.env"):
    e = {}
    for l in open(p):
        l = l.strip()
        if l and "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1); e[k.strip()] = v.strip()
    return e

ENV = load_env(); KEY = ENV["DO_INFERENCE_KEY"]; BASE = ENV["DO_INFERENCE_BASE"]
PROMPT = ("Extract ALL Vietnamese text visible in this TV news frame exactly with diacritics. "
          "Output only the text lines, or NONE if no text.")

def ocr(p):
    b64 = base64.b64encode(open(p, "rb").read()).decode()
    pl = {"model": "gemma-4-31B-it",
          "messages": [{"role": "user", "content": [
              {"type": "text", "text": PROMPT},
              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
          "max_tokens": 200, "temperature": 0.0}
    req = urllib.request.Request(BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(pl).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for a in range(2):
        try:
            with urllib.request.urlopen(req, timeout=90) as r: d = json.load(r)
            c = d["choices"][0]["message"]["content"]
            return c.strip() if c else ""
        except Exception:
            time.sleep(2)
    return ""

def main(pack):
    pack_l = pack.lower()
    final_fp = os.path.join(IDX, f"ocr_{pack_l}.parquet")
    partial_fp = os.path.join(IDX, f"ocr_{pack_l}_partial.parquet")

    kmap = pd.read_parquet(os.path.join(IDX, "k_id_map.parquet"))
    kmap = kmap[kmap.video_id.str.startswith(pack)]
    target_videos = kmap.video_id.nunique()
    if target_videos == 0:
        print(f"Không có video {pack} trong k_id_map. Build idmap trước."); return

    # Resume từ cả partial + final
    done_keys = set()
    rows_all = []
    for fp in [partial_fp, final_fp]:
        if os.path.exists(fp):
            prev = pd.read_parquet(fp)
            done_keys |= set(zip(prev.video_id, prev.kf_n))
            rows_all = prev.to_dict("records")
            break  # ưu tiên partial nếu có; nếu không thì final
    print(f"Pack {pack}: {target_videos} video target; resume {len(done_keys)} frame")

    # Thưa: 1 frame mỗi ~10s
    sel = []
    for vid, g in kmap.groupby("video_id"):
        g = g.sort_values("pts_time"); last = -100
        for r in g.itertuples():
            if r.pts_time - last >= 10:
                p = os.path.join(KF, vid, f"{int(r.kf_n):03d}.jpg")
                if os.path.exists(p):
                    sel.append((vid, int(r.kf_n), float(r.pts_time), p))
                    last = r.pts_time
    sel = [s for s in sel if (s[0], s[1]) not in done_keys]
    print(f"OCR {len(sel)} frames còn lại (thưa ~10s)")

    def work(item):
        vid, kf, t, p = item
        return (vid, kf, t), ocr(p)

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(work, it) for it in sel]
        for f in as_completed(futs):
            key, txt = f.result(); done += 1
            if txt and txt.upper() != "NONE" and len(txt) > 3:
                rows_all.append({"video_id": key[0], "kf_n": key[1], "pts_time": key[2],
                                 "ocr_text": txt})
            if done % 50 == 0:
                print(f"  {done}/{len(sel)} {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows_all).to_parquet(partial_fp)

    df = pd.DataFrame(rows_all)
    df.to_parquet(partial_fp)
    n_vid = df.video_id.nunique() if len(df) > 0 else 0
    if n_vid >= target_videos:
        df.to_parquet(final_fp)
        print(f"\nDONE FULL {pack}: {len(df)} frames text từ {n_vid}/{target_videos} video.")
    else:
        print(f"\nPartial {pack}: {len(df)} frames từ {n_vid}/{target_videos} video.")

if __name__ == "__main__":
    pack = sys.argv[1] if len(sys.argv) > 1 else "K01"
    main(pack)
