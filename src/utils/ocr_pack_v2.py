"""
OCR cho 1 pack BẤT KỲ (K01-K20, L21-L30) — dùng global_keyframes.parquet.
Idempotent, resume-able. Final khi đủ video pack.

Cách dùng:
  python src/utils/ocr_pack_v2.py L21
  python src/utils/ocr_pack_v2.py K11
"""
import os, sys, json, base64, time, urllib.request, urllib.error
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "utils"))
from paths import INDEX_DIR, KEYFRAMES_DIR, load_env  # noqa: E402

IDX = str(INDEX_DIR)
KF = str(KEYFRAMES_DIR)

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
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            c = d["choices"][0]["message"]["content"]
            return c.strip() if c else ""
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(8 * (a + 1)); continue
            return ""
        except Exception:
            time.sleep(3)
    return ""

def main(pack):
    pack_l = pack.lower()
    final_fp = os.path.join(IDX, f"ocr_{pack_l}.parquet")
    partial_fp = os.path.join(IDX, f"ocr_{pack_l}_partial.parquet")

    # Use global_keyframes.parquet (works for any pack)
    gkf = pd.read_parquet(os.path.join(IDX, "global_keyframes.parquet"))
    kmap = gkf[gkf.video_id.str.startswith(pack)]
    target_videos = kmap.video_id.nunique()
    if target_videos == 0:
        print(f"Pack {pack} không có trong global_keyframes."); return
    print(f"Pack {pack}: {target_videos} video target")

    # Resume
    done_keys = set()
    rows_all = []
    for fp in [partial_fp, final_fp]:
        if os.path.exists(fp):
            prev = pd.read_parquet(fp)
            done_keys |= set(zip(prev.video_id, prev.kf_n))
            rows_all = prev.to_dict("records")
            break

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
    print(f"OCR cần process: {len(sel)} frames (resume {len(done_keys)})")

    def work(item):
        vid, kf, t, p = item
        return (vid, kf, t), ocr(p)

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(work, it) for it in sel]
        for f in as_completed(futs):
            key, txt = f.result(); done += 1
            if txt and txt.upper() != "NONE" and len(txt) > 3:
                rows_all.append({"video_id": key[0], "kf_n": key[1],
                                 "pts_time": key[2], "ocr_text": txt})
            if done % 50 == 0:
                print(f"  {done}/{len(sel)} {time.time()-t0:.0f}s "
                      f"(text:{len(rows_all)})", flush=True)
                pd.DataFrame(rows_all).to_parquet(partial_fp)

    df = pd.DataFrame(rows_all)
    df.to_parquet(partial_fp)
    n_vid = df.video_id.nunique() if len(df) > 0 else 0
    # FINAL when ALL frames processed, even if some videos had no text
    # (some videos truly have no text overlay)
    if len(sel) == 0 and len(rows_all) >= 0:  # nothing left to process
        df.to_parquet(final_fp)
        print(f"DONE FULL {pack}: {len(df)} text frames từ {n_vid}/{target_videos} video "
              f"(some videos may have no text overlay).")
    elif n_vid >= target_videos:
        df.to_parquet(final_fp)
        print(f"DONE FULL {pack}: {len(df)} text frames từ {n_vid}/{target_videos} video.")
    else:
        print(f"Partial {pack}: {len(df)} frames từ {n_vid}/{target_videos}.")

if __name__ == "__main__":
    pack = sys.argv[1] if len(sys.argv) > 1 else "L21"
    main(pack)
