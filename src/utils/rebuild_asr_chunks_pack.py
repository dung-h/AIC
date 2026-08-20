"""
Rebuild ASR chunks CÓ TIMESTAMP cho 1 pack (K01, K02, ...).
Output: data/index/asr_chunks_<pack_lower>_ts.parquet (chunk, vid, start, end, kf_n, frame_idx)

Cần: data/asr_<pack_lower>/*.json (Deepgram), data/index/k_id_map.parquet (đã build)

Cách dùng:
  python src/rebuild_asr_chunks_pack.py K01
  python src/rebuild_asr_chunks_pack.py K02
"""
import os, sys, json, glob
import pandas as pd

IDX = r"D:\HCMAI\data\index"

def main(pack):
    pack_l = pack.lower()
    ASR = rf"D:\HCMAI\data\asr_{pack_l}"
    out_fp = os.path.join(IDX, f"asr_chunks_{pack_l}_ts.parquet")

    kmap = pd.read_parquet(os.path.join(IDX, "k_id_map.parquet"))

    def kf_near(vid, t):
        g = kmap[kmap.video_id == vid]
        if len(g) == 0: return None, None
        j = (g.pts_time - t).abs().idxmin()
        return int(g.loc[j, "kf_n"]), int(g.loc[j, "frame_idx"])

    rows = []
    files = sorted(glob.glob(os.path.join(ASR, "*.json")))
    print(f"Pack {pack}: {len(files)} ASR json files")
    for f in files:
        vid = os.path.splitext(os.path.basename(f))[0]
        try:
            res = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"[{vid}] read ERR: {e}"); continue
        try:
            alt = res["results"]["channels"][0]["alternatives"][0]
        except Exception:
            print(f"[{vid}] no alternatives"); continue
        words = alt.get("words", [])
        if not words: continue
        for i in range(0, len(words), 80):
            grp = words[i:i+80]
            if len(grp) < 5: continue
            text = " ".join(w.get("punctuated_word", w["word"]) for w in grp)
            start = float(grp[0]["start"]); end = float(grp[-1]["end"])
            mid = (start + end) / 2
            kf, fidx = kf_near(vid, mid)
            rows.append({"chunk": text, "vid": vid, "start": start, "end": end,
                         "kf_n": kf, "frame_idx": fidx})
    df = pd.DataFrame(rows)
    df.to_parquet(out_fp)
    print(f"Pack {pack}: {len(df)} chunks, {df.vid.nunique() if len(df)>0 else 0} videos -> {out_fp}")

if __name__ == "__main__":
    pack = sys.argv[1] if len(sys.argv) > 1 else "K02"
    main(pack)
