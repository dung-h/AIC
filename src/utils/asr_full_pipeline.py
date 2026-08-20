"""
Full ASR Pipeline: tải video zip → extract audio → XÓA zip → Deepgram ASR → rebuild chunks.
Tuần tự theo pack. Tiết kiệm disk (~10GB buffer max bất kỳ lúc nào).

Flow cho mỗi pack:
1. Download Videos_<pack>.zip (~8GB) nếu chưa có
2. Giải nén mp4 vào data/video_<pack>/video/
3. XÓA zip (free 8GB)
4. Extract audio wav (16kHz mono) → data/audio_<pack>/
5. XÓA mp4 gốc (free ~8GB nữa nếu cần, hoặc giữ nếu đủ chỗ)
6. Deepgram ASR → data/asr_<pack>/*.json
7. Rebuild chunks + embed → data/index/asr_chunks_<pack>_ts.parquet + emb_cache

Cách dùng:
  python src/utils/asr_full_pipeline.py K03
  python src/utils/asr_full_pipeline.py K03 K04 K05 ...
  python src/utils/asr_full_pipeline.py --all   (K03-K20)
"""
import os, sys, json, time, glob, subprocess, zipfile, shutil, wave
import urllib.request
import numpy as np, pandas as pd

IDX = r"D:\HCMAI\data\index"
ZIPS = r"D:\HCMAI\data\zips"
BASE_URL = "https://aic-data.ledo.io.vn"


def load_env(p=r"D:\HCMAI\.env"):
    e = {}
    for l in open(p):
        l = l.strip()
        if l and "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1); e[k.strip()] = v.strip()
    return e


ENV = load_env()
DG_KEY = ENV["DEEPGRAM_API_KEY"]


def download_pack(pack):
    """Download Videos_<pack>.zip nếu chưa có."""
    fn = f"Videos_{pack}.zip"
    fp = os.path.join(ZIPS, fn)
    if os.path.exists(fp) and os.path.getsize(fp) > 1e9:
        print(f"  [{pack}] zip exists ({os.path.getsize(fp)/1e9:.1f}GB), skip download")
        return fp
    url = f"{BASE_URL}/{fn}"
    print(f"  [{pack}] Downloading {url}...")
    subprocess.run(["curl.exe", "-s", "-o", fp, url], check=True)
    sz = os.path.getsize(fp) / 1e9
    print(f"  [{pack}] Downloaded: {sz:.1f}GB")
    return fp


def extract_videos(pack, zip_fp):
    """Giải nén mp4 từ zip."""
    vid_dir = rf"D:\HCMAI\data\video_{pack.lower()}\video"
    os.makedirs(vid_dir, exist_ok=True)
    existing = glob.glob(os.path.join(vid_dir, "*.mp4"))
    if len(existing) >= 25:
        print(f"  [{pack}] {len(existing)} mp4 already extracted, skip")
        return vid_dir
    print(f"  [{pack}] Extracting mp4...")
    with zipfile.ZipFile(zip_fp) as z:
        names = [n for n in z.namelist() if n.endswith(".mp4")]
        for n in names:
            target = os.path.join(rf"D:\HCMAI\data\video_{pack.lower()}", n.replace("/", os.sep))
            if not os.path.exists(target) or os.path.getsize(target) < 1000:
                z.extract(n, rf"D:\HCMAI\data\video_{pack.lower()}")
    mp4s = glob.glob(os.path.join(vid_dir, "*.mp4"))
    print(f"  [{pack}] {len(mp4s)} mp4 extracted")
    return vid_dir


def extract_audio(pack, vid_dir):
    """Extract audio wav (16kHz mono) từ mp4."""
    aud_dir = rf"D:\HCMAI\data\audio_{pack.lower()}"
    os.makedirs(aud_dir, exist_ok=True)
    mp4s = sorted(glob.glob(os.path.join(vid_dir, "*.mp4")))
    extracted = 0
    for mp4 in mp4s:
        vid = os.path.splitext(os.path.basename(mp4))[0]
        wav = os.path.join(aud_dir, f"{vid}.wav")
        if os.path.exists(wav) and os.path.getsize(wav) > 1000:
            continue
        r = subprocess.run(["ffmpeg", "-y", "-i", mp4, "-ar", "16000", "-ac", "1", "-vn", wav],
                           capture_output=True)
        if r.returncode == 0:
            extracted += 1
    print(f"  [{pack}] Audio extracted: {extracted} new wavs")
    return aud_dir


def deepgram_asr(pack, aud_dir):
    """ASR Deepgram nova-3 cho mỗi wav. Skip nếu json đã có."""
    out_dir = rf"D:\HCMAI\data\asr_{pack.lower()}"
    os.makedirs(out_dir, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(aud_dir, "*.wav")))
    todo = [w for w in wavs if not os.path.exists(
        os.path.join(out_dir, os.path.splitext(os.path.basename(w))[0] + ".json"))]
    print(f"  [{pack}] ASR: {len(todo)} wavs to process ({len(wavs)-len(todo)} done)")

    total_audio = 0; total_time = 0
    for wi, wav in enumerate(todo):
        vid = os.path.splitext(os.path.basename(wav))[0]
        try:
            with wave.open(wav) as w:
                dur = w.getnframes() / w.getframerate()
        except Exception:
            continue
        url = ("https://api.deepgram.com/v1/listen?model=nova-3&language=vi"
               "&punctuate=true&smart_format=true&utterances=true")
        data = open(wav, "rb").read()
        req = urllib.request.Request(url, data=data, method="POST",
            headers={"Authorization": f"Token {DG_KEY}", "Content-Type": "audio/wav"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                res = json.load(r)
            dt = time.time() - t0
            json.dump(res, open(os.path.join(out_dir, f"{vid}.json"), "w", encoding="utf-8"),
                      ensure_ascii=False)
            total_audio += dur; total_time += dt
            if (wi + 1) % 5 == 0:
                rtf = total_audio / total_time if total_time > 0 else 0
                print(f"    {wi+1}/{len(todo)} | {rtf:.0f}x realtime", flush=True)
        except Exception as e:
            print(f"    [{vid}] ERR: {str(e)[:80]}")

    if total_time > 0:
        print(f"  [{pack}] ASR done: {total_audio:.0f}s audio in {total_time:.0f}s "
              f"({total_audio/total_time:.0f}x realtime)")


def rebuild_chunks(pack):
    """Rebuild ASR chunks với timestamp."""
    pack_l = pack.lower()
    asr_dir = rf"D:\HCMAI\data\asr_{pack_l}"
    out_fp = os.path.join(IDX, f"asr_chunks_{pack_l}_ts.parquet")
    kmap = pd.read_parquet(os.path.join(IDX, "k_id_map.parquet"))

    def kf_near(vid, t):
        g = kmap[kmap.video_id == vid]
        if len(g) == 0: return None, None
        j = (g.pts_time - t).abs().idxmin()
        return int(g.loc[j, "kf_n"]), int(g.loc[j, "frame_idx"])

    rows = []
    for f in sorted(glob.glob(os.path.join(asr_dir, "*.json"))):
        vid = os.path.splitext(os.path.basename(f))[0]
        try:
            res = json.load(open(f, encoding="utf-8"))
            words = res["results"]["channels"][0]["alternatives"][0].get("words", [])
        except Exception:
            continue
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
    print(f"  [{pack}] Chunks: {len(df)} chunks, {df.vid.nunique()} videos -> {out_fp}")


def embed_chunks(pack):
    """Embed ASR chunks with bge-m3 (offline local for speed)."""
    pack_l = pack.lower()
    chunks_fp = os.path.join(IDX, f"asr_chunks_{pack_l}_ts.parquet")
    out_fp = os.path.join(IDX, f"emb_cache_asr_{pack_l}_chunks.npy")
    if not os.path.exists(chunks_fp):
        print(f"  [{pack}] No chunks file"); return
    if os.path.exists(out_fp):
        existing = np.load(out_fp)
        df = pd.read_parquet(chunks_fp)
        if len(existing) == len(df):
            print(f"  [{pack}] Embed already done ({len(existing)} rows)"); return

    df = pd.read_parquet(chunks_fp)
    print(f"  [{pack}] Embedding {len(df)} chunks (bge-m3 local)...")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
    from offline_fallback import TextEmbedderOffline
    embedder = TextEmbedderOffline()
    embs = embedder.embed(df.chunk.tolist(), batch_size=32)
    np.save(out_fp, embs)
    print(f"  [{pack}] Saved embed {embs.shape} -> {out_fp}")


def cleanup(pack, delete_zip=True, delete_mp4=True):
    """Xóa zip + mp4 sau khi extract audio xong."""
    if delete_zip:
        zip_fp = os.path.join(ZIPS, f"Videos_{pack}.zip")
        if os.path.exists(zip_fp):
            os.remove(zip_fp)
            print(f"  [{pack}] Deleted zip (freed {8:.0f}GB)")
    if delete_mp4:
        vid_dir = rf"D:\HCMAI\data\video_{pack.lower()}\video"
        if os.path.exists(vid_dir):
            shutil.rmtree(rf"D:\HCMAI\data\video_{pack.lower()}")
            print(f"  [{pack}] Deleted video dir")


def process_pack(pack, keep_video=False):
    """Full pipeline cho 1 pack."""
    print(f"\n{'='*60}")
    print(f"  PACK {pack}")
    print(f"{'='*60}")

    # Check if already done
    chunks_fp = os.path.join(IDX, f"asr_chunks_{pack.lower()}_ts.parquet")
    emb_fp = os.path.join(IDX, f"emb_cache_asr_{pack.lower()}_chunks.npy")
    if os.path.exists(chunks_fp) and os.path.exists(emb_fp):
        df = pd.read_parquet(chunks_fp)
        emb = np.load(emb_fp)
        if len(df) == len(emb) and len(df) > 0:
            print(f"  [{pack}] ALREADY DONE ({len(df)} chunks). Skip.")
            return

    zip_fp = download_pack(pack)
    vid_dir = extract_videos(pack, zip_fp)
    aud_dir = extract_audio(pack, vid_dir)
    # Delete zip immediately to free space
    cleanup(pack, delete_zip=True, delete_mp4=False)
    deepgram_asr(pack, aud_dir)
    rebuild_chunks(pack)
    embed_chunks(pack)
    # Optionally delete video dir
    if not keep_video:
        cleanup(pack, delete_zip=False, delete_mp4=True)
    print(f"  [{pack}] ✓ COMPLETE")


def main():
    if "--all" in sys.argv:
        packs = [f"K{i:02d}" for i in range(3, 21)]
    else:
        packs = [a for a in sys.argv[1:] if a.startswith("K")]
    if not packs:
        packs = ["K03"]
    print(f"ASR Full Pipeline: {packs}")
    print(f"Free disk: {shutil.disk_usage('D:\\').free // (1024**3)} GB\n")
    for pack in packs:
        process_pack(pack, keep_video=False)
    print(f"\n{'='*60}")
    print("ALL PACKS DONE")
    print(f"Free disk: {shutil.disk_usage('D:\\').free // (1024**3)} GB")


if __name__ == "__main__":
    main()
