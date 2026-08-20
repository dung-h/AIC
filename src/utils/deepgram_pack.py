"""
Deepgram ASR cho 1 pack video (K01, K02, ...) — chung cho mọi pack.
Resume-able: bỏ qua video đã có .json output.

Cách dùng:
  python src/deepgram_pack.py K02
  python src/deepgram_pack.py K01   # idempotent
"""
import os, sys, json, time, glob, subprocess, urllib.request

def load_env(p=r"D:\HCMAI\.env"):
    e = {}
    for l in open(p):
        l = l.strip()
        if l and "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1); e[k.strip()] = v.strip()
    return e

ENV = load_env(); KEY = ENV["DEEPGRAM_API_KEY"]

def main(pack):
    pack_l = pack.lower()
    VID_DIR = rf"D:\HCMAI\data\video_{pack_l}\video"
    AUD     = rf"D:\HCMAI\data\audio_{pack_l}"
    OUT     = rf"D:\HCMAI\data\asr_{pack_l}"
    os.makedirs(AUD, exist_ok=True); os.makedirs(OUT, exist_ok=True)

    mp4s = sorted(glob.glob(os.path.join(VID_DIR, "*.mp4")))
    if not mp4s:
        print(f"Không có mp4 nào trong {VID_DIR}. Hãy giải nén Videos_{pack}.zip trước.")
        return
    print(f"Pack {pack}: {len(mp4s)} videos")

    # Skip nếu đã có json
    todo = [m for m in mp4s if not os.path.exists(os.path.join(OUT, os.path.splitext(os.path.basename(m))[0] + ".json"))]
    print(f"Cần ASR: {len(todo)} video (skip {len(mp4s)-len(todo)} đã xong)")

    total_a = 0; total_t = 0
    for mp4 in todo:
        vid = os.path.splitext(os.path.basename(mp4))[0]
        wav = os.path.join(AUD, f"{vid}.wav")
        if not os.path.exists(wav) or os.path.getsize(wav) < 1000:
            r = subprocess.run(["ffmpeg", "-y", "-i", mp4, "-ar", "16000", "-ac", "1", "-vn", wav],
                               capture_output=True)
            if r.returncode != 0:
                print(f"[{vid}] ffmpeg ERR: {r.stderr.decode(errors='ignore')[:200]}")
                continue
        try:
            import wave
            with wave.open(wav) as w: dur = w.getnframes() / w.getframerate()
        except Exception as e:
            print(f"[{vid}] wave ERR: {e}"); continue

        url = ("https://api.deepgram.com/v1/listen?model=nova-3&language=vi"
               "&punctuate=true&smart_format=true&utterances=true")
        data = open(wav, "rb").read()
        req = urllib.request.Request(url, data=data, method="POST",
            headers={"Authorization": f"Token {KEY}", "Content-Type": "audio/wav"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                res = json.load(r)
            dt = time.time() - t0
            tr = res["results"]["channels"][0]["alternatives"][0]
            txt = tr.get("transcript", "")
            json.dump(res, open(os.path.join(OUT, f"{vid}.json"), "w", encoding="utf-8"),
                      ensure_ascii=False)
            total_a += dur; total_t += dt
            print(f"[{vid}] {dur:.0f}s -> {dt:.1f}s | conf={tr.get('confidence',0):.3f} "
                  f"| {len(txt.split())} words")
        except Exception as e:
            print(f"[{vid}] ERR: {e}")

    if total_t > 0:
        print(f"\n=== Pack {pack}: {total_a:.0f}s audio in {total_t:.1f}s "
              f"({total_a/total_t:.1f}x realtime) ===")

if __name__ == "__main__":
    pack = sys.argv[1] if len(sys.argv) > 1 else "K02"
    main(pack)
