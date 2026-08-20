"""Extract keyframes from Keyframes_<pack>.zip into data/keyframes/<pack>_V*.

Idempotent. Skip if already extracted.
"""
import os, sys, zipfile, time
import pandas as pd

KF_DIR = r"D:\HCMAI\data\keyframes"

def main(pack):
    zip_fp = rf"D:\HCMAI\data\zips\Keyframes_{pack}.zip"
    if not os.path.exists(zip_fp):
        print(f"No zip {zip_fp}"); return
    t0 = time.time()
    with zipfile.ZipFile(zip_fp) as z:
        names = z.namelist()
        # Skip if all already extracted
        # Check 1 random file
        print(f"Pack {pack}: {len(names)} files in zip")
        already = 0; extracted = 0
        for n in names:
            target = os.path.join(KF_DIR, n.replace("/", os.sep))
            if os.path.exists(target) and os.path.getsize(target) > 0:
                already += 1; continue
            z.extract(n, KF_DIR)
            extracted += 1
            if extracted % 1000 == 0:
                print(f"  {extracted} extracted, {time.time()-t0:.0f}s")
    print(f"DONE {pack}: extracted {extracted}, already {already}, in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    pack = sys.argv[1] if len(sys.argv) > 1 else "K03"
    main(pack)
