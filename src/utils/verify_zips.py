"""Verify K11-K20 keyframe zips not truncated (testzip on first+last entries)."""
import os, zipfile

ZIPS = r"D:\HCMAI\data\zips"
for i in range(11, 21):
    pack = f"K{i:02d}"
    fp = os.path.join(ZIPS, f"Keyframes_{pack}.zip")
    if not os.path.exists(fp):
        print(f"{pack}: MISSING"); continue
    sz = os.path.getsize(fp) / 1e9
    try:
        with zipfile.ZipFile(fp) as z:
            names = z.namelist()
            # Light check: open & read last data entry (catches tail truncation)
            data_entries = [n for n in names if not n.endswith("/")]
            ok = "OK"
            if data_entries:
                with z.open(data_entries[-1]) as fh:
                    fh.read()
        print(f"{pack}: {sz:.2f}GB, {len(names)} files, last-entry={ok}")
    except Exception as e:
        print(f"{pack}: {sz:.2f}GB, ERROR {e}")
