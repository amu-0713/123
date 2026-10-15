# ===== 重新打包：只留分析需要的欄位，切成多個 < 24MB 的 zip =====
import csv, glob, io, os, zipfile, zlib

SRC = "data/spot/1d"                 # 下載腳本的輸出資料夾
LIMIT = 24 * 1000 * 1000             # 每個 zip 上限（留點餘裕）
KEEP = ["open_time", "high", "low", "close", "quote_volume", "taker_buy_quote"]

def compact(path):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if "taker_buy_quote" not in (r.fieldnames or []):   # _symbols.csv 原樣保留
            f.seek(0)
            return f.read().encode()
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(KEEP)
        for row in r:
            t = row["open_time"]
            w.writerow([t[:10] if t.endswith(" 00:00:00+00:00") else t] + [row[c] for c in KEEP[1:]])
        return out.getvalue().encode()

parts, zf, size = [], None, 0
for p in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
    data = compact(p)
    est = len(zlib.compress(data, 9)) + 200
    if zf is None or size + est > LIMIT:
        if zf:
            zf.close()
        parts.append(f"binance_1d_part{len(parts) + 1}.zip")
        zf = zipfile.ZipFile(parts[-1], "w", zipfile.ZIP_DEFLATED, compresslevel=9)
        size = 0
    zf.writestr("data/spot/1d/" + os.path.basename(p), data)
    size += est
zf.close()

for n in parts:
    print(f"{n}: {os.path.getsize(n) / 1e6:.1f} MB")
try:
    from google.colab import files
    for n in parts:
        files.download(n)
except ImportError:
    pass
