#!/usr/bin/env python3
"""
下載 Binance USDT 永續合約的歷史資金費率（data.binance.vision 官方公開資料）。

只用 Python 標準函式庫。Colab：整段貼進一個 cell 執行，跑完自動下載 funding_rates.csv。

用法：
    python download_funding.py                          # 預設 BTC ETH BNB SOL XRP
    python download_funding.py --symbols BTCUSDT ETHUSDT

輸出 funding_rates.csv：symbol,time,rate
    time 為 UTC 結算時間（通常每 8 小時一次），rate 為該次費率（0.0001 = 0.01%）
"""
import argparse
import csv
import datetime as dt
import io
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILE_URL = "https://data.binance.vision/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def http_get(url, retries=5, timeout=60):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "funding-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:
            err = e
        time.sleep(2 ** i)
    raise RuntimeError(f"下載失敗 {url}: {err}")


def s3_keys(prefix):
    keys, marker = [], ""
    while True:
        q = {"prefix": prefix}
        if marker:
            q["marker"] = marker
        body = http_get(LIST_URL + "?" + urllib.parse.urlencode(q))
        if body is None:
            break
        root = ET.fromstring(body)
        page = [e.text for e in root.iter(NS + "Key")]
        keys += page
        if root.findtext(NS + "IsTruncated") != "true" or not page:
            break
        marker = page[-1]
    return keys


def to_iso(ts):
    ts = int(float(ts))
    if ts > 10 ** 14:
        ts //= 1000
    return dt.datetime.fromtimestamp(ts / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")


def parse(blob):
    """檔案格式：calc_time,funding_interval_hours,last_funding_rate（有些檔案有表頭）"""
    rows = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            lines = z.read(name).decode().splitlines()
            if not lines:
                continue
            head = lines[0].lower().split(",")
            if "calc_time" in head:
                ti, ri = head.index("calc_time"), head.index("last_funding_rate")
                lines = lines[1:]
            else:
                ti, ri = 0, -1
            for line in lines:
                p = line.split(",")
                if len(p) < 2 or not p[ti].strip().replace(".", "").isdigit():
                    continue
                rows.append((to_iso(p[ti]), float(p[ri])))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"])
    ap.add_argument("--out", default="funding_rates.csv")
    a, _ = ap.parse_known_args(argv)

    allrows = []
    for sym in a.symbols:
        keys = sorted(k for k in s3_keys(f"data/futures/um/monthly/fundingRate/{sym}/") if k.endswith(".zip"))
        rows = []
        for k in keys:
            blob = http_get(FILE_URL + k)
            if blob:
                rows += parse(blob)
        rows = sorted(set(rows))
        print(f"{sym:10s} {len(keys):3d} 個月  {len(rows):6d} 筆  "
              f"{rows[0][0][:10] if rows else '-'} ~ {rows[-1][0][:10] if rows else '-'}", flush=True)
        allrows += [(sym, t, r) for t, r in rows]

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "time", "rate"])
        w.writerows(allrows)
    print(f"\n完成：{os.path.abspath(a.out)}（{os.path.getsize(a.out) / 1e6:.2f} MB）→ 上傳這個檔案")
    try:
        from google.colab import files
        files.download(a.out)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
