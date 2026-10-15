#!/usr/bin/env python3
"""
批次下載 Binance 現貨 K 線（含已下市幣種），來源：data.binance.vision 官方公開資料。

只用 Python 標準函式庫，不需要 API key。

用法：
    Colab：整段貼進一個 cell 執行，跑完會自動下載 binance_spot_1d.zip
    python download_binance.py                       # 全部 USDT 現貨、日 K，輸出到 ./data/spot/1d/
    python download_binance.py --interval 1h         # 小時 K（檔案大很多）
    python download_binance.py --symbols BTCUSDT ETHUSDT
    python download_binance.py --workers 16 --out ./data

輸出：
    <out>/spot/<interval>/<SYMBOL>.csv   欄位與 data.zip 相同：
        open_time,open,high,low,close,volume,quote_volume,trades,taker_buy_base,taker_buy_quote
    <out>/spot/<interval>/_symbols.csv   每個幣的起訖日期、筆數、是否已停止更新（疑似下市）
    <out>/_cache/                        原始 zip 快取，中斷後重跑會跳過已下載的檔案
    ./binance_spot_<interval>.zip        跑完自動打包（不含快取），直接上傳這個檔（--no-zip 可關閉）

注意：
    * 2025-01-01 起現貨資料的時間戳改為微秒，這裡會自動判斷。
    * 預設排除穩定幣、法幣、槓桿代幣（xxxUP/xxxDOWN/BULL/BEAR）；被排除的清單會印出來。
"""
import argparse
import csv
import datetime as dt
import io
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILE_URL = "https://data.binance.vision/"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# 穩定幣 / 法幣 / 錨定資產，對 USDT 幾乎沒有波動，放進幣池會干擾橫截面因子
EXCLUDE_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USDS", "USDSB", "SUSD",
    "UST", "USTC", "USDE", "USD1", "RLUSD", "XUSD", "BFUSD", "AEUR", "EURI", "EUR",
    "GBP", "AUD", "BRL", "TRY", "RUB", "UAH", "NGN", "BIDR", "IDRT", "BKRW", "BVND",
    "PAXG", "WBTC", "WBETH", "BETH",
}
LEVERAGED_SUFFIX = ("UP", "DOWN", "BULL", "BEAR")

OUT_HEADER = ["open_time", "open", "high", "low", "close", "volume",
              "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]


# ---------------------------------------------------------------- HTTP

def http_get(url, retries=5, timeout=60):
    """回傳 bytes；404 回傳 None；其他錯誤重試。"""
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kline-downloader/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:  # 網路斷線、timeout 等
            err = e
        time.sleep(2 ** i)
    raise RuntimeError(f"下載失敗 {url}: {err}")


def s3_list(prefix):
    """列出 S3 prefix 底下的子目錄與檔案（自動翻頁）。"""
    dirs, files, marker = [], [], ""
    while True:
        q = {"delimiter": "/", "prefix": prefix}
        if marker:
            q["marker"] = marker
        body = http_get(LIST_URL + "?" + urllib.parse.urlencode(q))
        if body is None:
            break
        root = ET.fromstring(body)
        page_dirs = [e.text for e in root.iter(NS + "Prefix") if e.text != prefix]
        page_files = [e.text for e in root.iter(NS + "Key")]
        dirs += page_dirs
        files += page_files
        if root.findtext(NS + "IsTruncated") != "true":
            break
        marker = root.findtext(NS + "NextMarker") or max(page_dirs + page_files)
    return dirs, files


# ---------------------------------------------------------------- symbols

def list_symbols(quote):
    dirs, _ = s3_list("data/spot/monthly/klines/")
    syms = sorted({d.rstrip("/").split("/")[-1] for d in dirs})
    return [s for s in syms if s.endswith(quote) and len(s) > len(quote)]


def filter_symbols(syms, quote):
    bases = {s[: -len(quote)] for s in syms}
    keep, dropped = [], []
    for s in syms:
        b = s[: -len(quote)]
        lev = any(b.endswith(x) and b[: -len(x)] in bases for x in LEVERAGED_SUFFIX)
        if b in EXCLUDE_BASES or lev:
            dropped.append(s)
        else:
            keep.append(s)
    return keep, dropped


# ---------------------------------------------------------------- parse

def to_iso(ts):
    ts = int(ts)
    if ts > 10 ** 14:      # 微秒（2025 起現貨）
        ts //= 1000
    return dt.datetime.fromtimestamp(ts / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S+00:00")


def parse_zip(blob):
    rows = []
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            text = z.read(name).decode()
            for line in text.splitlines():
                p = line.split(",")
                if not p or not p[0].strip().isdigit():   # 跳過可能存在的表頭
                    continue
                # open_time,o,h,l,c,vol,close_time,quote_vol,trades,taker_base,taker_quote,ignore
                rows.append([to_iso(p[0]), p[1], p[2], p[3], p[4], p[5], p[7], p[8], p[9], p[10]])
    return rows


# ---------------------------------------------------------------- per-symbol download

def cached_get(url, cache_dir):
    path = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    blob = http_get(url)
    if blob is None:
        return None
    try:
        zipfile.ZipFile(io.BytesIO(blob)).testzip()
    except zipfile.BadZipFile:
        raise RuntimeError(f"zip 損毀 {url}")
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, path)
    return blob


def download_symbol(sym, interval, out_dir, cache_root, today):
    cache_dir = os.path.join(cache_root, interval, sym)
    os.makedirs(cache_dir, exist_ok=True)

    _, keys = s3_list(f"data/spot/monthly/klines/{sym}/{interval}/")
    zips = sorted(k for k in keys if k.endswith(".zip"))
    if not zips:
        return sym, 0, None, None

    rows = []
    for k in zips:
        blob = cached_get(FILE_URL + k, cache_dir)
        if blob:
            rows += parse_zip(blob)

    # 最後一個月度檔之後的日檔（只針對還在交易的幣）
    last_month = zips[-1].rsplit("-", 2)[-2:]          # ['YYYY', 'MM.zip']
    y, m = int(last_month[0]), int(last_month[1][:2])
    next_month = dt.date(y + (m == 12), m % 12 + 1, 1)
    if (today - next_month).days < 40:                 # 最近一個月仍有資料 → 視為仍在交易
        d = next_month
        while d < today:
            url = f"{FILE_URL}data/spot/daily/klines/{sym}/{interval}/{sym}-{interval}-{d:%Y-%m-%d}.zip"
            blob = cached_get(url, cache_dir)
            if blob:
                rows += parse_zip(blob)
            d += dt.timedelta(days=1)

    rows = sorted({r[0]: r for r in rows}.values())    # 去重、排序
    with open(os.path.join(out_dir, f"{sym}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUT_HEADER)
        w.writerows(rows)
    return sym, len(rows), rows[0][0][:10], rows[-1][0][:10]


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval", default="1d", help="1d / 4h / 1h ...（預設 1d）")
    ap.add_argument("--quote", default="USDT")
    ap.add_argument("--symbols", nargs="*", help="只下載指定幣種，例如 BTCUSDT ETHUSDT")
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-all", action="store_true", help="不排除穩定幣與槓桿代幣")
    ap.add_argument("--no-zip", action="store_true", help="跑完不要打包成 zip")
    a, _ = ap.parse_known_args()   # parse_known_args：在 Colab / Jupyter 貼上直接跑也不會報錯

    out_dir = os.path.join(a.out, "spot", a.interval)
    cache_root = os.path.join(a.out, "_cache")
    os.makedirs(out_dir, exist_ok=True)

    if a.symbols:
        syms = [s.upper() for s in a.symbols]
    else:
        print("列出幣種中…", flush=True)
        syms = list_symbols(a.quote)
        if not a.keep_all:
            syms, dropped = filter_symbols(syms, a.quote)
            print(f"排除 {len(dropped)} 個：{' '.join(dropped)}")
    print(f"共 {len(syms)} 個幣種，interval={a.interval}，輸出到 {out_dir}", flush=True)

    today = dt.datetime.now(dt.timezone.utc).date()
    summary, failed = [], []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(download_symbol, s, a.interval, out_dir, cache_root, today): s for s in syms}
        for i, fu in enumerate(as_completed(futs), 1):
            s = futs[fu]
            try:
                sym, n, first, last = fu.result()
                summary.append((sym, n, first, last))
                print(f"[{i}/{len(syms)}] {sym:14s} {n:6d} 筆  {first} ~ {last}", flush=True)
            except Exception as e:
                failed.append(s)
                print(f"[{i}/{len(syms)}] {s:14s} 失敗：{e}", file=sys.stderr, flush=True)

    cutoff = (today - dt.timedelta(days=7)).isoformat()
    with open(os.path.join(out_dir, "_symbols.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "rows", "first", "last", "delisted"])
        for sym, n, first, last in sorted(summary):
            w.writerow([sym, n, first, last, int(bool(last) and last < cutoff)])

    print(f"\n完成：{len(summary)} 成功，{len(failed)} 失敗")
    if failed:
        print("失敗的幣種（重跑會從快取續傳）：", " ".join(failed))

    if not a.no_zip:
        zip_path = f"binance_spot_{a.interval}.zip"
        base = os.path.dirname(os.path.abspath(a.out))     # zip 內路徑為 data/spot/<interval>/...
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for name in sorted(os.listdir(out_dir)):
                p = os.path.join(out_dir, name)
                z.write(p, os.path.relpath(os.path.abspath(p), base))
        print(f"已打包：{os.path.abspath(zip_path)}（{os.path.getsize(zip_path) / 1e6:.1f} MB）→ 上傳這個檔案即可")
        try:                                               # 在 Colab 上自動下載到本機
            from google.colab import files
            files.download(zip_path)
        except ImportError:
            pass


if __name__ == "__main__":
    main()
