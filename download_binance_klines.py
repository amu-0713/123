"""
從幣安公開資料網站 (data.binance.vision) 下載歷史 K 線，不需要帳號或 API 金鑰。

用法：
    python download_binance_klines.py
    （在 Colab 或 Jupyter 裡也可以直接整段貼上執行）

輸出：
    data/<MARKET>/<INTERVAL>/<SYMBOL>.csv
    欄位：open_time(UTC), open, high, low, close, volume, quote_volume, trades,
          taker_buy_base, taker_buy_quote

需要：pandas（Colab 已內建）
"""

import io
import os
import time
import zipfile
import urllib.request
import urllib.error
from datetime import date, timedelta

import pandas as pd

# ======== 設定 ========
# 要下載的幣。注意：這份清單是「現在」的大幣，有倖存者偏差。
# 做正式回測時，請另外加入當時存在、後來下市或崩盤的幣。
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]

# "spot" = 現貨；"futures/um" = USDT 本位永續合約（要研究槓桿就用這個）
MARKET = "spot"

# K 線週期。先抓 1h，之後可以自己合成 4h、1d。
# 可用值例如 "1m" "5m" "15m" "1h" "4h" "1d"
INTERVAL = "1h"

# 起始月份（含）
START = "2020-01"

# 是否補抓本月到昨天的每日檔（月檔要等該月結束才會出現）
INCLUDE_CURRENT_MONTH = True

OUT_DIR = "data"
CACHE_DIR = "cache_zip"   # 下載過的 zip 會存這裡，重跑時不會重抓
BASE = "https://data.binance.vision/data"
# ======================

COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def month_range(start: str, end_exclusive: date):
    y, m = map(int, start.split("-"))
    while date(y, m, 1) < end_exclusive:
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


def fetch_zip(url: str, cache_path: str):
    """下載 zip（有快取就讀快取）。檔案不存在時回傳 None。"""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return f.read()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None          # 該月份沒有資料（還沒上市、已下市，或檔案尚未產生）
            print(f"  HTTP {e.code}，重試中：{url}")
        except Exception as e:
            print(f"  連線錯誤（{e}），重試中：{url}")
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"下載失敗：{url}")


def read_zip(data: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None)
    # 部分檔案第一列是欄位名稱，偵測到就丟掉
    if not str(df.iloc[0, 0]).lstrip("-").isdigit():
        df = df.iloc[1:]
    df = df.iloc[:, :12]
    df.columns = COLS
    return df


def to_utc(ts: pd.Series) -> pd.Series:
    """幣安現貨資料自 2025 年起時間戳改為微秒，之前是毫秒；這裡自動判斷。"""
    ts = ts.astype("int64")
    unit_us = ts > 10**14
    out = pd.to_datetime(ts.where(~unit_us, ts // 1000), unit="ms", utc=True)
    return out


def download_symbol(symbol: str) -> pd.DataFrame:
    today = date.today()
    first_of_month = today.replace(day=1)
    frames = []

    # 月檔
    for y, m in month_range(START, first_of_month):
        fname = f"{symbol}-{INTERVAL}-{y}-{m:02d}.zip"
        url = f"{BASE}/{MARKET}/monthly/klines/{symbol}/{INTERVAL}/{fname}"
        data = fetch_zip(url, os.path.join(CACHE_DIR, MARKET, symbol, fname))
        if data is not None:
            frames.append(read_zip(data))

    # 本月的日檔
    if INCLUDE_CURRENT_MONTH:
        d = first_of_month
        while d < today:
            fname = f"{symbol}-{INTERVAL}-{d.isoformat()}.zip"
            url = f"{BASE}/{MARKET}/daily/klines/{symbol}/{INTERVAL}/{fname}"
            data = fetch_zip(url, os.path.join(CACHE_DIR, MARKET, symbol, fname))
            if data is not None:
                frames.append(read_zip(data))
            d += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["open_time"] = to_utc(df["open_time"])
    num_cols = ["open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base", "taker_buy_quote"]
    df[num_cols] = df[num_cols].astype(float)
    df["trades"] = df["trades"].astype("int64")
    df = (df.drop(columns=["close_time", "ignore"])
            .drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True))
    return df


def main():
    out_dir = os.path.join(OUT_DIR, MARKET.replace("/", "_"), INTERVAL)
    os.makedirs(out_dir, exist_ok=True)
    for sym in SYMBOLS:
        print(f"下載 {sym} ...")
        df = download_symbol(sym)
        if df.empty:
            print(f"  {sym}：沒有資料（代號錯誤或該市場沒有這個幣）")
            continue
        path = os.path.join(out_dir, f"{sym}.csv")
        df.to_csv(path, index=False)
        # 簡單檢查缺漏：實際 K 線數 vs 理論 K 線數
        step = pd.Timedelta(INTERVAL.replace("m", "min"))
        expected = int((df["open_time"].iloc[-1] - df["open_time"].iloc[0]) / step) + 1
        print(f"  {sym}：{len(df)} 根 K 線，"
              f"{df['open_time'].iloc[0]:%Y-%m-%d} ~ {df['open_time'].iloc[-1]:%Y-%m-%d}，"
              f"缺 {expected - len(df)} 根 → {path}")
    print("完成")


if __name__ == "__main__":
    main()
