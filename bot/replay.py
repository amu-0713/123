#!/usr/bin/env python3
"""
歷史重播：用本機的歷史日 K 假裝成幣安 API，讓 bot 以紙上交易模式每天執行一次，
檢查實盤程式與回測（factor_research.py --rebal-days 6,3）是否一致。

用法：
  python bot/replay.py --data <download_binance.py 輸出的 1d 資料夾> --start 2024-01-01 --end 2025-06-30
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import engine  # noqa: E402
import run as runner  # noqa: E402


class FakeClient:
    """模擬幣安公開 API：只回傳 now 之前的資料（含當天尚未收盤的那根，與真實 API 相同）"""

    def __init__(self, data_dir):
        self.data = {}
        for f in glob.glob(os.path.join(data_dir, "*USDT.csv")):
            d = pd.read_csv(f)
            if d.empty:
                continue
            ts = (pd.to_datetime(d["open_time"], utc=True, format="mixed")
                  - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)        # 毫秒時間戳
            arr = np.column_stack([ts.values, d["high"].values, d["low"].values, d["close"].values,
                                   d["quote_volume"].values, d["taker_buy_quote"].values])
            self.data[os.path.basename(f)[:-4]] = arr
        self.now_ms = 0

    def set_now(self, now):
        self.now_ms = int(pd.Timestamp(now).value // 10 ** 6)

    def _upto(self, sym):
        a = self.data[sym]
        return a[a[:, 0] <= self.now_ms]

    def exchange_info(self):
        day = 86400000
        syms = []
        for s, a in self.data.items():
            past = a[a[:, 0] <= self.now_ms]
            if len(past) and past[-1, 0] >= self.now_ms - 2 * day:                    # 昨天還有日 K = 仍在交易
                syms.append({"symbol": s, "baseAsset": s[:-4], "quoteAsset": "USDT", "status": "TRADING",
                             "isSpotTradingAllowed": True,
                             "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.00000001", "minQty": "0"},
                                         {"filterType": "NOTIONAL", "minNotional": "5"}]})
        return {"symbols": syms}

    def klines(self, symbol, interval="1d", limit=220):
        a = self._upto(symbol)[-limit:]
        return [[int(r[0]), 0, r[1], r[2], r[3], 0, 0, r[4], 0, 0, r[5], 0] for r in a]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--out", default="replay_nav.csv")
    a = ap.parse_args(argv)

    fake = FakeClient(a.data)
    state = engine.new_state(a.capital)
    quiet = lambda m: None  # noqa: E731
    navs = []
    for day in pd.date_range(a.start, a.end, freq="D", tz="UTC"):
        now = day + pd.Timedelta(minutes=5)                     # 每天 UTC 00:05 執行
        fake.set_now(now)
        state = runner.run_once(fake, state, "paper", now, a.capital, idle_rate=0.02, fee=0.001, log=quiet)
        h = state["nav_history"][-1]
        navs.append(h)
        if day.day == 1 or day == pd.Timestamp(a.end, tz="UTC"):
            pos = {k: list(L["pos"]) for k, L in state["ledgers"].items()}
            print(f"{h['date']} 淨值 {h['nav']:,.0f}  縮放 {h['scale']:.0%}  持倉 {pos}", flush=True)
    nav = pd.DataFrame(navs).drop_duplicates("date").set_index("date")
    nav.to_csv(a.out)
    print(f"完成，淨值序列：{os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
