"""
匯出網站用的 JSON（給 my-stock-rank 網站的「加密貨幣」頁面）。

用法：
  python tools/export_web.py --data <download_binance.py 輸出的 data/spot/1d> --out <網站 repo>/public/crypto

輸出：
  summary.json    績效指標、逐年報酬、淨值與回撤（週取樣）、最近一次調倉持倉
  daily_nav.json  [[日期, 淨值], ...]，日期為日 K 收盤當天（UTC 00:00 = 台灣 08:00），給混合配置頁使用

只輸出報酬率與淨值指數，不含任何帳戶金額。
"""
import argparse
import contextlib
import io
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import factor_research as fr  # noqa: E402

# 與 docs/strategy.md 最終設定相同
CFG = dict(top_n=100, min_days=120, k=5, pool=20, high_n=60, high_th=0.9, rel_btc=30, exit_old=40, exit_new=20,
           new_weeks=2, combo={"lowvol_30": 0.5, "liq_30": 0.5}, vt=0.4, vt_lb=14, fee=0.001, idle=0.02,
           days=(6, 3), start="2018-06-01")


def run_strategy(P):
    c = P["close"]
    idx = c.index
    U = fr.universe(P, CFG["top_n"], CFG["min_days"])
    F = {}
    for n in CFG["combo"]:
        fr.ensure_factor(F, P, n)
    score = fr.composite(F, U, CFG["combo"])
    g0 = c >= CFG["high_th"] * c.rolling(CFG["high_n"]).max()
    rel = c.div(c["BTC"], axis=0)
    gate = (g0 & (rel >= rel.rolling(CFG["rel_btc"]).mean())).copy()
    gate["BTC"] = g0["BTC"]
    ma = lambda n: c >= c.rolling(n).mean()
    idle = pd.Series(CFG["idle"] / 365, index=idx)
    start = pd.Timestamp(CFG["start"], tz="UTC")
    parts, ledgers, first_dates = [], {}, None
    for name, wd in zip("AB", CFG["days"]):
        d0 = idx[idx.dayofweek == wd]
        d0 = d0[d0 >= start]
        d0 = d0[U.reindex(d0).sum(axis=1) >= CFG["k"] + 2]
        first_dates = d0 if first_dates is None else first_dates
        r, _, hold = fr.backtest(score, P, U, d0, CFG["k"], pd.Series(True, index=idx), CFG["fee"], gate, "fill",
                                 CFG["pool"], ma(CFG["exit_old"]), fallback="BTC", fallback_ok=g0["BTC"],
                                 exit_gate_new=ma(CFG["exit_new"]), new_weeks=CFG["new_weeks"], idle_daily=idle)
        parts.append(r)
        ledgers[name] = hold[-1]
    r = pd.concat(parts, axis=1, sort=True).loc[max(p.index[0] for p in parts):].fillna(0).mean(axis=1)
    r, scale = fr.vol_target(r, first_dates, CFG["vt"], CFG["vt_lb"], fee=CFG["fee"], idle_daily=idle)
    return r, ledgers, scale


def pct(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x) * 100, nd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    with contextlib.redirect_stdout(io.StringIO()):
        P = fr.load_panel(a.data)
    r, ledgers, scale = run_strategy(P)
    r = r.iloc[:-1]                                                         # 最後一根可能是未收盤的日 K
    btc = P["close_ff"]["BTC"].pct_change().reindex(r.index).fillna(0)
    close_date = lambda ix: (ix + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # 日 K 收盤的日期

    eq, eq_b = (1 + r).cumprod(), (1 + btc).cumprod()
    dd, dd_b = eq / eq.cummax() - 1, eq_b / eq_b.cummax() - 1
    m, mb = fr.metrics(r), fr.metrics(btc)
    yearly = lambda x: (1 + x).groupby(x.index.year).prod() - 1
    ys, yb = yearly(r), yearly(btc)

    wk = pd.Series(eq.index, index=eq.index).resample("W-SUN").last().dropna()   # 週取樣（每週最後一個交易日）：圖表用
    curve = [dict(date=(t + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                  strategy=round(float(eq.loc[:t].iloc[-1]), 4), btc=round(float(eq_b.loc[:t].iloc[-1]), 4),
                  dd=pct(dd.loc[:t].iloc[-7:].min(), 1), dd_btc=pct(dd_b.loc[:t].iloc[-7:].min(), 1)) for t in wk]

    def metric_block(mm):
        return dict(cagr=pct(mm["CAGR"]), mdd=pct(mm["MDD"]), vol=pct(mm["Vol"]), sharpe=round(float(mm["Sharpe"]), 2),
                    calmar=round(float(mm["Calmar"]), 2), month_win=pct(mm["MonthWin"], 1))

    holdings = {}
    for name, (t, s) in ledgers.items():
        holdings[name] = dict(date=(t + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                              rebalance_day="週日" if name == "A" else "週四",
                              positions=[dict(coin=x.split(":")[0], weight=round(float(x.split(":")[1]) * 100, 1))
                                         for x in s.split(",") if x])
    summary = dict(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        start_date=close_date(r.index[:1])[0], end_date=close_date(r.index[-1:])[0],
        method_note="2018-06 起回測。幣安 USDT 現貨成交額前 100 名（含已下市幣），單邊手續費加滑價 0.1%，"
                    "閒置資金年化 2%。兩本帳（週日、週四）錯開調倉，40% 波動目標。此為歷史回測，不代表未來報酬。",
        metrics=dict(strategy=metric_block(m), btc=metric_block(mb)),
        yearly=[dict(year=int(y), strategy=pct(ys[y], 1), btc=pct(yb[y], 1)) for y in ys.index],
        curve=curve,
        holdings=holdings,
        exposure_now=round(float(scale.dropna().iloc[-1]) * 100, 1) if scale.notna().any() else None,
    )
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, separators=(",", ":"))
    nav = [[d, round(float(v), 6)] for d, v in zip(close_date(eq.index), eq.values)]
    with open(os.path.join(a.out, "daily_nav.json"), "w", encoding="utf-8") as f:
        json.dump(nav, f, separators=(",", ":"))
    print(f"已輸出 {a.out}：{summary['start_date']} ~ {summary['end_date']}，年化 {summary['metrics']['strategy']['cagr']}%")


if __name__ == "__main__":
    main()
