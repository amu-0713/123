#!/usr/bin/env python3
"""
加密貨幣橫截面因子研究：單因子評估 → 因子相關性 → 組合回測。

用法：
    python factor_research.py --data data.zip                    # 10 幣小時 K（自動轉日線）
    python factor_research.py --data binance_spot_1d.zip         # 下載腳本產生的大幣池
    python factor_research.py --data data/spot/1d --top-n 50 --k 5
    python factor_research.py --combo "mom_28:1,taker_14:1,lowvol_90:1" --regime breadth
    Colab：整段貼進 cell，修改最下面 ARGS 後執行

流程（每週日收盤調倉，持有 7 天）：
    1. 幣池：上市滿 --min-days 天，且 30 日平均成交額排名前 --top-n
    2. 單因子：Rank IC、IC t 值、IC 逐年、前 k 名超額報酬、前後三分位價差
    3. 因子相關性：每期截面 rank 相關的平均
    4. 組合：各因子在幣池內轉成百分位分數後加權，取前 k 名等權，
       可加市場濾網，計入手續費，輸出 CAGR / Sharpe / Sortino / MDD / Calmar / 逐年報酬

所有因子方向都已調整成「分數越高，預期報酬越高」。
"""
import argparse
import glob
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ================================================================ 資料

def load_panel(path):
    """讀 zip 或資料夾裡所有 <SYMBOL>.csv，回傳日線面板 dict(close, high, low, qv, tbq)。"""
    frames = {}

    def read(name, fh):
        sym = os.path.basename(name)[:-4]
        if sym.startswith("_"):
            return
        d = pd.read_csv(fh, usecols=["open_time", "high", "low", "close", "quote_volume", "taker_buy_quote"])
        if d.empty:
            return
        d.index = pd.to_datetime(d.pop("open_time"), utc=True, format="mixed")
        frames[sym.replace("USDT", "")] = d

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.endswith(".csv"):
                    read(n, io.BytesIO(z.read(n)))
    else:
        for f in glob.glob(os.path.join(path, "**", "*.csv"), recursive=True):
            read(f, f)
    if not frames:
        sys.exit(f"找不到 CSV：{path}")

    agg = {"close": "last", "high": "max", "low": "min", "quote_volume": "sum", "taker_buy_quote": "sum"}
    daily = {s: d.resample("1D").agg(agg) for s, d in frames.items()}
    daily = {s: d[d.quote_volume > 0] for s, d in daily.items()}           # 去掉停牌日
    col = lambda c: pd.DataFrame({s: d[c] for s, d in daily.items()}).sort_index()
    P = {"close": col("close"), "high": col("high"), "low": col("low"),
         "qv": col("quote_volume"), "tbq": col("taker_buy_quote")}
    P["alive"] = P["close"].notna()
    # 回測計價用：停牌與下市後維持最後收盤（等同以最後價格出場），上市前保持 NaN
    P["close_ff"] = P["close"].ffill()
    return P


# ================================================================ 因子

def build_factors(P):
    c, qv, tbq = P["close"], P["qv"], P["tbq"]
    r = c.pct_change(fill_method=None)
    mkt = r.mean(axis=1)
    F = {}

    # --- 動能
    for n in [7, 28, 56, 90]:
        F[f"mom_{n}"] = c.pct_change(n, fill_method=None)
    F["mom_90_skip7"] = c.shift(7) / c.shift(90) - 1
    F["riskadj_mom_56"] = c.pct_change(56, fill_method=None) / r.rolling(56).std()

    # --- 短期反轉
    F["rev_3"] = -c.pct_change(3, fill_method=None)

    # --- 風險（低風險 = 高分）
    for n in [30, 90]:
        F[f"lowvol_{n}"] = -r.rolling(n).std()
    F["lowdownvol_60"] = -r.where(r < 0, 0).rolling(60).std()
    cov = r.rolling(90).cov(mkt)
    F["lowbeta_90"] = -cov.div(mkt.rolling(90).var(), axis=0)

    # --- 主動買占比
    ratio = lambda n: tbq.rolling(n).sum() / qv.rolling(n).sum()
    for n in [7, 14, 30]:
        F[f"taker_{n}"] = ratio(n)
    F["taker_14_demean"] = ratio(14) - ratio(180)                          # 扣掉幣種自身長期水準
    daily_ratio = tbq / qv
    F["taker_14_z"] = (ratio(14) - ratio(180)) / daily_ratio.rolling(180).std()   # 相對自己歷史的 z 分數

    # --- 長期反轉（價值替代）
    F["ltrev_365"] = -c.pct_change(365, fill_method=None)
    F["ath_drawdown"] = -(c / c.cummax())                                  # 距歷史高點跌越深分數越高

    # --- 量能 / 規模替代
    F["vol_growth"] = qv.rolling(7).mean() / qv.rolling(60).mean()
    F["small_size"] = -np.log(qv.rolling(30).mean())                       # 成交額小 ≈ 小市值
    F["illiq_amihud"] = (r.abs() / qv).rolling(30).mean()

    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items()}


# ================================================================ 幣池與調倉日

def universe(P, top_n, min_days):
    age = P["alive"].cumsum()
    liq = P["qv"].rolling(30, min_periods=20).mean()
    ok = P["alive"] & (age >= min_days) & liq.notna()
    rank = liq.where(ok).rank(axis=1, ascending=False)
    return rank <= top_n


def rebalance_dates(idx, freq):
    sun = idx[idx.dayofweek == 6]
    if freq == "2W":
        sun = sun[::2]
    elif freq == "M":
        sun = idx[idx.is_month_end]
    return sun


# ================================================================ 單因子評估

def evaluate_factors(F, P, U, dates, k):
    px = P["close_ff"]
    fwd = px.reindex(dates).shift(-1) / px.reindex(dates) - 1               # 到下一個調倉日的報酬
    rows, ic_series = [], {}
    for name, f in F.items():
        fv = f.reindex(dates).where(U.reindex(dates))
        ics, topx, spread = [], [], []
        for t in dates[:-1]:
            x, y = fv.loc[t].dropna(), fwd.loc[t]
            y = y[x.index].dropna()
            x = x[y.index]
            if len(x) < max(5, k + 2):
                continue
            ics.append((t, x.rank().corr(y.rank())))
            top = x.nlargest(k).index
            topx.append(y[top].mean() - y.mean())
            q = pd.qcut(x.rank(method="first"), 3, labels=False)
            spread.append(y[q == 2].mean() - y[q == 0].mean())
        if len(ics) < 20:
            continue
        s = pd.Series(dict(ics))
        ic_series[name] = s
        per_year = 365 / np.median(np.diff(dates).astype("timedelta64[D]").astype(int))
        rows.append(dict(
            factor=name, n=len(s), IC=s.mean(), IC_t=s.mean() / s.std() * np.sqrt(len(s)),
            IC_pos=(s > 0).mean(), ICIR_ann=s.mean() / s.std() * np.sqrt(per_year),
            topk_excess_ann=np.mean(topx) * per_year, tercile_spread_ann=np.mean(spread) * per_year,
            yrs_IC_pos=f"{(s.groupby(s.index.year).mean() > 0).sum()}/{s.index.year.nunique()}",
        ))
    summary = pd.DataFrame(rows).set_index("factor").sort_values("IC_t", ascending=False)
    by_year = pd.DataFrame({n: s.groupby(s.index.year).mean() for n, s in ic_series.items()}).T
    return summary, by_year.loc[summary.index]


def factor_correlation(F, U, dates, names):
    mats = []
    for t in dates:
        m = pd.DataFrame({n: F[n].loc[t] for n in names}).where(U.loc[t], axis=0).dropna()
        if len(m) >= 5:
            mats.append(m.rank().corr())
    return sum(mats) / len(mats)


# ================================================================ 組合回測

def composite(F, U, weights):
    score = None
    for name, w in weights.items():
        pct = F[name].where(U).rank(axis=1, pct=True)                      # 幣池內百分位 0~1
        score = pct * w if score is None else score.add(pct * w, fill_value=0)
    return score.where(U)


def regime_filter(P, U, kind, n=100):
    c = P["close"]
    if kind == "none":
        return pd.Series(True, index=c.index)
    if kind == "breadth":                                                   # 幣池中過半在 n 日均線上
        above = (c > c.rolling(n).mean()).astype(float).where(U)
        return above.sum(axis=1) / U.sum(axis=1).replace(0, np.nan) >= 0.5
    if kind == "ew":                                                        # 幣池等權指數在 n 日均線上
        r = c.pct_change(fill_method=None).where(U.shift(1, fill_value=False)).mean(axis=1).fillna(0)
        idx = (1 + r).cumprod()
        return idx > idx.rolling(n).mean()
    if kind == "btc":
        b = c["BTC"]
        return b > b.rolling(n).mean()
    raise ValueError(kind)


def backtest(score, P, U, dates, k, regime, fee):
    px = P["close_ff"]
    idx = px.index
    on = regime.reindex(dates).fillna(False)
    target = {}
    for t in dates:
        s = score.loc[t].dropna()
        target[t] = (pd.Series(1.0 / min(k, len(s)), index=s.nlargest(k).index)
                     if on.loc[t] and len(s) else pd.Series(dtype=float))

    rets = pd.Series(0.0, index=idx)
    turnover, w_prev, holdings = [], pd.Series(dtype=float), []
    for i, t in enumerate(dates):
        end = dates[i + 1] if i + 1 < len(dates) else idx[-1]
        w = target[t]
        tv = w.subtract(w_prev, fill_value=0).abs().sum()
        turnover.append(tv)
        holdings.append((t, ",".join(w.index)))
        seg = px.loc[t:end]
        if len(w) and len(seg) > 1:
            rel = seg[w.index] / seg[w.index].iloc[0]                       # 期間內權重自然漂移
            val = (rel * w).sum(axis=1)
            cash = 1 - w.sum()
            val = val + cash
            r = val.pct_change().iloc[1:]
            rets.loc[r.index] = r.values
            w_prev = (rel.iloc[-1] * w) / val.iloc[-1]
        else:
            w_prev = pd.Series(dtype=float)
        rets.loc[seg.index[1] if len(seg) > 1 else t] -= fee * tv
    rets = rets.loc[dates[0]:]
    return rets, pd.Series(turnover, index=dates), holdings


def metrics(r, ann=365):
    eq = (1 + r).cumprod()
    yrs = len(r) / ann
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    dn = r[r < 0].std() * np.sqrt(ann)
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    return dict(CAGR=cagr, Vol=r.std() * np.sqrt(ann), Sharpe=r.mean() / r.std() * np.sqrt(ann),
                Sortino=r.mean() * ann / dn if dn else np.nan, MDD=mdd,
                Calmar=cagr / -mdd if mdd < 0 else np.nan,
                MonthWin=((1 + r).resample("ME").prod() - 1 > 0).mean(),
                years=" ".join(f"{y}:{v:+.0%}" for y, v in yearly.items()))


def parse_combo(s):
    return {p.split(":")[0].strip(): float(p.split(":")[1]) if ":" in p else 1.0 for p in s.split(",")}


# ================================================================ main

DEFAULT_COMBOS = [
    "taker_14:1",
    "mom_28:1",
    "taker_14:1,mom_28:1",
    "taker_14:1,lowvol_90:1",
    "taker_14:1,mom_28:1,lowvol_90:1",
    "taker_14:1,lowbeta_90:1",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data.zip", help="zip 檔或資料夾")
    ap.add_argument("--top-n", type=int, default=30, help="幣池：成交額前 N 名")
    ap.add_argument("--min-days", type=int, default=120, help="幣池：至少上市天數")
    ap.add_argument("--k", type=int, default=3, help="持有前 k 名")
    ap.add_argument("--freq", default="W", choices=["W", "2W", "M"])
    ap.add_argument("--regime", default="breadth", choices=["none", "breadth", "ew", "btc"])
    ap.add_argument("--fee", type=float, default=0.001, help="單邊手續費 + 滑價")
    ap.add_argument("--start", default="2020-06-01")
    ap.add_argument("--combo", action="append", help='例如 "mom_28:1,taker_14:1"，可重複給多組')
    ap.add_argument("--out", default="factor_out")
    a, _ = ap.parse_known_args(argv)

    os.makedirs(a.out, exist_ok=True)
    P = load_panel(a.data)
    F = build_factors(P)
    U = universe(P, a.top_n, a.min_days)
    dates = rebalance_dates(P["close"].index, a.freq)
    dates = dates[(dates >= pd.Timestamp(a.start, tz="UTC"))]
    dates = dates[U.reindex(dates).sum(axis=1) >= max(5, a.k + 2)]

    print(f"資料：{len(P['close'].columns)} 個幣，{P['close'].index[0]:%Y-%m-%d} ~ {P['close'].index[-1]:%Y-%m-%d}")
    print(f"幣池：成交額前 {a.top_n}、上市 ≥{a.min_days} 天；平均 {U.reindex(dates).sum(axis=1).mean():.1f} 個幣；"
          f"調倉 {a.freq}，共 {len(dates)} 期\n")

    # 1. 單因子
    summ, by_year = evaluate_factors(F, P, U, dates, a.k)
    print("=" * 30, "單因子（依 IC t 值排序）", "=" * 30)
    print(summ.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n--- 逐年 IC")
    print(by_year.to_string(float_format=lambda x: f"{x:+.3f}"))
    summ.to_csv(f"{a.out}/factor_summary.csv")
    by_year.to_csv(f"{a.out}/factor_ic_by_year.csv")

    # 2. 相關性（取 IC t 前 10 名）
    top = list(summ.index[:10])
    corr = factor_correlation(F, U, dates, top)
    print("\n" + "=" * 30, "因子相關性（前 10 名）", "=" * 30)
    print(corr.to_string(float_format=lambda x: f"{x:+.2f}"))
    corr.to_csv(f"{a.out}/factor_corr.csv")

    # 3. 組合回測
    combos = a.combo or DEFAULT_COMBOS
    reg = regime_filter(P, U, a.regime)
    ew = P["close_ff"].pct_change().where(U.shift(1, fill_value=False)).mean(axis=1).fillna(0).loc[dates[0]:]
    rows = [dict(combo=f"[基準] 幣池等權持有", **metrics(ew))]
    daily = {"benchmark_ew": ew}
    for cs in combos:
        w = parse_combo(cs)
        missing = [n for n in w if n not in F]
        if missing:
            print(f"略過 {cs}：未知因子 {missing}")
            continue
        r, tv, hold = backtest(composite(F, U, w), P, U, dates, a.k, reg, a.fee)
        m = metrics(r)
        m["turnover_ann"] = tv.mean() * len(dates) / (len(r) / 365)
        rows.append(dict(combo=cs, **m))
        daily[cs] = r
        pd.DataFrame(hold, columns=["date", "holdings"]).to_csv(
            f"{a.out}/holdings_{cs.replace(':', '').replace(',', '_')}.csv", index=False)
    res = pd.DataFrame(rows).set_index("combo")
    print("\n" + "=" * 30, f"組合回測  top{a.k} {a.freq}  濾網={a.regime}  手續費={a.fee:.2%}", "=" * 30)
    fmt = res.drop(columns="years").copy()
    for c in ["CAGR", "Vol", "MDD", "MonthWin"]:
        fmt[c] = fmt[c].map(lambda x: f"{x:.0%}")
    print(fmt.to_string(float_format=lambda x: f"{x:.2f}"))
    print("\n--- 逐年報酬")
    for name, y in res["years"].items():
        print(f"{name:40s} {y}")
    res.to_csv(f"{a.out}/backtest_summary.csv")
    pd.DataFrame(daily).to_csv(f"{a.out}/backtest_daily_returns.csv")
    print(f"\n結果已存到 {os.path.abspath(a.out)}/")


if __name__ == "__main__":
    main()
