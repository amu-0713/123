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

    frames = clean(frames)
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


EXCLUDE = {"BULL", "BEAR"}            # 舊的槓桿代幣（下載腳本的過濾抓不到沒有前綴的這兩個）


def clean(frames, max_gap_days=7, redenom_tol=0.1):
    """
    1. 排除槓桿代幣
    2. 交易中斷超過 max_gap_days 天 → 視為重新上架的不同資產，拆成 SYM、SYM~2 …（例：LUNA 崩盤後的新 LUNA）
    3. 單日價格剛好跳 10 / 100 / 1000 倍 → 視為改面額，把之前的價格按比例調整（例：QUICK、SUN、BNX、COCOS）
    """
    out, log = {}, []
    for sym, d in frames.items():
        if sym in EXCLUDE:
            log.append(f"排除 {sym}")
            continue
        d = d.sort_index()
        dvol = d["close"].resample("1D").last().pct_change().std()
        if dvol < 0.01:                   # 日波動 <1%：穩定幣、法幣幣、代幣化股票（例：U、KGST、SPYB）
            log.append(f"排除 {sym}（日波動 {dvol:.2%}，非一般加密貨幣）")
            continue
        gaps = d.index.to_series().diff().dt.days
        cuts = list(gaps[gaps > max_gap_days].index)
        bounds = [0] + [d.index.get_loc(t) for t in cuts] + [len(d)]
        segs = [d.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
        for i, seg in enumerate(segs):
            seg = seg.copy()
            ratio = seg["close"] / seg["close"].shift(1)
            lg = np.log10(ratio)
            jumps = lg[(lg.abs() >= 0.9) & ((lg - lg.round()).abs() < redenom_tol)]
            for t, v in jumps.items():
                k = 10.0 ** round(v)
                seg.loc[seg.index < t, ["close", "high", "low"]] *= k
                log.append(f"{sym} {t:%Y-%m-%d} 改面額 ×{k:g}")
            name = sym if i == 0 else f"{sym}~{i + 1}"
            if i:
                log.append(f"{sym} {seg.index[0]:%Y-%m-%d} 中斷後重新上架 → {name}")
            out[name] = seg
    if log:
        print("資料清洗：\n  " + "\n  ".join(log))
    return out


# ================================================================ 因子

def build_factors(P):
    """因子庫。名稱即定義的方向（例如 liq_level 越大 = 越活絡），好壞由 IC 正負判斷。"""
    c, h, l, qv, tbq = P["close"], P["high"], P["low"], P["qv"], P["tbq"]
    r = c.pct_change(fill_method=None)
    lp = np.log(c)
    mkt = r.mean(axis=1)
    F = {}

    # ---------------- 動能
    for n in [7, 14, 28, 56, 90, 180, 365]:
        F[f"mom_{n}"] = c.pct_change(n, fill_method=None)
    F["mom_90_skip7"] = c.shift(7) / c.shift(90) - 1
    F["mom_365_skip28"] = c.shift(28) / c.shift(365) - 1
    F["riskadj_mom_56"] = c.pct_change(56, fill_method=None) / r.rolling(56).std()
    beta = r.rolling(90).cov(mkt).div(mkt.rolling(90).var(), axis=0)
    resid = r - beta.shift(1).mul(mkt, axis=0)                               # 扣掉市場 beta 後的報酬
    for n in [28, 56, 90]:
        F[f"resmom_{n}"] = resid.rolling(n).sum()
    F["updays_60"] = (r > 0).astype(float).where(r.notna()).rolling(60).mean()   # 溫水煮青蛙：漲的天數比例
    t = pd.Series(np.arange(len(c)), index=c.index, dtype=float)
    F["trend_corr_90"] = lp.rolling(90).corr(t)                             # 價格走勢的直線程度（帶方向）
    F["ma_dist_50"] = c / c.rolling(50).mean() - 1
    F["ma_dist_200"] = c / c.rolling(200).mean() - 1
    F["high_365"] = c / c.rolling(365, min_periods=120).max()               # 距一年高點
    F["ath_ratio"] = c / c.cummax()                                         # 距歷史高點

    # ---------------- 短期反轉
    F["ret_1"] = r
    F["ret_3"] = c.pct_change(3, fill_method=None)

    # ---------------- 風險
    for n in [30, 90]:
        F[f"lowvol_{n}"] = -r.rolling(n).std()
    F["lowdownvol_60"] = -r.where(r < 0, 0).rolling(60).std()
    F["lowbeta_90"] = -beta
    F["lowidio_90"] = -resid.rolling(90).std()                              # 低特質波動
    F["max_ret_30"] = r.rolling(30).max()                                   # 樂透性：單日最大漲幅
    F["skew_90"] = r.rolling(90).skew()
    F["range_30"] = np.log(h / l).rolling(30).mean()                        # 日內高低價差

    # ---------------- 流動性 / 規模
    F["liq_level"] = np.log(qv.rolling(30).mean())                          # 成交額（規模替代）
    F["amihud_illiq"] = (r.abs() / qv).rolling(30).mean()
    F["liq_trend_30_180"] = qv.rolling(30).mean() / qv.rolling(180).mean()  # 成交額長期變化
    F["vol_growth_7_60"] = qv.rolling(7).mean() / qv.rolling(60).mean()     # 短期爆量
    F["vol_cv_30"] = qv.rolling(30).std() / qv.rolling(30).mean()           # 成交額不穩定度
    F["vol_rank_change"] = np.log(qv.rolling(30).mean()).rank(axis=1, pct=True).diff(28)   # 成交額排名變化

    # ---------------- 量價
    dlv = np.log(qv).diff()
    F["pv_corr_30"] = r.rolling(30).corr(dlv)                               # 量價相關
    F["upvol_share_30"] = qv.where(r > 0, 0).rolling(30).sum() / qv.rolling(30).sum()   # 上漲日成交額占比
    F["vw_ret_30"] = (r * qv).rolling(30).sum() / qv.rolling(30).sum()      # 成交額加權報酬
    F["clv_14"] = ((2 * c - h - l) / (h - l)).rolling(14).mean()            # 收盤在日內區間的位置
    F["attention_dir"] = F["vol_growth_7_60"] * np.sign(F["mom_7"])         # 爆量配方向

    # ---------------- 主動買占比
    ratio = lambda n: tbq.rolling(n).sum() / qv.rolling(n).sum()
    for n in [7, 14, 30]:
        F[f"taker_{n}"] = ratio(n)
    F["taker_14_demean"] = ratio(14) - ratio(180)

    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in F.items()}


def ensure_factor(F, P, name):
    """
    依名稱動態產生參數化因子（已存在就直接用）：
      lowvol_N      N 日報酬標準差取負（越穩越高分）
      lowdownvol_N  N 日下行波動取負
      liq_N         log(N 日平均成交額)
      liqmed_N      log(N 日成交額中位數)，較不受單日爆量影響
    """
    if name in F:
        return F[name]
    import re
    m = re.fullmatch(r"(lowvol|lowdownvol|liq|liqmed)_(\d+)", name)
    if not m:
        raise KeyError(f"未知因子：{name}")
    kind, n = m.group(1), int(m.group(2))
    r = P["close"].pct_change(fill_method=None)
    if kind == "lowvol":
        f = -r.rolling(n, min_periods=max(5, n * 2 // 3)).std()
    elif kind == "lowdownvol":
        f = -r.where(r < 0, 0).rolling(n, min_periods=max(5, n * 2 // 3)).std()
    elif kind == "liq":
        f = np.log(P["qv"].rolling(n, min_periods=max(3, n * 2 // 3)).mean())
    else:
        f = np.log(P["qv"].rolling(n, min_periods=max(3, n * 2 // 3)).median())
    F[name] = f.replace([np.inf, -np.inf], np.nan)
    return F[name]


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

def _residualize(x, Z):
    """x、Z 為截面 rank，回傳 x 對 Z 回歸後的殘差。"""
    A = np.column_stack([np.ones(len(x)), Z])
    beta, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ beta


def evaluate_factors(F, P, U, dates, k, neutral=()):
    """neutral：控制變數因子名稱，額外計算扣掉它們之後的 IC（看是否提供新資訊）。"""
    px = P["close_ff"]
    fwd = px.reindex(dates).shift(-1) / px.reindex(dates) - 1               # 到下一個調倉日的報酬
    rows, ic_series = [], {}
    for name, f in F.items():
        fv = f.reindex(dates).where(U.reindex(dates))
        nv = [F[n].reindex(dates) for n in neutral if n != name]
        ics, nics, topx, spread = [], [], [], []
        for t in dates[:-1]:
            x, y = fv.loc[t].dropna(), fwd.loc[t]
            y = y[x.index].dropna()
            x = x[y.index]
            if len(x) < max(5, k + 2):
                continue
            ics.append((t, x.rank().corr(y.rank())))
            if nv:
                Z = pd.DataFrame({i: z.loc[t, x.index] for i, z in enumerate(nv)}).dropna()
                if len(Z) >= 10:
                    xr = _residualize(x[Z.index].rank().values, Z.rank().values)
                    nics.append(pd.Series(xr).corr(pd.Series(y[Z.index].rank().values)))
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
            **({"nIC": np.mean(nics), "nIC_t": np.mean(nics) / np.std(nics) * np.sqrt(len(nics))} if nics else {}),
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


def trend_signal(P, spec):
    """
    各幣自己的趨勢開關（True = 多頭）：
      sma100   收盤 > 100 日均線
      mom60    60 日報酬 > 0
      donch60  收盤在 60 日高點的 10% 以內
    """
    c = P["close"]
    kind, n = spec.rstrip("0123456789"), int(spec[len(spec.rstrip("0123456789")):])
    if kind == "sma":
        return c > c.rolling(n).mean()
    if kind == "mom":
        return c.pct_change(n, fill_method=None) > 0
    if kind == "donch":
        return c >= 0.9 * c.rolling(n).max()
    raise ValueError(spec)


def backtest(score, P, U, dates, k, regime, fee, gate=None, gate_mode="slot", pool=None, exit_gate=None,
             entry_ok=None, fallback=None, fallback_ok=None):
    """
    gate：各幣趨勢開關（DataFrame bool）
      gate_mode="slot"   先依分數選前 k 名，趨勢不成立的那一格改持有現金（每格固定 1/k）
      gate_mode="filter" 只在趨勢成立的幣裡選前 k 名，資金平均分配（有幣就滿倉）
      gate_mode="fill"   只在分數前 pool 名裡，依分數順序挑趨勢成立的幣補滿 k 格（每格固定 1/k，不足的格子持有現金）
    exit_gate：每日出場條件（DataFrame bool，False = 出場）。持有中的幣某天收盤不成立就以當天收盤賣出，
      該格改持有現金到下個調倉日；進場仍只在調倉日。None = 只在調倉日檢查。
    entry_ok：Series bool。調倉日為 False 時不買進新幣，只保留上一期仍持有、且仍符合條件的幣。
    fallback："BTC" 或 "top1"（當期分數第一名）。fallback_ok 為 True 的調倉日，空出來的格子改買這個幣，
      不放現金（該幣可能因此超過 1/k）。
    """
    px = P["close_ff"]
    idx = px.index
    on = regime.reindex(dates).fillna(False)
    eok = entry_ok.reindex(dates).fillna(False) if entry_ok is not None else None
    fok = fallback_ok.reindex(dates).fillna(False) if fallback_ok is not None else None

    rets = pd.Series(0.0, index=idx)
    turnover, w_prev, holdings = [], pd.Series(dtype=float), []
    held = set()                                                            # 上一期結束時仍持有的幣
    for i, t in enumerate(dates):
        # ---- 決定目標權重
        s = score.loc[t].dropna()
        if gate is not None and gate_mode == "fill":
            s = s.nlargest(pool or k)
            s = s[gate.loc[t].reindex(s.index).fillna(False).astype(bool)]
        if gate is not None and gate_mode == "filter":
            s = s[gate.loc[t].reindex(s.index).fillna(False).astype(bool)]
        if eok is not None and not eok.loc[t]:
            s = s[s.index.isin(held)]                                       # 市場不佳：不買新幣
        pick = s.nlargest(k).index
        if not on.loc[t] or not len(pick):
            w = pd.Series(dtype=float)
        elif gate is not None and gate_mode in ("slot", "fill"):
            keep = [x for x in pick if gate.loc[t, x]]
            w = pd.Series(1.0 / k, index=keep)
        else:
            w = pd.Series(1.0 / len(pick), index=pick)
        if fallback and fok is not None and fok.loc[t] and w.sum() < 1 - 1e-9:
            fb = "BTC" if fallback == "BTC" else (score.loc[t].dropna().idxmax() if score.loc[t].notna().any() else None)
            if fb is not None and pd.notna(px.loc[t].get(fb)):
                w = w.add(pd.Series({fb: 1 - w.sum()}), fill_value=0)

        # ---- 持有到下個調倉日
        end = dates[i + 1] if i + 1 < len(dates) else idx[-1]
        tv = w.subtract(w_prev, fill_value=0).abs().sum()
        turnover.append(tv)
        holdings.append((t, ",".join(f"{c}:{v:.2f}" for c, v in w.items())))  # 幣:權重
        seg = px.loc[t:end]
        exited = []
        if len(w) and len(seg) > 1:
            rel = seg[w.index] / seg[w.index].iloc[0]                       # 期間內權重自然漂移
            if exit_gate is not None:
                eg = exit_gate.reindex(index=seg.index[1:], columns=w.index).fillna(True).astype(bool)
                for col in w.index:
                    off = eg.index[~eg[col].values]
                    if len(off):                                            # 當天收盤賣出，之後凍結為現金
                        d = off[0]
                        rel.loc[rel.index > d, col] = rel.loc[d, col] * (1 - fee)
                        exited.append(col)
            val = (rel * w).sum(axis=1) + (1 - w.sum())
            r = val.pct_change().iloc[1:]
            rets.loc[r.index] = r.values
            w_prev = (rel.iloc[-1] * w).drop(exited) / val.iloc[-1]
        else:
            w_prev = pd.Series(dtype=float)
        held = set(w.index) - set(exited)
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
    ap.add_argument("--trend", default="", help="各幣趨勢開關，例如 sma100 / mom60 / donch60（空字串 = 不用）")
    ap.add_argument("--trend-mode", default="slot", choices=["slot", "filter", "fill"])
    ap.add_argument("--pool", type=int, default=0, help="fill 模式：只在分數前 N 名裡遞補")
    ap.add_argument("--rel-btc", type=int, default=0,
                    help="相對強度：非 BTC 的幣還需「幣價/BTC」站上 N 日均線才可進場（0 = 不用）")
    ap.add_argument("--fallback", default="", choices=["", "BTC", "top1"],
                    help="空格不放現金，改買此幣；條件為該幣本身也符合 --trend 趨勢開關")
    ap.add_argument("--exit-ma", type=int, default=0, help="每日出場：持有中的幣收盤跌破 N 日均線就賣（0 = 只在調倉日檢查）")
    ap.add_argument("--start", default="2020-06-01")
    ap.add_argument("--combo", action="append", help='例如 "mom_28:1,taker_14:1"，可重複給多組')
    ap.add_argument("--neutral", default="lowvol_90,liq_level",
                    help="計算中性化 IC 時要扣掉的因子（逗號分隔，空字串 = 不做）")
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
    summ, by_year = evaluate_factors(F, P, U, dates, a.k, [n for n in a.neutral.split(",") if n])
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
        try:
            for n in w:
                ensure_factor(F, P, n)
        except KeyError as e:
            print(f"略過 {cs}：{e}")
            continue
        gate = trend_signal(P, a.trend) if a.trend else None
        base_gate = gate
        if gate is not None and a.rel_btc and "BTC" in P["close"]:
            rel = P["close"].div(P["close"]["BTC"], axis=0)
            gate = gate & (rel >= rel.rolling(a.rel_btc).mean())
            gate["BTC"] = base_gate["BTC"]
        exit_gate = P["close"] >= P["close"].rolling(a.exit_ma).mean() if a.exit_ma else None
        sc = composite(F, U, w)
        fb_ok = None
        if a.fallback and gate is not None:
            fb_ok = base_gate["BTC"] if a.fallback == "BTC" else pd.Series(
                [bool(base_gate.loc[t, sc.loc[t].idxmax()]) if sc.loc[t].notna().any() else False for t in dates], index=dates)
        r, tv, hold = backtest(sc, P, U, dates, a.k, reg, a.fee, gate, a.trend_mode, a.pool or None, exit_gate,
                               fallback=a.fallback or None, fallback_ok=fb_ok)
        m = metrics(r)
        m["turnover_ann"] = tv.mean() * len(dates) / (len(r) / 365)
        invested = pd.Series([sum(float(x.split(":")[1]) for x in hh.split(",") if x) for _, hh in hold])
        m["cash_avg"] = 1 - invested.clip(upper=1).mean()                      # 調倉日當下的平均空手率（不含週中每日出場）
        m["all_cash_wk"] = (invested == 0).mean()                              # 全部空手的週數比例
        rows.append(dict(combo=cs, **m))
        daily[cs] = r
        pd.DataFrame(hold, columns=["date", "holdings"]).to_csv(
            f"{a.out}/holdings_{cs.replace(':', '').replace(',', '_')}.csv", index=False)
    res = pd.DataFrame(rows).set_index("combo")
    print("\n" + "=" * 30, f"組合回測  top{a.k} {a.freq}  濾網={a.regime}  趨勢開關={a.trend or '無'}({a.trend_mode}{f' pool={a.pool}' if a.trend_mode == 'fill' else ''})  手續費={a.fee:.2%}", "=" * 30)
    fmt = res.drop(columns="years").copy()
    for c in ["CAGR", "Vol", "MDD", "MonthWin", "cash_avg", "all_cash_wk"]:
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
