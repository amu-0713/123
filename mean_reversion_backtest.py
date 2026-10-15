"""
最簡單的均值回歸回測（只做多、一次只持有一筆），讀取 download_binance_klines.py 產生的 CSV。

規則（全部只用加減乘除與移動平均）：
    1. 偏離度 z = (收盤 - N 根均價) / N 根標準差
    2. z < -ENTRY_Z（跌得夠深）→ 下一根 K 線開盤買進（避免偷看未來）
    3. 出場，先到者為準：
         a. 收盤回到均線之上（回歸完成）→ 下一根開盤賣出
         b. 跌破停損價 → 以停損價出場（若開盤直接跳空低於停損，用開盤價）
         c. 持有超過 MAX_HOLD 根 → 下一根開盤賣出
    4. 每次進出場都扣手續費與滑價

輸出的重點不是總報酬，而是：
    勝率、盈虧比、獲利因子、每筆期望值、最大不利波動（MAE）、連續虧損次數
其中 MAE 決定你「最多能開幾倍槓桿而不被強平」。

用法：
    python mean_reversion_backtest.py
"""

import os

import pandas as pd

# ======== 設定 ========
DATA_DIR = "data/spot/1h"     # 對應下載腳本的輸出資料夾
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
TIMEFRAME = "4h"              # 由 1h 合成；1h 雜訊多、手續費吃得兇，建議先從 4h 開始

N = 20                        # 均線長度
ENTRY_Z = 2.0                 # 偏離幾個標準差才進場
STOP_PCT = 0.08               # 停損：進場價下方 8%
MAX_HOLD = 30                 # 最多持有幾根 K 線

FEE = 0.0005                  # 單邊手續費（合約 taker 約 0.05%）
SLIPPAGE = 0.0005             # 單邊滑價

SPLIT_DATE = "2024-01-01"     # 之前 = 樣本內（調參數用），之後 = 樣本外（只看不調）
LEVERAGES = [1, 2, 3, 5]
# ======================


def load(symbol):
    path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["open_time"]).set_index("open_time")
    df = df.resample(TIMEFRAME).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    df["ma"] = df["close"].rolling(N).mean()
    df["sd"] = df["close"].rolling(N).std()
    df["z"] = (df["close"] - df["ma"]) / df["sd"]
    return df


def backtest(df, symbol):
    """逐根 K 線走一遍，回傳每筆交易的紀錄。"""
    o = df["open"].values
    l = df["low"].values
    c = df["close"].values
    ma = df["ma"].values
    z = df["z"].values
    t = df.index

    trades = []
    i = N
    while i < len(df) - 1:
        if not (z[i] < -ENTRY_Z):
            i += 1
            continue

        # 下一根開盤進場
        entry_i = i + 1
        entry = o[entry_i] * (1 + SLIPPAGE)
        stop = entry * (1 - STOP_PCT)
        lowest = entry
        exit_price, reason, j = None, None, entry_i

        while j < len(df):
            if o[j] <= stop:                       # 開盤就跳空跌破停損
                exit_price, reason = o[j], "停損(跳空)"
                break
            lowest = min(lowest, l[j])
            if l[j] <= stop:                       # 盤中觸及停損
                exit_price, reason = stop, "停損"
                break
            if j + 1 >= len(df):                   # 資料結束
                exit_price, reason = c[j], "資料結束"
                break
            if c[j] >= ma[j]:                      # 回到均線
                exit_price, reason = o[j + 1], "回歸"
                j += 1
                break
            if j - entry_i + 1 >= MAX_HOLD:        # 持有太久
                exit_price, reason = o[j + 1], "超時"
                j += 1
                break
            j += 1

        exit_price *= (1 - SLIPPAGE)
        ret = exit_price / entry - 1 - 2 * FEE
        trades.append({
            "symbol": symbol,
            "entry_time": t[entry_i],
            "exit_time": t[j],
            "ret": ret,
            "mae": lowest / entry - 1,             # 持倉期間最糟的浮虧
            "bars": j - entry_i,
            "reason": reason,
        })
        i = j  # 出場後才找下一筆，同一個幣不重疊
    return trades


def max_consecutive_losses(rets):
    worst = run = 0
    for r in rets:
        run = run + 1 if r <= 0 else 0
        worst = max(worst, run)
    return worst


def report(tr, title):
    print(f"\n===== {title} =====")
    if len(tr) == 0:
        print("沒有交易")
        return
    wins = tr[tr["ret"] > 0]["ret"]
    losses = tr[tr["ret"] <= 0]["ret"]
    win_rate = len(wins) / len(tr)
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    payoff = avg_win / -avg_loss if avg_loss < 0 else float("inf")
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else float("inf")
    expectancy = tr["ret"].mean()

    print(f"交易次數       {len(tr)}")
    print(f"勝率           {win_rate:.1%}")
    print(f"平均獲利       {avg_win:+.2%}")
    print(f"平均虧損       {avg_loss:+.2%}")
    print(f"盈虧比         {payoff:.2f}")
    print(f"獲利因子       {pf:.2f}   （>1.3 才算有點東西）")
    print(f"每筆期望值     {expectancy:+.3%}")
    print(f"最大單筆虧損   {tr['ret'].min():+.2%}")
    print(f"最糟 MAE       {tr['mae'].min():+.2%}   ← 槓桿 × 這個數字接近 -100% 就會被強平")
    print(f"MAE 第 5 百分位 {tr['mae'].quantile(0.05):+.2%}")
    print(f"最多連續虧損   {max_consecutive_losses(tr['ret'].tolist())} 筆")
    print(f"平均持有       {tr['bars'].mean():.1f} 根")
    print("出場原因       " + ", ".join(f"{k} {v}" for k, v in tr["reason"].value_counts().items()))

    # 槓桿試算：每筆用全部資金 × 槓桿，依出場時間順序複利
    # （多個幣同時持倉時這裡會高估，只當作量級參考）
    seq = tr.sort_values("exit_time")
    print("槓桿   總報酬        最大回撤   是否曾被強平")
    for lev in LEVERAGES:
        equity, peak, mdd, liquidated = 1.0, 1.0, 0.0, False
        for r, mae in zip(seq["ret"], seq["mae"]):
            if mae * lev <= -0.9:            # 留 10% 給維持保證金，粗略判定
                liquidated = True
                equity = 0.0
                break
            equity *= 1 + r * lev
            peak = max(peak, equity)
            mdd = min(mdd, equity / peak - 1)
        print(f"{lev:>3}x   {equity - 1:>+10.1%}   {mdd:>8.1%}   {'是' if liquidated else '否'}")


def main():
    all_trades = []
    for sym in SYMBOLS:
        df = load(sym)
        if df is None:
            print(f"找不到 {sym} 的資料，略過")
            continue
        all_trades += backtest(df, sym)

    tr = pd.DataFrame(all_trades)
    if tr.empty:
        print("沒有任何交易，請確認資料路徑")
        return

    print(f"參數：{TIMEFRAME}  N={N}  ENTRY_Z={ENTRY_Z}  STOP={STOP_PCT:.0%}  MAX_HOLD={MAX_HOLD}")
    split = pd.Timestamp(SPLIT_DATE, tz="UTC")
    report(tr[tr["entry_time"] < split], f"樣本內（{SPLIT_DATE} 之前）")
    report(tr[tr["entry_time"] >= split], f"樣本外（{SPLIT_DATE} 之後）")

    print("\n===== 各幣（全期間）=====")
    g = tr.groupby("symbol")["ret"]
    print(pd.DataFrame({
        "筆數": g.size(),
        "勝率": g.apply(lambda r: (r > 0).mean()).map("{:.0%}".format),
        "期望值": g.mean().map("{:+.2%}".format),
    }))

    tr.to_csv("trades.csv", index=False)
    print("\n每筆交易明細已存到 trades.csv")


if __name__ == "__main__":
    main()
