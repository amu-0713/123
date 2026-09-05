"""
突破進場、跌破慢均線出場的趨勢型策略（狀態機模擬，不強求滿倉）。

規則：
- 進場：只在「月底」檢查——final_cond 過關 + 月線站上20期布林上軌
  （真正突破，不是軟門檻），這是月線尺度的結構性訊號，只需要每月看
  一次
- 出場：**每天**檢查——收盤價跌破「N個交易日均線」（用日頻資料算，
  預設約等於5個月=105個交易日）就出場，不用等到月底快照才後知後覺。
  進場判斷慢（月頻）、出場判斷快（日頻）是刻意的不對稱設計：噴出要
  確認結構性突破才進場，但轉弱要盡快反應才出場。
- 持倉上限 max_holdings 檔，等權重；空位沒有足夠新訊號遞補時維持
  空手，不強求滿倉。

逐日狀態機模擬（不是一次性矩陣排名），因為進出場時間點是各自獨立
判斷的，不是每期重新洗牌。用 numpy 陣列操作加速，避免逐日對大張
DataFrame 做 .loc 賦值太慢。
"""
import numpy as np
import pandas as pd

from strategy import compute_final_cond


def build_position_breakout_trend(data: dict, params: dict):
    price = data["price"]

    final_cond, mom_price, mkt_p, is_bear, peg = compute_final_cond(data, params)

    bb_period = params.get("bb_period", 20)
    n_std = params.get("n_std", 2)
    exit_ma_days = params.get("exit_ma_days", 105)  # 約5個月的交易日數
    max_holdings = params.get("max_holdings", 8)

    # ---- 進場訊號：月線布林突破，只在月底判斷 ----
    bar_close = mom_price.resample("ME").last()
    bar_ma = bar_close.rolling(bb_period).mean()
    bar_std = bar_close.rolling(bb_period).std()
    upper = bar_ma + n_std * bar_std
    entry_signal_monthly = (bar_close > upper).fillna(False)
    final_cond_monthly = final_cond.reindex(bar_close.index, method="ffill").fillna(False)
    entry_candidates_monthly = (entry_signal_monthly & final_cond_monthly).fillna(False)
    prev_month_close = bar_close.shift(1)
    month_ret = bar_close / prev_month_close - 1  # 候選太多時用來排序取捨

    month_end_dates = set(bar_close.index)

    # ---- 出場訊號：兩個條件用 OR 合併，任一觸發就出場（抓反應較快的那個）----
    # 1) 日頻均線（約5個月），每天判斷。ema=True 時改用指數移動平均，
    #    對近期價格反應更快，理論上能再縮短出場的反應延遲。
    use_ema = params.get("exit_ma_type", "sma") == "ema"
    if use_ema:
        exit_ma_daily = mom_price.ewm(span=exit_ma_days, min_periods=exit_ma_days).mean()
    else:
        exit_ma_daily = mom_price.rolling(exit_ma_days).mean()
    exit_signal_ma = mom_price < exit_ma_daily

    # 2) 跌破「前一個月K棒」的最低價（唐奇安通道式出場）：用月線最低價
    #    算出上個月的低點，展開成本月每天固定的參考線，每天檢查現價
    #    有沒有跌破。
    low = data["low"]
    monthly_low = low.resample("ME").min()
    prev_month_low = monthly_low.shift(1).reindex(mom_price.index, method="ffill")
    exit_signal_prev_low = mom_price < prev_month_low

    exit_signal_daily = (exit_signal_ma | exit_signal_prev_low).fillna(False)

    # 3) 月底判斷：當月K棒收盤沒有收在布林上軌之外，代表突破站不住，
    #    提前出場（不用等日頻均線或前月低點那麼慢才反應）。跟前兩個
    #    出場條件一樣用 OR 合併，但這個只在月底這天檢查（概念本身就是
    #    「這個月收在哪裡」，日頻沒有意義）。
    exit_signal_fail_monthly = (bar_close < upper).fillna(False)

    idx = price.index
    columns = list(price.columns)
    col_pos = {c: i for i, c in enumerate(columns)}

    exit_np = exit_signal_daily.reindex(columns=columns).values
    exit_fail_np_by_month = {
        dt: set(exit_signal_fail_monthly.columns[exit_signal_fail_monthly.loc[dt].values])
        for dt in exit_signal_fail_monthly.index
    }
    entry_cand_np_by_month = {
        dt: set(entry_candidates_monthly.columns[entry_candidates_monthly.loc[dt].values])
        for dt in entry_candidates_monthly.index
    }
    month_ret_reindexed = month_ret.reindex(columns=columns)

    arr = np.zeros((len(idx), len(columns)))
    warmup = max(bb_period * 21, exit_ma_days) + 5
    held = set()

    for i, dt in enumerate(idx):
        if i < warmup:
            continue

        # 1) 出場：每天都檢查，任何持股一旦跌破日頻均線/前月低點就出場
        row = exit_np[i]
        to_exit = {s for s in held if row[col_pos[s]]}
        held -= to_exit

        # 2) 月底額外出場判斷：當月收盤沒站在布林上軌外，突破站不住，
        #    提前出場；接著才判斷有沒有新的突破訊號補空位。
        if dt in month_end_dates:
            fail_set = exit_fail_np_by_month.get(dt, ())
            held -= {s for s in held if s in fail_set}

            open_slots = max_holdings - len(held)
            if open_slots > 0:
                candidates = [s for s in entry_cand_np_by_month.get(dt, ()) if s not in held]
                if len(candidates) > open_slots:
                    rets = month_ret_reindexed.loc[dt, candidates]
                    candidates = rets.sort_values(ascending=False).index[:open_slots].tolist()
                held |= set(candidates)

        if held:
            for s in held:
                arr[i, col_pos[s]] = 1.0 / max_holdings

    raw_position = pd.DataFrame(arr, index=idx, columns=columns)
    return raw_position
