"""
個股停損（跟 risk_overlay.py 的「全倉」層級不同，這裡是每一檔個別追蹤）。

兩種版本：
- apply_individual_stop_loss：固定停損，從「進場價」起算虧損幅度。
- apply_individual_trailing_stop：移動停損，從「進場後的最高價」起算
  回落幅度——會保護已經賺到的獲利，不是只守進場價。

兩者規則都一樣：觸發當天把那檔的權重歸零，直到「下一次季度再平衡」
才恢復；空出來的部位維持空手，不會拿其他股票遞補。

只是用來『找觸發日』的近似估計（用原始收盤價逐日追蹤，不計盤中停損的
滑價），實際績效仍然交給 sim() 用完整費用重新算一次。
"""
import pandas as pd


def apply_individual_stop_loss(
    position_final: pd.DataFrame,
    price: pd.DataFrame,
    open_p: pd.DataFrame,
    quarterly_dates: list,
    stop_loss_pct: float,
):
    """回傳 (position, resample_dates)。"""
    combined = position_final.copy()
    idx = price.index
    extra_dates = set()

    sorted_q = sorted(quarterly_dates)
    for qi, q_start in enumerate(sorted_q):
        if q_start not in idx:
            continue
        q_end = sorted_q[qi + 1] if qi + 1 < len(sorted_q) else idx[-1]

        start_i = idx.get_loc(q_start)
        entry_i = min(start_i + 1, len(idx) - 1)
        entry_date = idx[entry_i]

        holdings = position_final.loc[q_start]
        held_stocks = holdings[holdings > 0].index
        if len(held_stocks) == 0:
            continue

        entry_price = open_p.loc[entry_date, held_stocks]

        period_mask = (idx > q_start) & (idx <= q_end)
        period_idx = idx[period_mask]
        if len(period_idx) == 0:
            continue

        for stock in held_stocks:
            ep = entry_price.get(stock)
            if pd.isna(ep) or ep == 0:
                continue
            path = price.loc[period_idx, stock]
            ret = path / ep - 1
            hit = ret[ret <= -stop_loss_pct]
            if len(hit) > 0:
                stop_date = hit.index[0]
                combined.loc[stop_date:q_end, stock] = 0
                extra_dates.add(stop_date)

    resample_dates = sorted(set(quarterly_dates) | extra_dates)
    return combined, resample_dates


def apply_individual_trailing_stop(
    position_final: pd.DataFrame,
    price: pd.DataFrame,
    open_p: pd.DataFrame,
    quarterly_dates: list,
    trail_pct: float,
):
    """回傳 (position, resample_dates)。跟 apply_individual_stop_loss 差別
    只在於觸發門檻：這裡是「進場後至今的最高價」回落 trail_pct，不是
    「進場價」虧損 trail_pct——會保護中途已經賺到的獲利。"""
    combined = position_final.copy()
    idx = price.index
    extra_dates = set()

    sorted_q = sorted(quarterly_dates)
    for qi, q_start in enumerate(sorted_q):
        if q_start not in idx:
            continue
        q_end = sorted_q[qi + 1] if qi + 1 < len(sorted_q) else idx[-1]

        start_i = idx.get_loc(q_start)
        entry_i = min(start_i + 1, len(idx) - 1)
        entry_date = idx[entry_i]

        holdings = position_final.loc[q_start]
        held_stocks = holdings[holdings > 0].index
        if len(held_stocks) == 0:
            continue

        entry_price = open_p.loc[entry_date, held_stocks]

        period_mask = (idx > q_start) & (idx <= q_end)
        period_idx = idx[period_mask]
        if len(period_idx) == 0:
            continue

        for stock in held_stocks:
            ep = entry_price.get(stock)
            if pd.isna(ep) or ep == 0:
                continue
            path = price.loc[period_idx, stock]
            running_peak = path.cummax().clip(lower=ep)  # 高點不會低於進場價
            trail_level = running_peak * (1 - trail_pct)
            hit = path[path <= trail_level]
            if len(hit) > 0:
                stop_date = hit.index[0]
                combined.loc[stop_date:q_end, stock] = 0
                extra_dates.add(stop_date)

    resample_dates = sorted(set(quarterly_dates) | extra_dates)
    return combined, resample_dates
