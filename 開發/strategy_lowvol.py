"""
高息低波策略（你提供的第二個策略，忠實搬移，未修改邏輯）。

用途：跟主策略（動態多因子 + 停損換股）比較相關性、評估混合/動態配置
的可能性。這裡只做「原樣搬移＋可重複執行」，不改邏輯，除非你要求。

註：原始腳本有抓 rev_m（月營收）但實際上沒用到，屬於死代碼，這裡保留
抓取（反正跟主策略共用快取，不會多花額外請求)，但沒有用在因子計算裡。
"""
import numpy as np
import pandas as pd


def build_position_lowvol(data: dict, params: dict):
    price = data["price"]
    open_p = data["open_p"]
    yield_ratio = data["yield_ratio"] / 100
    vol = data["vol"]
    info = data["company_info"]

    industry = info.set_index("stock_id")["產業類別"]
    industry.index = industry.index.astype(str)
    is_fin = industry.str.contains("金融").fillna(False)

    ma_window = params.get("ma_window", 240)
    std_window = params.get("std_window", 240)
    liq_pct = params.get("liq_pct", 0.5)
    dy_range = params.get("dy_range", (0.6, 0.9))
    w_dy = params.get("w_dy", 0.33)
    w_std = params.get("w_std", 0.67)
    max_holdings = params.get("max_holdings", 12)
    max_financial = params.get("max_financial", 4)

    ma_n = price.rolling(ma_window).mean()
    liq_filter = vol.rank(axis=1, pct=True) > liq_pct
    ma_filter = price > ma_n

    std_n = price.ffill().pct_change(fill_method=None).rolling(std_window).std()

    dy_rank = yield_ratio.rank(axis=1, pct=True)
    dy_lo, dy_hi = dy_range
    dy_filter = (dy_rank > dy_lo) & (dy_rank < dy_hi)

    std_score = std_n.rank(axis=1, pct=True, ascending=False)  # 波動越低分數越高
    dy_score = dy_rank
    score = dy_score * w_dy + std_score * w_std

    final_filter = dy_filter & liq_filter & ma_filter
    score = score.where(final_filter)

    raw_position = pd.DataFrame(0, index=score.index, columns=score.columns, dtype=int)

    for dt in score.index:
        s = score.loc[dt].dropna().sort_values(ascending=False)
        if s.empty:
            continue

        selected = []
        fin_count = 0
        for stock in s.index:
            fin_flag = is_fin.get(stock, False)
            if fin_flag:
                if fin_count < max_financial:
                    selected.append(stock)
                    fin_count += 1
            else:
                selected.append(stock)
            if len(selected) >= max_holdings:
                break

        if len(selected) < max_holdings:
            remaining = [stk for stk in s.index if stk not in selected]
            for stock in remaining:
                selected.append(stock)
                if len(selected) >= max_holdings:
                    break

        raw_position.loc[dt, selected] = 1

    # raw_position 是用 pd.DataFrame() 直接建立的一般 DataFrame（不是
    # FinlabDataFrame），score 的欄位範圍（yield_ratio/vol 交集出來的）
    # 也不一定跟 price 一樣。這裡對齊回 price 的 index/columns，避免後面
    # T+1 判斷的布林運算 (buy_order & cannot_buy_t1) 因為兩邊欄位對不齊、
    # 又沒有 FinlabDataFrame 的自動對齊機制，產生非純布林 mask 而報錯。
    raw_position = raw_position.reindex(index=price.index, columns=price.columns).fillna(0).astype(int)

    # 等權重版本（0/1 flag 換算成每天總和=1 的權重），給熊市替換/停損換股
    # 目的地這種混合情境用；跟下面 position_final（0/1 flag，給獨立回測
    # 用）是兩種不同用途。
    raw_count = raw_position.sum(axis=1).replace(0, np.nan)
    raw_weight = raw_position.div(raw_count, axis=0).fillna(0)

    limit_pct = pd.Series(0.095, index=price.index)
    limit_pct.loc[: "2015-05-31"] = 0.065
    limit_up_price_next = price.mul(1 + limit_pct, axis=0)
    cannot_buy_t1 = open_p.shift(-1) >= limit_up_price_next

    resample_freq = params.get("resample_freq", "QE-JAN")
    target_pos_qe = raw_position.resample(resample_freq).last()
    prev_target_pos_qe = target_pos_qe.shift(1).fillna(0)
    prev_position = prev_target_pos_qe.reindex(raw_position.index).ffill().fillna(0)

    buy_order = raw_position > prev_position
    position_final = raw_position.copy()
    blocked_buy = (buy_order & cannot_buy_t1).fillna(False)
    position_final[blocked_buy] = prev_position[blocked_buy]

    return position_final, raw_weight
