"""
策略核心邏輯：把 price / open / pe / monthly_revenue / 成交金額 / 還原股價
轉成每日目標持股權重矩陣 position_final。

從原始單檔腳本拆出來，方便：
1. 樣本內／樣本外用同一份邏輯跑，不用複製貼上兩份程式碼
2. 之後做參數敏感度測試時，只改 config.PARAMS，不用動這裡的邏輯

跟原始腳本邏輯相同，額外加了兩個可選開關（預設關閉，行為等同原本）：
- params["use_adj_price"]：動能/回撤/相關性/均線濾網改用還原股價算
  （PE、實際下單價格仍用原始股價，這兩個本來就該對應真實市場價格）
- params["regime_confirm_days"]：大盤牛熊訊號要連續 N 天才切換，
  避免在 MA30/MA60 糾結時來回打臉、頻繁換股
"""
import numpy as np
import pandas as pd


def _apply_hysteresis(is_bear_raw: pd.Series, confirm_days: int) -> pd.Series:
    """牛熊訊號要連續 confirm_days 天不變才真的切換狀態。"""
    if confirm_days <= 1:
        return is_bear_raw

    out = []
    state = bool(is_bear_raw.iloc[0])
    streak = 0
    candidate = state
    for val in is_bear_raw:
        val = bool(val)
        if val == candidate:
            streak += 1
        else:
            candidate = val
            streak = 1
        if streak >= confirm_days:
            state = candidate
        out.append(state)
    return pd.Series(out, index=is_bear_raw.index)


def compute_final_cond(data: dict, params: dict):
    """基本面+均線+流動性濾網，回傳 (final_cond, mom_price, mkt_p, is_bear)。
    抽成獨立函式，讓 build_position 跟其他策略（例如布林擴張）共用同一套
    篩選邏輯，不用複製貼上。"""
    price = data["price"]
    pe = data["pe"]
    rev_m = data["rev_m"]
    vol = data["vol"]

    # 動能/回撤/相關性/均線濾網用哪個價格序列算：預設用原始收盤價（跟
    # 原始腳本一樣），開 use_adj_price 就改用還原股價，避免除權息造成
    # 的「假下跌」系統性拉低高股息股的動能排名、拉高回撤估計。
    mom_price = data["adj_close"] if params.get("use_adj_price") else price

    # 大盤(0050)一律用還原股價，不受 use_adj_price 開關影響。
    # 原因：0050 在 2025/6 做過 1:4 分割，原始收盤價在分割當天會出現
    # 「單日跌 75%」的假資料（188.65 -> 47.57），還原股價才是正確連續的
    # 序列。這不是「除權息風格選擇」，是實打實的資料錯誤，一定要修，
    # 否則牛熊判斷的均線、以及所有個股的 corr_mkt 因子在分割當天附近
    # 都會被這個假崩盤污染。
    mkt_p = data["adj_close"]["0050"]

    ma_s, ma_m, ma_l = params["ma_windows"]
    mkt_ma_s, mkt_ma_l = params["mkt_ma_windows"]

    # ---- 均線 ----
    ma_short = mom_price.rolling(ma_s).mean()
    ma_mid = mom_price.rolling(ma_m).mean()
    ma_long = mom_price.rolling(ma_l).mean()

    mkt_ma_short = mkt_p.rolling(mkt_ma_s).mean()
    mkt_ma_long = mkt_p.rolling(mkt_ma_l).mean()

    # ---- 大盤牛熊 & 個股多頭排列濾網 ----
    is_bear_raw = mkt_ma_short < mkt_ma_long
    confirm_days = params.get("regime_confirm_days", 1)
    is_bear = _apply_hysteresis(is_bear_raw, confirm_days) if confirm_days > 1 else is_bear_raw

    c_ma_filter = (ma_short > ma_mid) & (ma_mid > ma_long)

    # ---- 選股過濾條件 ----
    rev_ma = rev_m.rolling(params["rev_ma_window"]).mean()
    rev_g = (rev_m / rev_m.shift(params["rev_growth_lag"])) - 1

    growth_pct = (rev_g * 100).replace(0, np.nan)
    peg = pe / growth_pct

    peg_lo, peg_hi = params["peg_range"]
    c_rev_positive = rev_ma > 0
    c_peg_range = (peg > peg_lo) & (peg < peg_hi)
    c_rev_high = rev_ma == rev_ma.rolling(params["rev_high_window"]).max()
    c_hist = rev_m.notnull().rolling(params["rev_high_window"] + 1).min() == 1
    c_valid = peg.notnull() & rev_g.notnull()
    c_liq = vol.rolling(params["liq_window"]).min() > params["liq_min_value"]

    final_cond = (
        c_rev_positive
        & c_peg_range
        & c_rev_high
        & c_hist
        & c_valid
        & c_ma_filter
        & c_liq
    ).fillna(False)

    return final_cond, mom_price, mkt_p, is_bear, peg


def build_position(data: dict, params: dict, bear_weight_override: pd.DataFrame = None):
    price = data["price"]
    open_p = data["open_p"]

    final_cond, mom_price, mkt_p, is_bear, peg = compute_final_cond(data, params)

    # ---- 多因子評分 ----
    rs = mom_price.ffill().pct_change(params["rs_window"], fill_method=None)
    rets = mom_price.pct_change(fill_method=None)
    mkt_rets = mkt_p.pct_change(fill_method=None)

    dd = rets.where(rets < 0, 0).rolling(params["dd_window"]).std().replace(0, np.nan)
    corr_mkt = rets.rolling(params["corr_window"]).corr(mkt_rets)

    # 因子排名表：{因子名: 排名分數 (0~1，分數越高越好)}。rs/peg/corr/dd 是
    # 原本就有的因子；margin/beta/size/hist_maxdd 是額外的風險因子，預設
    # 權重都是0（不影響原本邏輯），要透過 params["weights"] 指定非零權重
    # 才會生效。
    factor_ranks = {
        "rs": rs.where(final_cond).rank(axis=1, pct=True),
        "peg": (1 / peg).where(final_cond).rank(axis=1, pct=True),
        "dd": (-dd).where(final_cond).rank(axis=1, pct=True),
        "corr": (-corr_mkt).where(final_cond).rank(axis=1, pct=True),
    }

    # 融資使用率：越低分數越高（融資=散戶槓桿比重高，系統性下跌容易被
    # 追繳斷頭，賣壓被放大）。
    margin_usage = data.get("margin_usage")
    if margin_usage is not None:
        factor_ranks["margin"] = (-margin_usage).where(final_cond).rank(axis=1, pct=True)

    # 正式迴歸 beta（不是相關係數，有考慮波動幅度）：越低分數越高。
    # corr 只看「跟大盤同不同方向」，beta 額外考慮「跟大盤同向時波動被
    # 放大多少」，兩者衡量的風險不完全一樣。
    beta_window = params.get("beta_window", params["corr_window"])
    cov_mkt = rets.rolling(beta_window).cov(mkt_rets)
    var_mkt = mkt_rets.rolling(beta_window).var()
    beta = cov_mkt.div(var_mkt, axis=0)
    factor_ranks["beta"] = (-beta).where(final_cond).rank(axis=1, pct=True)

    # 市值：越大分數越高（大型股籌碼穩定、法人持股比重高，系統性下跌
    # 時通常比小型股抗跌）。
    market_value = data.get("market_value")
    if market_value is not None:
        factor_ranks["size"] = market_value.where(final_cond).rank(axis=1, pct=True)

    # 個股自身歷史最大回撤（過去 hist_dd_window 天內，這檔股票曾經從高點
    # 跌過多深）：跌得越淺分數越高。比日內波動度（dd）更直接衡量「這檔
    # 股票真的崩起來會有多深」，抓的是尾部風險而不是日常波動。
    hist_dd_window = params.get("hist_dd_window", 252)
    roll_peak = mom_price.rolling(hist_dd_window, min_periods=20).max()
    daily_dd_from_peak = mom_price / roll_peak - 1
    hist_maxdd = daily_dd_from_peak.rolling(hist_dd_window, min_periods=20).min()
    factor_ranks["hist_maxdd"] = hist_maxdd.where(final_cond).rank(axis=1, pct=True)

    # 創新高強度（技術面，取代 rs 用）：price / N日內最高價，越接近1分數
    # 越高（=正在創新高）。跟 rs（N日報酬率）衡量的不是同一件事：rs 抓
    # 「這段期間漲了多少%」，可能只是跌深反彈；這個因子抓「有沒有站上
    # 前高、走最小阻力方向」，是動能文獻裡獨立驗證過的52週新高效應。
    high_prox_window = params.get("high_prox_window", 240)
    roll_high = mom_price.rolling(high_prox_window, min_periods=20).max()
    high_prox = mom_price / roll_high
    factor_ranks["high_prox"] = high_prox.where(final_cond).rank(axis=1, pct=True)

    is_bear_mask = is_bear.reindex(r_rs_index := factor_ranks["rs"].index).ffill().fillna(True)
    regime = pd.Series(np.where(is_bear_mask, "bear", "bull"), index=r_rs_index)

    weights = pd.DataFrame(params["weights"])

    score = pd.DataFrame(0.0, index=r_rs_index, columns=price.columns)
    for factor_name, r_factor in factor_ranks.items():
        if factor_name not in weights.columns:
            continue
        w_factor = regime.map(weights[factor_name])
        score = score.add(r_factor.mul(w_factor, axis=0).fillna(0), fill_value=0)

    # ---- 目標持股（依牛熊決定檔數，等權重）----
    n_bull = params["n_bull"]
    n_bear = params["n_bear"]
    score_ranks = score.rank(axis=1, ascending=False)

    bull_mask = score_ranks <= n_bull
    bull_count = bull_mask.sum(axis=1).replace(0, np.nan)
    weight_bull = bull_mask.div(bull_count, axis=0).fillna(0)

    # bear_liq_min_value：熊市選股額外加一道（通常比 liq_min_value 更嚴格
    # 的）流動性門檻，只影響熊市候選池，不影響牛市。動機：熊市集中持股
    # 時，排名比較後面的候選股本身防禦因子(dd/corr)就比較差，擴大檔數
    # 反而更糟；與其擴大檔數，改成在既有排名前提下先篩掉流動性太差的
    # candidate，看能不能篩掉真正容易被恐慌盤打殘的小型股。
    bear_liq_min_value = params.get("bear_liq_min_value")
    if bear_liq_min_value:
        c_liq_bear = vol.rolling(params["liq_window"]).min() > bear_liq_min_value
        score_bear = score.where(c_liq_bear)
    else:
        score_bear = score
    score_ranks_bear = score_bear.rank(axis=1, ascending=False)

    bear_mask = score_ranks_bear <= n_bear
    bear_count = bear_mask.sum(axis=1).replace(0, np.nan)
    weight_bear = bear_mask.div(bear_count, axis=0).fillna(0)

    # bear_weight_override：熊市時要用哪個部位取代 weight_bear（例如換成
    # 高息低波策略的當下選股），不傳就是原本的邏輯（5檔集中成長股）。
    if bear_weight_override is not None:
        weight_bear = bear_weight_override.reindex(
            index=weight_bull.index, columns=weight_bull.columns
        ).fillna(0)

    raw_position = weight_bull.where(~is_bear_mask, weight_bear).fillna(0)

    # pe / rev_m 的股票涵蓋範圍跟 price 不完全一致（例如個股沒有本益比、
    # 或跟 price 的交易日曆對不上），score/raw_position 經過一路 rank/where
    # 之後，欄位可能跟 price 不再完全相同。這裡先對齊回 price 的
    # index/columns，讓後面 T+1 漲停判斷的布林運算 (buy_order & cannot_buy_t1)
    # 兩邊形狀一致，不會因為欄位對不齊產生 NaN，讓 mask 變成非純布林型別
    # （原始腳本把這個對齊動作放在最後才做，順序太晚，曾在此實際觸發
    # `position_final[blocked_buy] = ...` 報錯）。
    raw_position = raw_position.reindex(index=price.index, columns=price.columns).fillna(0)

    # ---- T+1 漲停買不到防呆（一律用原始股價，因為漲停帶是對應真實
    # 成交價，不能用還原股價算） ----
    limit_pct = pd.Series(params["limit_pct_new"], index=price.index)
    limit_pct.loc[: params["limit_pct_switch_date"]] = params["limit_pct_old"]

    limit_up_price_next = price.mul(1 + limit_pct, axis=0)
    cannot_buy_t1 = open_p.shift(-1) >= limit_up_price_next

    target_pos_qe = raw_position.resample("QE").last()
    prev_target_pos_qe = target_pos_qe.shift(1).fillna(0)
    prev_position = prev_target_pos_qe.reindex(raw_position.index).ffill().fillna(0)

    buy_order = raw_position > prev_position
    position_final = raw_position.copy()
    blocked_buy = buy_order & cannot_buy_t1
    position_final[blocked_buy] = prev_position[blocked_buy]

    position_final = position_final.reindex(index=price.index, columns=price.columns).fillna(0)

    missing = set(position_final.columns) - set(price.columns)
    if missing:
        raise ValueError(f"存在未對齊股票欄位: {missing}")

    # raw_position 一併回傳：全倉停損換股（risk_overlay.py）需要拿「當下
    # 模型分數最新的目標持股」當作觸發停損那天的換股標的，不能只靠
    # position_final（它的 T+1 防呆是綁季度比較的，季度中的每一天不一定
    # 有意義）。
    return position_final, raw_position
