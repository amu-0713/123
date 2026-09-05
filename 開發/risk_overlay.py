"""
全倉停損 overlay（可選，不開就等同原本純季度再平衡）。

不做個股停損，只做「整體」層級的風控：
- 觸發來源 (stop_trigger_source)：
    "portfolio"：自己組合的權益回撤（用季度持倉展開估的近似值，不計手續費）
    "market"：大盤（0050）本身的回撤，固定百分比門檻
    "market_atr"：大盤（0050）本身的回撤，門檻改成 atr_k 倍的 ATR（换算成
                  價格百分比），會隨當下波動度自動放大/縮小，而不是固定
                  15% 這種寫死的數字
- 觸發後的動作 (stop_action)：
    "switch"：立刻換成當下模型分數最新的目標持股，維持滿倉
    "cash"：出場觀望 stop_cash_days 個交易日，之後才用當時最新的
            目標持股重新進場
    "defensive"：換成 defensive_weight（例如另一個低波動策略的當下
            選股），回撤縮小到 portfolio_stop_recover 以內才換回原本的
            raw_position——平常維持原策略全部曝險，只有觸發時才防禦性
            換股，跟静態按比例混合兩本帳不同。

stop_min_gap_days：觸發/換回事件跟下一次「原訂季度再平衡」如果間隔小於
這個交易日數，就跳過那次季度再平衡（沿用剛換好的持股），避免觸發後沒
幾天又遇到季度末，平白多一次手續費。

這些估計都只是用來『找觸發日』的近似值，不會呼叫 finlab API；
實際績效仍然交給 sim() 用完整費用重新算一次。
"""
import pandas as pd


def _quarterly_dates(index: pd.DatetimeIndex) -> list:
    dates = pd.date_range(index[0], index[-1], freq="QE")
    return [d for d in dates if index[0] <= d <= index[-1]]


def _drawdown(cum_ret: pd.Series) -> pd.Series:
    running_max = cum_ret.cummax()
    return cum_ret / running_max - 1


def _find_triggers(dd: pd.Series, stop_dd: float, recover_dd: float) -> list:
    """固定百分比門檻版本，只記錄「進入」事件（給 switch / cash 用）。"""
    triggers = []
    armed = True
    for date, d in dd.items():
        if armed and d <= -stop_dd:
            triggers.append(date)
            armed = False
        elif not armed and d >= -recover_dd:
            armed = True
    return triggers


def _find_triggers_dynamic(dd: pd.Series, threshold: pd.Series, recover_frac: float) -> list:
    """門檻本身隨時間變動（例如 ATR 換算出來的百分比）的版本。"""
    triggers = []
    armed = True
    for date in dd.index:
        d = dd.loc[date]
        th = threshold.get(date)
        if th is None or pd.isna(th) or pd.isna(d):
            continue
        if armed and d <= -th:
            triggers.append(date)
            armed = False
        elif not armed and d >= -th * recover_frac:
            armed = True
    return triggers


def _find_enter_exit_events(dd: pd.Series, stop_dd: float, recover_dd: float) -> list:
    """回傳 [(date, 'enter'), (date, 'exit'), ...]，給 stop_action='defensive' 用：
    進入防禦模式、以及回穩後退出防禦模式各自的事件日。"""
    events = []
    state = "normal"
    for date, d in dd.items():
        if state == "normal" and d <= -stop_dd:
            events.append((date, "enter"))
            state = "defensive"
        elif state == "defensive" and d >= -recover_dd:
            events.append((date, "exit"))
            state = "normal"
    return events


def _portfolio_drawdown(position_final: pd.DataFrame, price: pd.DataFrame, quarterly_dates: list) -> pd.Series:
    q_snapshot = position_final.reindex(quarterly_dates, method="ffill")
    daily_weight = q_snapshot.reindex(price.index, method="ffill").fillna(0)
    daily_ret = price.pct_change(fill_method=None).fillna(0)
    port_ret = (daily_weight.shift(1).fillna(0) * daily_ret).sum(axis=1)
    equity = (1 + port_ret).cumprod()
    return _drawdown(equity)


def _market_drawdown(mkt_price: pd.Series) -> pd.Series:
    """mkt_price 一定要傳還原股價（例如 adj_close['0050']），不能用原始
    收盤價——0050 在 2025/6 做過 1:4 分割，原始收盤價分割當天會出現
    假的單日跌 75%，會讓回撤/解除冷卻邏輯整個壞掉（解除冷卻需要價格
    回到接近前波高點，分割後原始價格永遠回不去，等於永久卡在觸發後、
    再也無法重新觸發)。"""
    mkt_ret = mkt_price.pct_change(fill_method=None).fillna(0)
    equity = (1 + mkt_ret).cumprod()
    return _drawdown(equity)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _market_atr_dd_and_threshold(
    price: pd.DataFrame,
    adj_close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    atr_window: int,
    atr_k: float,
    mkt_col: str = "0050",
):
    """回傳 (dd, threshold)：threshold 是每天「atr_k 倍 ATR」換算成的百分比
    （ATR / 收盤價 * atr_k），會隨大盤波動度自動變化，取代固定 15%。

    high/low 目前只有原始（未還原）版本，用還原收盤價 / 原始收盤價 的
    比值把 high/low 也換算成還原後的尺度，避免分割當天原始 high/low
    跟還原後 close 尺度對不上，產生一根假的巨大 True Range。"""
    close = price[mkt_col]
    adj_c = adj_close[mkt_col]
    ratio = adj_c / close
    adj_high = high[mkt_col] * ratio
    adj_low = low[mkt_col] * ratio

    tr = _true_range(adj_high, adj_low, adj_c)
    atr = tr.rolling(atr_window).mean()
    threshold = (atr_k * atr / adj_c).reindex(adj_c.index)
    dd = _market_drawdown(adj_c)
    return dd, threshold


def _suppress_close_quarterly_dates(quarterly_dates, extra_dates, price_index, min_gap_days):
    """extra_dates 之後 min_gap_days 個交易日以內的季度再平衡日直接跳過
    （沿用 extra_dates 剛換好的持股），避免觸發後沒幾天又遇到季度末，
    平白多一次手續費。"""
    if min_gap_days <= 0 or not extra_dates:
        return quarterly_dates

    loc = {d: i for i, d in enumerate(price_index)}
    extra_i = sorted(loc[d] for d in extra_dates if d in loc)

    keep = []
    for qd in quarterly_dates:
        qi = loc.get(qd)
        too_close = qi is not None and any(0 <= qi - ei < min_gap_days for ei in extra_i)
        if not too_close:
            keep.append(qd)
    return keep


def build_overlay_position(
    position_final: pd.DataFrame,
    raw_position: pd.DataFrame,
    price: pd.DataFrame,
    params: dict,
    adj_close: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    defensive_weight: pd.DataFrame = None,
):
    """回傳 (position, resample_dates)，可以直接丟給 sim(resample=resample_dates)。

    params 沒設 portfolio_stop_dd（或設 None/0）時，完全等同原本純季度
    再平衡（resample_dates 只有季度末）。stop_trigger_source="market" /
    "market_atr" 都要傳 adj_close（還原股價，見 _market_drawdown 說明）。
    stop_trigger_source="market_atr" 另外還要傳 high/low。
    stop_action="defensive" 要傳 defensive_weight（例如另一個防禦性策略
    的當下選股，權重需已正規化成每天總和=1）。
    """
    quarterly_dates = _quarterly_dates(price.index)
    stop_dd = params.get("portfolio_stop_dd")

    if not stop_dd:
        return position_final, quarterly_dates

    recover_dd = params.get("portfolio_stop_recover", 0.05)
    trigger_source = params.get("stop_trigger_source", "portfolio")
    action = params.get("stop_action", "switch")
    cash_days = params.get("stop_cash_days", 20)
    min_gap_days = params.get("stop_min_gap_days", 20)

    if trigger_source == "market_atr":
        if adj_close is None or high is None or low is None:
            raise ValueError("stop_trigger_source='market_atr' 需要傳 adj_close/high/low")
        atr_window = params.get("atr_window", 20)
        atr_k = params.get("atr_k", 3.0)
        recover_frac = params.get("atr_recover_frac", recover_dd / stop_dd)
        dd, threshold = _market_atr_dd_and_threshold(price, adj_close, high, low, atr_window, atr_k)
        get_triggers = lambda: _find_triggers_dynamic(dd, threshold, recover_frac)  # noqa: E731
    elif trigger_source == "market":
        if adj_close is None:
            raise ValueError("stop_trigger_source='market' 需要傳 adj_close")
        dd = _market_drawdown(adj_close["0050"])
        get_triggers = lambda: _find_triggers(dd, stop_dd, recover_dd)  # noqa: E731
    else:
        dd = _portfolio_drawdown(position_final, price, quarterly_dates)
        get_triggers = lambda: _find_triggers(dd, stop_dd, recover_dd)  # noqa: E731

    combined = position_final.copy()
    extra_dates = []

    if action == "defensive":
        if defensive_weight is None:
            raise ValueError("stop_action='defensive' 需要傳 defensive_weight")
        events = _find_enter_exit_events(dd, stop_dd, recover_dd)
        for d, kind in events:
            if d not in combined.index:
                continue
            combined.loc[d] = defensive_weight.loc[d] if kind == "enter" else raw_position.loc[d]
            extra_dates.append(d)
    else:
        triggers = [d for d in get_triggers() if d in position_final.index]
        idx = price.index
        for d in triggers:
            if action == "switch":
                combined.loc[d] = raw_position.loc[d]
                extra_dates.append(d)
            elif action == "cash":
                pos_i = idx.get_loc(d)
                end_i = min(pos_i + cash_days, len(idx) - 1)
                combined.loc[idx[pos_i:end_i + 1]] = 0
                extra_dates.append(d)
                reentry_date = idx[end_i]
                if end_i > pos_i:
                    combined.loc[reentry_date] = raw_position.loc[reentry_date]
                    extra_dates.append(reentry_date)
            else:
                raise ValueError(f"未知的 stop_action: {action}")

    kept_quarterly = _suppress_close_quarterly_dates(quarterly_dates, extra_dates, price.index, min_gap_days)
    resample_dates = sorted(set(kept_quarterly) | set(extra_dates))
    return combined, resample_dates
