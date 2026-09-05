"""
布林突破策略（新策略，跟主策略/高息低波策略獨立，但共用 final_cond 濾網）。

v2 修正（相對 v1 的兩個問題）：
1. 排名依據不能用帶寬變化量——帶寬衡量的是波動度，波動度大不代表股票
   強，只代表抖得兇。改用 %B = (收盤-下軌)/(上軌-下軌)，衡量的是「價格
   站在通道的哪個位置」，%B > 1 代表站上上軌，是真正的突破確認，拿來
   排名才有意義（%B 越高代表突破越強勢）。
2. 「更高週期」不是在日線上把 rolling window 拉長（例如60天），是先把
   收盤價轉成週線/月線的K棒，再用標準的 20 期布林——這才是技術分析
   真正切換時框的做法，也讓換股頻率可以直接對應到 bar_freq（週線->
   週頻換股，月線->月頻換股），不會卡死在某個特定頻率。

跟 final_cond（營收轉正、PEG合理區間、營收創高、均線多頭、流動性）
疊加使用：基本面排除地雷，技術面抓突破時機。
"""
import numpy as np
import pandas as pd

from strategy import compute_final_cond


def build_position_bbexpansion(data: dict, params: dict):
    price = data["price"]

    final_cond, mom_price, mkt_p, is_bear, peg = compute_final_cond(data, params)

    bar_freq = params.get("bar_freq", "W")  # 'W' 週線, 'ME' 月線
    bb_period = params.get("bb_period", 20)
    n_std = params.get("n_std", 2)
    n_top = params.get("n_top", 15)

    # 轉成週線/月線收盤價（真正換K棒週期，不是日線上拉長rolling窗口）
    bar_close = mom_price.resample(bar_freq).last()

    bar_ma = bar_close.rolling(bb_period).mean()
    bar_std = bar_close.rolling(bb_period).std()
    upper = bar_ma + n_std * bar_std
    lower = bar_ma - n_std * bar_std

    percent_b = (bar_close - lower) / (upper - lower)

    # 原本用 %B > 1（站上上軌）當硬性篩選條件，太嚴格：實測樣本內有
    # 5% 的月份「整個市場」都找不到同時符合基本面+站上上軌的股票，
    # 候選池月月劇烈變動（0~83檔都有），持股檔數不穩定，反而放大風險。
    # 改成軟門檻（percent_b > min_percent_b，預設 0.5 = 站上中軌即可），
    # 排名依據還是 %B（突破強度），不是硬性二元判斷。
    min_percent_b = params.get("min_percent_b", 0.5)
    breakout = percent_b > min_percent_b

    # 展開回日線索引（前向填補），跟 final_cond 對齊後再組合
    percent_b_daily = percent_b.reindex(mom_price.index, method="ffill")
    breakout_daily = breakout.reindex(mom_price.index, method="ffill").fillna(False)

    combined_cond = (final_cond & breakout_daily).fillna(False)
    score = percent_b_daily.where(combined_cond)

    ranks = score.rank(axis=1, ascending=False)
    mask = ranks <= n_top
    count = mask.sum(axis=1).replace(0, np.nan)
    raw_position = mask.div(count, axis=0).fillna(0)
    raw_position = raw_position.reindex(index=price.index, columns=price.columns).fillna(0)

    return raw_position
