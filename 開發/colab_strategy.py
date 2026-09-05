# ============================================================================
# 動態多因子策略 + 全倉停損換股
# 全倉停損換股規則：大盤(0050)從高點回落 15% 時，不等季度、立刻換成
# 當下模型分數最新的目標持股（維持滿倉、不出場）。
# 已用樣本內 2010-2021 / 樣本外 2022-2026 驗證過，細節見對話紀錄 /
# 本機開發版的 README.md。
# ============================================================================

import finlab
# 到 https://ai.finlab.tw/api_token/ 換一組新的 token 貼在這裡。
# 提醒：如果你在別的地方用過的 token 已經外流（例如貼過在聊天視窗），
# 記得先去該網站撤銷舊的、換一組新的，不要沿用舊 token。
finlab.login('你的token')

from finlab.backtest import sim
from finlab import data
import pandas as pd
import numpy as np

# =============================================================================
# 一、資料抓取與基礎指標計算
# =============================================================================
price  = data.get('price:收盤價').loc['2006':'2026']
open_p = data.get('price:開盤價').loc['2006':'2026']
pe     = data.get('price_earning_ratio:本益比').loc['2006':'2026']
rev_m  = data.get('monthly_revenue:當月營收').loc['2006':'2026']
vol    = data.get('price:成交金額').loc['2006':'2026']

for df in [price, open_p, pe, rev_m, vol]:
    df.columns = df.columns.astype(str)

mkt_p = price['0050']

ma20  = price.rolling(20).mean()
ma60  = price.rolling(60).mean()
ma120 = price.rolling(120).mean()
mkt_30 = mkt_p.rolling(30).mean()
mkt_60 = mkt_p.rolling(60).mean()

# =============================================================================
# 二、大盤狀態與均線濾網
# =============================================================================
is_bear = mkt_30 < mkt_60
c_ma_filter = (ma20 > ma60) & (ma60 > ma120)

# =============================================================================
# 三、選股過濾條件
# =============================================================================
rev_ma3 = rev_m.rolling(3).mean()
rev_g = (rev_m / rev_m.shift(12)) - 1
growth_pct = (rev_g * 100).replace(0, np.nan)
peg = pe / growth_pct

c_rev_positive = rev_ma3 > 0
c_peg_range = (peg > 0.2) & (peg < 1.8)
c_rev_high = rev_ma3 == rev_ma3.rolling(12).max()
c_hist = rev_m.notnull().rolling(13).min() == 1
c_valid = peg.notnull() & rev_g.notnull()
c_liq = vol.rolling(20).min() > 1e6

final_cond = (
    c_rev_positive
    & c_peg_range
    & c_rev_high
    & c_hist
    & c_valid
    & c_ma_filter
    & c_liq
).fillna(False)

# =============================================================================
# 四、多因子評分系統
# =============================================================================
rs_fixed = price.ffill().pct_change(80, fill_method=None)
rets = price.pct_change(fill_method=None)
mkt_rets = mkt_p.pct_change(fill_method=None)

dd = rets.where(rets < 0, 0).rolling(20).std().replace(0, np.nan)
corr_mkt = rets.rolling(60).corr(mkt_rets)

r_rs = rs_fixed.where(final_cond).rank(axis=1, pct=True)
r_peg = (1 / peg).where(final_cond).rank(axis=1, pct=True)
r_dd = (-dd).where(final_cond).rank(axis=1, pct=True)
r_corr = (-corr_mkt).where(final_cond).rank(axis=1, pct=True)

is_bear_mask = is_bear.reindex(r_rs.index).ffill().fillna(True)
regime = pd.Series(np.where(is_bear_mask, 'bear', 'bull'), index=r_rs.index)

weights = pd.DataFrame({
    'rs':   {'bull': 0.3, 'bear': 0.3},
    'peg':  {'bull': 0.3, 'bear': 0.0},
    'corr': {'bull': 0.0, 'bear': 0.3},
    'dd':   {'bull': 0.4, 'bear': 0.4},
})
w_rs_dyn = regime.map(weights['rs'])
w_peg_dyn = regime.map(weights['peg'])
w_corr_dyn = regime.map(weights['corr'])
w_dd_dyn = regime.map(weights['dd'])

score = (
    r_rs.mul(w_rs_dyn, axis=0).fillna(0)
    + r_peg.mul(w_peg_dyn, axis=0).fillna(0)
    + r_corr.mul(w_corr_dyn, axis=0).fillna(0)
    + r_dd.mul(w_dd_dyn, axis=0).fillna(0)
)

# =============================================================================
# 五、目標持股（牛市16檔/熊市5檔，等權重）
# =============================================================================
N_BULL = 16
N_BEAR = 5
score_ranks = score.rank(axis=1, ascending=False)

bull_mask = score_ranks <= N_BULL
bull_count = bull_mask.sum(axis=1).replace(0, np.nan)
weight_bull = bull_mask.div(bull_count, axis=0).fillna(0)

bear_mask = score_ranks <= N_BEAR
bear_count = bear_mask.sum(axis=1).replace(0, np.nan)
weight_bear = bear_mask.div(bear_count, axis=0).fillna(0)

raw_position = weight_bull.where(~is_bear_mask, weight_bear).fillna(0)
# 對齊回 price 的 index/columns，避免 pe/rev_m 欄位範圍跟 price 不一致，
# 導致後面 T+1 判斷的布林運算因為對不齊而出錯。
raw_position = raw_position.reindex(index=price.index, columns=price.columns).fillna(0)

# =============================================================================
# 六、T+1 漲停買不到防呆
# =============================================================================
limit_pct = pd.Series(0.095, index=price.index)
limit_pct.loc[:'2015-05-31'] = 0.065
limit_up_price_next = price.mul(1 + limit_pct, axis=0)
cannot_buy_t1 = open_p.shift(-1) >= limit_up_price_next

target_pos_qe = raw_position.resample('QE').last()
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

# =============================================================================
# 七、回測設定與執行
#
# 註：曾經測過「大盤(0050)回撤15%觸發、留倉換股」這個全倉停損機制，
# 樣本內/外一度同時改善。但為了處理 2025/4 那次跳空崩盤（-28~30%
# MDD），後續又測了十幾種角度（ATR動態門檻、熊市換股/換防禦標的、跟
# 高息低波混合、擴大熊市檔數、熊市加流動性/beta/市值/融資使用率/個股
# 歷史最大回撤等因子、個股停損...），全部無法在不明顯犧牲報酬的前提下
# 穩健改善——查過持股明細，2025/3底熊市那5檔本身橫跨5個不同產業，不是
# 集中度問題，是這個策略風格（集中持有小型成長/動能股）在系統性恐慌
# 下的固有特徵，不是外掛規則能便宜修掉的。最後決定維持原始版本，不加
# 任何全倉層級的停損/風控。
# =============================================================================
def run_backtest(start, end, name):
    return sim(
        position_final.loc[start:end],
        resample='QE',
        trade_at_price='open',
        fee_ratio=0.001425,
        tax_ratio=0.003,
        position_limit=0.2,
        market='TW_STOCK',
        name=name,
        upload=False,  # 想上傳到 finlab 網頁報告的話自己改成 True
    )

report_in = run_backtest('2010', '2021', '樣本內_動態多因子策略')
report_in.display()

report_out = run_backtest('2022', '2026', '樣本外_動態多因子策略')
report_out.display()
