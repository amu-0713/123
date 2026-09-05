"""集中管理路徑、樣本切分、與可調策略參數。"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# --- finlab 登入 token：不寫死在程式碼裡，執行前用環境變數帶入 ---
# PowerShell:  $env:FINLAB_TOKEN = "你的token"
# Bash:        export FINLAB_TOKEN="你的token"
FINLAB_TOKEN = os.environ.get("FINLAB_TOKEN")

# --- 資料抓取範圍（含暖身期，讓 120 日均線 / 12 月營收年增率等指標
#     在樣本內起點 2010 就已經有值，不會因為指標還沒熱機而失真） ---
DATA_START = "2006"
DATA_END = "2026"

# --- 樣本切分 ---
IN_SAMPLE = ("2010", "2021")   # 樣本內：拿來挑參數/迭代
OUT_SAMPLE = ("2022", "2026")  # 樣本外：只驗證，不根據結果回頭調參數

# --- 策略參數（之後做敏感度測試只改這裡，不用動 strategy.py 的邏輯） ---
PARAMS = dict(
    ma_windows=(20, 60, 120),      # 個股多頭排列均線 (短, 中, 長)
    mkt_ma_windows=(30, 60),       # 大盤牛熊判斷均線 (短, 長)
    rev_ma_window=3,               # 營收移動平均月數
    rev_growth_lag=12,             # 營收年增率比較月數
    peg_range=(0.2, 1.8),          # PEG 篩選區間
    rs_window=80,                  # 動能因子 lookback（交易日）
    dd_window=20,                  # 下檔波動 window
    corr_window=60,                # 與大盤相關性 window
    rev_high_window=12,            # 營收創高比較月數
    liq_window=20,
    liq_min_value=1e6,             # 20 日內每日成交金額門檻（元）
    n_bull=16,
    n_bear=5,
    weights={                      # 牛熊動態因子權重
        "rs":   {"bull": 0.3, "bear": 0.3},
        "peg":  {"bull": 0.3, "bear": 0.0},
        "corr": {"bull": 0.0, "bear": 0.3},
        "dd":   {"bull": 0.4, "bear": 0.4},
    },
    limit_pct_new=0.095,           # 2015/06 起漲停判定門檻
    limit_pct_old=0.065,           # 2015/06 前漲停判定門檻
    limit_pct_switch_date="2015-05-31",
    position_limit=0.2,
    fee_ratio=0.001425,
    tax_ratio=0.003,

    # --- 全倉停損換股：最終決定不採用 ---
    # 測過大盤回撤15%觸發、留倉換股，樣本內/外一度看起來有效，但後續
    # 為了處理 2025 那次大回撤又測了十幾種角度（ATR、熊市換股、防禦
    # 切換、跟高息低波混合、擴大熊市檔數、熊市加流動性/beta/市值/融資/
    # 歷史回撤等因子、個股停損...），沒有一個能在不明顯犧牲報酬的前提
    # 下穩健改善，本質上是策略風格（集中持有小型成長/動能股）在系統性
    # 恐慌下的固有特徵，不是外掛規則能便宜修掉的。最後維持原始版本，
    # 不加任何全倉層級的停損/風控 overlay。
    portfolio_stop_dd=None,
)

# --- 全倉停損（可選）相關參數說明，供 EXPERIMENTS 覆蓋測試用 ---
# portfolio_stop_dd:      觸發停損的回撤幅度，None = 關閉
# portfolio_stop_recover: 觸發後回撤縮小到這個幅度以內才解除冷卻
# stop_trigger_source:    "portfolio"(自己組合回撤) 或 "market"(大盤0050回撤)
# stop_action:            "switch"(立刻換成當下最新目標持股) 或
#                          "cash"(出場 stop_cash_days 天後才用最新目標重新進場)
# stop_cash_days:         stop_action="cash" 時的觀望交易日數

# --- 一次跑過的實驗清單：(名稱, 覆蓋 PARAMS 的 dict) ---
# PARAMS 現在預設就是「大盤回撤15%觸發、留倉換股」這個定案版本。
# 這裡的 EXPERIMENTS 拿它當基準，"正式版" 是空 dict，"無停損_對照組" 是
# 特地關掉 portfolio_stop_dd 的對照版，其餘是之前測過、留著備查的變體。
EXPERIMENTS = [
    ("正式版", {}),
    ("無停損_對照組", {"portfolio_stop_dd": None}),
    ("還原股價因子", {"use_adj_price": True}),
    ("牛熊訊號5日確認", {"regime_confirm_days": 5}),
    ("流動性門檻x5", {"liq_min_value": 5e6}),
    ("熊市7檔", {"n_bear": 7}),
    ("停損換股_組合回撤15%(非大盤)", {"stop_trigger_source": "portfolio"}),
    ("停損出場20日(非留倉)", {"stop_action": "cash", "stop_cash_days": 20}),
    ("停損換股_大盤回撤10%", {"portfolio_stop_dd": 0.10}),
    ("停損換股_大盤回撤20%", {"portfolio_stop_dd": 0.20}),
    ("停損換股_ATR_k8", {"stop_trigger_source": "market_atr", "atr_window": 20, "atr_k": 8}),
    ("停損換股_ATR_k10", {"stop_trigger_source": "market_atr", "atr_window": 20, "atr_k": 10}),
    ("停損換股_ATR_k12", {"stop_trigger_source": "market_atr", "atr_window": 20, "atr_k": 12}),
    ("停損換股_ATR_k15", {"stop_trigger_source": "market_atr", "atr_window": 20, "atr_k": 15}),
]
