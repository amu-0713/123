"""
資料下載與本地快取。

目的：避免每次跑回測/參數測試都重複呼叫 finlab API，消耗使用量。
第一次執行會呼叫 data.get() 抓取，並存成本地 parquet 快取；
之後只要快取檔都存在，就直接讀本地檔案，"不會再打 API"。

如果之後 finlab 更新了資料，想抓最新的，呼叫
    load_all_data(force_refresh=True)
才會重新下載並覆蓋快取。
"""
import pandas as pd
from finlab.dataframe import FinlabDataFrame

from config import CACHE_DIR, DATA_START, DATA_END, FINLAB_TOKEN

# 欄位代稱 -> finlab 資料集名稱
FIELDS = {
    "price": "price:收盤價",
    "open_p": "price:開盤價",
    "pe": "price_earning_ratio:本益比",
    "rev_m": "monthly_revenue:當月營收",
    "vol": "price:成交金額",
    "adj_close": "etl:adj_close",  # 還原股價，只用來算動能/回撤/相關性等報酬類因子
    "high": "price:最高價",   # 算 ATR 用
    "low": "price:最低價",    # 算 ATR 用
    "yield_ratio": "price_earning_ratio:殖利率(%)",  # 高息低波策略用
    "margin_usage": "margin_transactions:融資使用率",  # 熊市防禦因子：融資使用率越高，系統性下跌時越容易被斷頭放大跌幅
    "market_value": "etl:market_value",  # 熊市防禦因子：市值大小
    "rd_ratio": "fundamental_features:研究發展費用率",
    "pm_ratio": "fundamental_features:管理費用率",
    "eq_ratio": "fundamental_features:淨值除資產",
    "volume_shares": "price:成交股數",
}

# 不是「日期 x 股票代碼」矩陣的資料集，格式不一樣，不能套用同一套
# .loc[DATA_START:DATA_END] + columns.astype(str) 的處理邏輯。
NON_TS_FIELDS = {
    "company_info": "company_basic_info",  # 高息低波策略拿來判斷金融股用
}


def _cache_path(name: str):
    return CACHE_DIR / f"{name}.parquet"


def _login_if_needed():
    import finlab

    if FINLAB_TOKEN:
        finlab.login(FINLAB_TOKEN)
        return

    # 沒帶 FINLAB_TOKEN：finlab 本身在需要驗證時（例如 sim() 內部抓
    # benchmark/還原股價等資料）會自動跳出瀏覽器登入流程，並把登入
    # 憑證存在本機，下次執行就會直接沿用，不用每次都帶 token。
    # 這裡不主動擋下來，讓 finlab 自己的機制接手。


def load_all_data(force_refresh: bool = False) -> dict:
    """回傳 {name: DataFrame}，欄位（股票代碼）已統一轉成字串。

    註：這裡不管有沒有命中我們自己的 parquet 快取，都會嘗試登入
    （有帶 FINLAB_TOKEN 就用它登入；沒帶就交給 finlab 自己的機制）。
    因為 finlab.backtest.sim() 內部還會另外抓 benchmark / 還原股價 /
    注意股票等資料（跟這裡快取的 5 個欄位是不同層，我們管不到），
    那一層一樣需要登入才能正常運作，跟本地快取是否命中無關。
    這裡的登入只是驗證身份，不會重複消耗 data.get() 的資料用量。
    """
    _login_if_needed()

    # 每個欄位各自判斷有沒有快取，只下載缺的那幾個（例如之後在 FIELDS
    # 裡新增欄位，不會連已經快取好的其他欄位也重抓一次）。
    all_names = {**FIELDS, **NON_TS_FIELDS}
    missing = [name for name in all_names if force_refresh or not _cache_path(name).exists()]

    if missing:
        print(f"[data_loader] 缺少快取：{missing}，呼叫 finlab API 下載中...")
        from finlab import data

        for name in missing:
            key = all_names[name]
            df = data.get(key)
            if name in FIELDS:
                df = df.loc[DATA_START:DATA_END]
                df.columns = df.columns.astype(str)
            df.to_parquet(_cache_path(name))
        print("[data_loader] 下載完成，已存入本地快取，之後執行不會再重複下載。")
    else:
        print("[data_loader] 偵測到完整本地快取，直接讀取，未呼叫 finlab API 下載資料。")

    # data.get() 回傳的是 finlab 自己的 FinlabDataFrame，>、&、/ 等運算子
    # 被覆寫成「兩邊取日期聯集 + 前向填補 + 欄位取交集」再運算，這對月營收
    # 這種稀疏更新的資料是必要的（否則非公告日全部變 NaN）。純 pandas
    # 的 read_parquet 讀回來是普通 DataFrame，會遺失這個行為，所以這裡
    # 讀回來後要重新包成 FinlabDataFrame，維持跟即時抓取時一致的運算語意。
    # NON_TS_FIELDS（例如 company_basic_info）不是時間序列矩陣，不需要
    # 也不能套用這個行為，讀回一般 DataFrame 就好。
    result = {name: FinlabDataFrame(pd.read_parquet(_cache_path(name))) for name in FIELDS}
    for name in NON_TS_FIELDS:
        result[name] = pd.read_parquet(_cache_path(name))

    return result
