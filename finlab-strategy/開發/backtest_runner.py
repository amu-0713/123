"""
封裝 finlab.backtest.sim。

預設 upload=False（不發布到 finlab 雲端 dashboard），
且只用文字/終端機圖表輸出（to_terminal / get_metrics），
不呼叫會產生網頁圖表的 display()，避免非必要的雲端互動與額外消耗。
"""
import json

from finlab.backtest import sim


def run_backtest(position, start: str, end: str, name: str, params: dict, resample="QE"):
    """resample 可以是 'QE' 這種字串，也可以是明確的日期清單
    （risk_overlay.build_stop_and_switch_position 產生的 resample_dates）。
    傳日期清單時，要注意 position 本身（不是 .loc[start:end] 之後）
    要涵蓋這些日期，sim() 才找得到。"""
    sliced = position.loc[start:end]
    # resample 若是日期清單，不用在這裡自己按 start/end 篩，sim() 內部
    # 會用切片後 position 自己的 index 範圍去篩，這裡篩反而容易因為
    # end 年份被 pandas 解讀成該年 1/1 而誤刪掉後面的日期。

    report = sim(
        sliced,
        resample=resample,
        trade_at_price="open",
        fee_ratio=params["fee_ratio"],
        tax_ratio=params["tax_ratio"],
        position_limit=params["position_limit"],
        market="TW_STOCK",
        name=name,
        upload=False,
    )
    return report


def get_key_metrics(report) -> dict:
    """抓出比較用的關鍵指標，扁平化方便做表格。"""
    m = report.get_metrics()
    return {
        "CAGR": m["profitability"]["annualReturn"],
        "Sharpe": m["ratio"]["sharpeRatio"],
        "Calmar": m["ratio"]["calmarRatio"],
        "MaxDD": m["risk"]["maxDrawdown"],
        "WinRate": m["winrate"]["winRate"],
        "AvgNStock": m["profitability"]["avgNStock"],
        "Capacity": m["liquidity"]["capacity"],
    }


def print_summary(report, title: str):
    print(f"\n===== {title} =====")
    try:
        metrics = report.get_metrics()
        print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    except Exception as e:
        print(f"(get_metrics 失敗: {e})")

    try:
        report.to_terminal()
    except Exception as e:
        print(f"(to_terminal 失敗: {e})")
