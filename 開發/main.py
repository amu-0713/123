"""
正式版入口程式。

策略 = 原本的多因子選股 + 季度再平衡，沒有加任何全倉層級的停損/風控
overlay（測過十幾種都無法在不犧牲報酬的前提下穩健改善 2025 那次的
回撤，詳見對話紀錄與 README.md，最後決定維持原始邏輯）。

流程：抓資料（有本地快取就不重複打 finlab API）
      -> 依 config.PARAMS 建立策略部位
      -> 分別跑「樣本內」與「樣本外」回測，純文字輸出結果

用法：
    (第一次，需要抓資料) $env:FINLAB_TOKEN = "你的token"; python main.py
    (之後，有快取+已登入)  python main.py
"""
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import PARAMS, IN_SAMPLE, OUT_SAMPLE
from data_loader import load_all_data
from strategy import build_position
from backtest_runner import run_backtest, print_summary


def main():
    data = load_all_data()
    position_final, raw_position = build_position(data, PARAMS)

    reports = {}
    for sample_name, (start, end) in [("樣本內", IN_SAMPLE), ("樣本外", OUT_SAMPLE)]:
        report = run_backtest(
            position_final, start, end, name=f"{sample_name}_動態多因子策略", params=PARAMS, resample="QE"
        )
        print_summary(report, f"{sample_name} {start}~{end}")
        reports[sample_name] = report

    return reports


if __name__ == "__main__":
    main()
