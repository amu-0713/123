"""
一次跑過 config.EXPERIMENTS 裡列的所有變體，樣本內/樣本外各跑一次，
印出比較表格。全程用本地快取資料（+ sim() 內部自己的資料，那層是
finlab 自己快取，不會重複消耗我們自己的 5+1 個欄位額度）。
"""
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from config import PARAMS, IN_SAMPLE, OUT_SAMPLE, EXPERIMENTS
from data_loader import load_all_data
from strategy import build_position
from risk_overlay import build_overlay_position
from backtest_runner import run_backtest, get_key_metrics


def main():
    data = load_all_data()
    price = data["price"]
    adj_close = data.get("adj_close")
    high = data.get("high")
    low = data.get("low")

    rows = []
    for exp_name, overrides in EXPERIMENTS:
        params = {**PARAMS, **overrides}
        position_final, raw_position = build_position(data, params)

        for sample_name, (start, end) in [("樣本內", IN_SAMPLE), ("樣本外", OUT_SAMPLE)]:
            position, resample_dates = build_overlay_position(
                position_final, raw_position, price, params, adj_close=adj_close, high=high, low=low
            )
            name = f"{sample_name}_{exp_name}"
            report = run_backtest(position, start, end, name=name, params=params, resample=resample_dates)
            metrics = get_key_metrics(report)
            metrics.update(sample=sample_name, experiment=exp_name)
            rows.append(metrics)
            print(
                f"{sample_name:5s} {exp_name:22s} "
                f"CAGR={metrics['CAGR']:6.1%}  Sharpe={metrics['Sharpe']:5.2f}  "
                f"MaxDD={metrics['MaxDD']:7.1%}  Calmar={metrics['Calmar']:5.2f}  "
                f"WinRate={metrics['WinRate']:5.1%}  AvgN={metrics['AvgNStock']:4.1f}  "
                f"Capacity={metrics['Capacity']/1e4:8.0f}萬"
            )

    return rows


if __name__ == "__main__":
    main()
