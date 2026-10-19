#!/usr/bin/env python3
"""bot 的單元測試（不需要網路）：python bot/test_bot.py"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import engine  # noqa: E402
from binance_api import floor_step, symbol_rules  # noqa: E402
from brokers import LiveBroker, PaperBroker  # noqa: E402
import run as runner  # noqa: E402


def test_floor_step():
    assert floor_step(1.23456789, "0.001") == "1.234"
    assert floor_step(0.0009, "0.001") == "0"
    assert floor_step(12345.6, "1.00000000") == "12345"
    assert floor_step(0.1 + 0.2, "0.1") == "0.3"


def test_symbol_rules():
    info = {"symbols": [{"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT", "status": "TRADING",
                         "isSpotTradingAllowed": True,
                         "filters": [{"filterType": "LOT_SIZE", "stepSize": "0.0001", "minQty": "0.0001"},
                                     {"filterType": "NOTIONAL", "minNotional": "5"}]},
                        {"symbol": "ETHBTC", "baseAsset": "ETH", "quoteAsset": "BTC", "status": "TRADING", "filters": []}]}
    r = symbol_rules(info)
    assert list(r) == ["ETHUSDT"] and r["ETHUSDT"]["step"] == "0.0001" and r["ETHUSDT"]["min_notional"] == 5.0


class MockClient:
    """模擬實盤 API：記錄下單，回傳類似幣安的成交回報"""
    def __init__(self, bal, earn=0.0, prices=None):
        self.bal, self.earn, self.px, self.orders = dict(bal), earn, prices or {}, []

    def free(self, a): return self.bal.get(a, 0.0)
    def balances(self): return {k: v for k, v in self.bal.items() if v > 0}
    def earn_product_id(self, a): return "USDT001"
    def earn_balance(self, a): return self.earn

    def earn_redeem(self, pid, amount=None):
        amt = self.earn if amount is None else float(amount)
        self.earn -= amt
        self.bal["USDT"] = self.bal.get("USDT", 0) + amt

    def earn_subscribe(self, pid, amount):
        self.earn += float(amount)
        self.bal["USDT"] -= float(amount)

    def market_sell(self, sym, qty):
        c, q = sym[:-4], float(qty)
        usdt = q * self.px[c]
        self.bal[c] -= q
        self.bal["USDT"] = self.bal.get("USDT", 0) + usdt * 0.999
        self.orders.append(("SELL", sym, qty))
        return {"symbol": sym, "executedQty": qty, "cummulativeQuoteQty": str(usdt),
                "fills": [{"commission": str(usdt * 0.001), "commissionAsset": "USDT"}]}

    def market_buy_quote(self, sym, amt):
        c, u = sym[:-4], float(amt)
        q = u / self.px[c]
        self.bal["USDT"] -= u
        self.bal[c] = self.bal.get(c, 0) + q * 0.999
        self.orders.append(("BUY", sym, amt))
        return {"symbol": sym, "executedQty": str(q), "cummulativeQuoteQty": amt,
                "fills": [{"commission": str(q * 0.001), "commissionAsset": c}]}


def test_live_broker_orders_and_earn():
    rules = {"ETHUSDT": {"step": "0.001", "min_qty": 0.001, "min_notional": 5.0},
             "XYZUSDT": {"step": "1", "min_qty": 1, "min_notional": 5.0}}
    c = MockClient({"USDT": 10.0, "ETH": 1.23456, "XYZ": 3}, earn=1000.0, prices={"ETH": 2000.0, "XYZ": 1.0})
    b = LiveBroker(c, rules, log=lambda m: None, buffer_usdt=20)
    q, u = b.sell("ETH", 1.23456, 2000.0)
    assert c.orders[-1] == ("SELL", "ETHUSDT", "1.234") and abs(q - 1.234) < 1e-12 and abs(u - 2468 * 0.999) < 1e-6
    q, u = b.sell("XYZ", 3, 1.0)                              # 3 USDT < 最低 5 → 零頭，不下單
    assert u == 0.0 and len(c.orders) == 1
    have = b.ensure_usdt(3000.0)                              # 現貨不足 → 從活期贖回
    assert have >= 2465 and c.earn < 1000
    q, spent = b.buy("ETH", 1000.0, 2000.0)
    assert c.orders[-1] == ("BUY", "ETHUSDT", "1000.00") and abs(q - 0.5 * 0.999) < 1e-9
    b.park_usdt()                                             # 多餘 USDT 存回活期，保留 20
    assert abs(c.free("USDT") - 20) < 0.01


def test_apply_fills_two_ledgers():
    st = engine.new_state(1000)
    st["ledgers"]["A"]["pos"]["ETH"] = {"qty": 1.0, "age": 3}
    st["ledgers"]["B"]["pos"]["ETH"] = {"qty": 3.0, "age": 0}
    sells = [("A", "ETH", 1.0, "跌破", True), ("B", "ETH", 3.0, "跌破", True)]
    engine.apply_fills(st, sells, [], {"ETH": (4.0, 400.0)}, {}, {}, {"ETH": 100.0}, log=lambda m: None)
    assert "ETH" not in st["ledgers"]["A"]["pos"] and "ETH" not in st["ledgers"]["B"]["pos"]
    assert abs(st["ledgers"]["A"]["usdt"] - 600) < 1e-9 and abs(st["ledgers"]["B"]["usdt"] - 800) < 1e-9
    buys = [("A", "BTC", 300.0), ("B", "BTC", 100.0)]
    engine.apply_fills(st, [], buys, {}, {"BTC": (0.004, 400.0)}, {"A": {"BTC": 0}}, {"BTC": 1e5}, log=lambda m: None)
    assert abs(st["ledgers"]["A"]["pos"]["BTC"]["qty"] - 0.003) < 1e-12
    assert abs(st["ledgers"]["B"]["pos"]["BTC"]["qty"] - 0.001) < 1e-12
    assert abs(st["ledgers"]["A"]["usdt"] - 300) < 1e-9


def test_paper_dust_removed():
    st = engine.new_state(1000)
    st["ledgers"]["A"]["pos"]["XYZ"] = {"qty": 2.0, "age": 5}
    w = {"USDT": 1000.0, "XYZ": 2.0}
    pb = PaperBroker(w, log=lambda m: None)
    q, u = pb.sell("XYZ", 2.0, 1.0)                           # 2 USDT < 5 → 零頭
    engine.apply_fills(st, [("A", "XYZ", 2.0, "跌破", True)], [], {"XYZ": (q, u)}, {}, {}, {"XYZ": 1.0},
                       log=lambda m: None)
    assert "XYZ" not in st["ledgers"]["A"]["pos"]              # 零頭從帳上移除
    assert st["ledgers"]["A"]["usdt"] == 500.0                 # 沒有賣到錢，USDT 不變


def test_vol_scale():
    navs = [1000.0]
    for i in range(19):                                        # 每天交替 +5% / -5%
        navs.append(navs[-1] * (1.05 if i % 2 else 0.95))
    hist = [{"date": str(d.date()), "nav": v, "scale": 1.0}
            for d, v in zip(pd.date_range("2024-01-01", periods=20), navs)]
    s = engine.vol_scale(hist)
    assert 0 < s < 1                                           # 日波動 5% ≈ 年化 95% → 縮放約 0.42
    assert engine.vol_scale(hist[:5]) == 1.0                   # 資料不足 → 不縮放


def test_reconcile_baseline():
    st = engine.new_state(1000)
    c = MockClient({"USDT": 5000.0}, earn=0.0)
    runner.reconcile_live(st, c, log=lambda m: None)          # 第一次：帳戶有 5000，但只管理 1000
    assert abs(sum(L["usdt"] for L in st["ledgers"].values()) - 1000) < 1e-9
    st["usdt_real_last"] = 5000.0
    c.earn = 10.0                                              # 活期利息 10
    runner.reconcile_live(st, c, log=lambda m: None)
    assert abs(sum(L["usdt"] for L in st["ledgers"].values()) - 1010) < 1e-9


def test_backup_state():
    import json, tempfile
    with tempfile.TemporaryDirectory() as d:
        sp, lp, bd = os.path.join(d, "state_live.json"), os.path.join(d, "x_live.log"), os.path.join(d, "bk")
        json.dump({"a": 1}, open(sp, "w"))
        open(lp, "w").write("log")
        for day in ["2026-01-01", "2026-01-02", "2026-01-03"]:
            runner.backup_state(sp, lp, bd, day, keep=2, log=lambda m: None)
        assert json.load(open(os.path.join(bd, "state_live.json"))) == {"a": 1}
        assert sorted(os.listdir(os.path.join(bd, "history"))) == ["state_live_2026-01-02.json", "state_live_2026-01-03.json"]
        assert os.path.exists(os.path.join(bd, "x_live.log"))
        runner.backup_state(sp, lp, lp, "2026-01-04", log=lambda m: None)   # 備份路徑錯誤只警告


if __name__ == "__main__":
    n = 0
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            n += 1
            print(f"ok  {name}")
    print(f"全部 {n} 項測試通過")
