"""
下單執行層。PaperBroker（模擬成交）與 LiveBroker（幣安實盤）介面相同：
  sell(幣, 數量, 參考價) -> (成交數量, 到手 USDT)
  buy(幣, USDT, 參考價)  -> (到手數量, 花費 USDT)
  ensure_usdt(需求)      -> 可用 USDT（實盤會先從活期贖回）
  park_usdt()            -> 把多餘 USDT 存回活期（紙上交易不動作）
無法成交的零頭（低於最低下單金額）回傳 (要求數量, 0)，由引擎記為零頭並從帳上移除。
"""
import time

from binance_api import BinanceError, floor_step

MIN_NOTIONAL_DEFAULT = 5.0


class PaperBroker:
    def __init__(self, wallet, fee=0.001, log=print):
        self.w = wallet                      # {"USDT": 數量, 幣: 數量}，存在 state 裡
        self.fee, self.log = fee, log

    def sell(self, coin, qty, price):
        qty = min(qty, self.w.get(coin, 0.0))
        if not price or qty <= 0:
            return 0.0, 0.0
        if qty * price < MIN_NOTIONAL_DEFAULT:
            self.log(f"    {coin} 剩餘 {qty * price:.2f} USDT 低於最低下單金額，記為零頭")
            self.w[coin] -= qty
            return qty, 0.0
        usdt = qty * price * (1 - self.fee)
        self.w[coin] -= qty
        if self.w[coin] <= 1e-12:
            del self.w[coin]
        self.w["USDT"] = self.w.get("USDT", 0.0) + usdt
        return qty, usdt

    def buy(self, coin, usdt, price):
        usdt = min(usdt, self.w.get("USDT", 0.0))
        if not price or usdt < MIN_NOTIONAL_DEFAULT:
            return 0.0, 0.0
        qty = usdt / price * (1 - self.fee)
        self.w["USDT"] -= usdt
        self.w[coin] = self.w.get(coin, 0.0) + qty
        return qty, usdt

    def ensure_usdt(self, need):
        return self.w.get("USDT", 0.0)

    def park_usdt(self):
        pass


class LiveBroker:
    def __init__(self, client, rules, log=print, buffer_usdt=20.0, use_earn=True):
        self.c, self.rules, self.log = client, rules, log
        self.buffer, self.use_earn = buffer_usdt, use_earn
        self._pid = None

    def _rule(self, coin):
        return self.rules.get(coin + "USDT", {"step": "0.00000001", "min_qty": 0.0,
                                              "min_notional": MIN_NOTIONAL_DEFAULT})

    def _product(self):
        if self._pid is None:
            self._pid = self.c.earn_product_id("USDT")
        return self._pid

    def sell(self, coin, qty, price):
        r = self._rule(coin)
        qty = min(qty, self.c.free(coin))
        q = floor_step(qty, r["step"])
        if float(q) <= 0 or float(q) < r["min_qty"] or float(q) * (price or 0) < r["min_notional"]:
            self.log(f"    {coin} 數量 {qty:.8g} 低於最低下單量或金額，記為零頭")
            return qty, 0.0
        try:
            o = self.c.market_sell(coin + "USDT", q)
        except BinanceError as e:
            self.log(f"    !! 賣出 {coin} 失敗：{e}")
            return 0.0, 0.0
        got = float(o.get("executedQty", 0))
        usdt = float(o.get("cummulativeQuoteQty", 0))
        usdt -= sum(float(f["commission"]) for f in o.get("fills", []) if f.get("commissionAsset") == "USDT")
        self.log(f"    賣出 {coin} {got:.8g} → {usdt:,.2f} USDT")
        return got, usdt

    def buy(self, coin, usdt, price):
        r = self._rule(coin)
        if usdt < r["min_notional"]:
            return 0.0, 0.0
        amt = f"{int(usdt * 100) / 100:.2f}"                  # USDT 取到小數 2 位（無條件捨去）
        try:
            o = self.c.market_buy_quote(coin + "USDT", amt)
        except BinanceError as e:
            self.log(f"    !! 買進 {coin} 失敗：{e}")
            return 0.0, 0.0
        got = float(o.get("executedQty", 0))
        got -= sum(float(f["commission"]) for f in o.get("fills", []) if f.get("commissionAsset") == coin)
        spent = float(o.get("cummulativeQuoteQty", 0))
        self.log(f"    買進 {coin} {got:.8g}，花費 {spent:,.2f} USDT")
        return got, spent

    def ensure_usdt(self, need):
        free = self.c.free("USDT")
        if free >= need or not self.use_earn:
            return free
        amt = need - free + 1.0
        try:
            earn = self.c.earn_balance("USDT")
            amt = min(amt, earn)
            if amt > 0.01:
                self.c.earn_redeem(self._product(), f"{amt:.8f}")
                self.log(f"    從活期贖回 {amt:,.2f} USDT")
        except BinanceError as e:
            self.log(f"    !! 活期贖回失敗：{e}")
        for _ in range(12):                                   # 等到帳（最多約 60 秒）
            free = self.c.free("USDT")
            if free >= need:
                break
            time.sleep(5)
        return free

    def park_usdt(self):
        if not self.use_earn:
            return
        free = self.c.free("USDT")
        amt = free - self.buffer
        if amt < 1.0:
            return
        try:
            self.c.earn_subscribe(self._product(), f"{int(amt * 100) / 100:.2f}")
            self.log(f"    {amt:,.2f} USDT 存入活期")
        except BinanceError as e:
            self.log(f"    !! 活期申購失敗：{e}")
