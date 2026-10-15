"""
幣安現貨 REST API 的最小封裝（只用標準函式庫）。

只用到：
  公開   GET  /api/v3/exchangeInfo, /api/v3/klines, /api/v3/ticker/price
  簽名   GET  /api/v3/account
         POST /api/v3/order                                 （市價單）
         GET  /sapi/v1/simple-earn/flexible/list            （活期產品）
         GET  /sapi/v1/simple-earn/flexible/position        （活期持倉）
         POST /sapi/v1/simple-earn/flexible/subscribe       （申購）
         POST /sapi/v1/simple-earn/flexible/redeem          （贖回）

API key 權限只需要「讀取」與「現貨交易」；不要開提幣。
上線前請對照幣安官方文件確認端點與參數（幣安偶爾會調整 API）。
"""
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from decimal import Decimal, ROUND_DOWN

BASE = "https://api.binance.com"


class BinanceError(RuntimeError):
    pass


class Client:
    def __init__(self, key=None, secret=None, base=BASE, recv_window=10000, timeout=30):
        self.key, self.secret, self.base = key, secret, base
        self.recv_window, self.timeout = recv_window, timeout

    # ------------------------------------------------------------ 底層
    def _request(self, method, path, params=None, signed=False, retries=3):
        """
        重試規則：
          - 429/418（限速，請求被拒絕、未執行）→ 等待後重試
          - 網路錯誤、5xx → 只有 GET 重試。POST（下單、申購、贖回）不重試，
            因為請求可能其實已經執行，重送會重複下單；由下一次執行時的對帳處理。
          - 其他 4xx（參數錯、餘額不足等）→ 直接拋出
        """
        if signed and not (self.key and self.secret):
            raise BinanceError("需要 API key/secret")
        headers = {"User-Agent": "quant-bot/1.0"}
        if self.key:
            headers["X-MBX-APIKEY"] = self.key
        last = None
        for i in range(retries):
            p = dict(params or {})
            if signed:                                     # 每次重送都重新簽名（時間戳記會變）
                p["timestamp"] = int(time.time() * 1000)
                p["recvWindow"] = self.recv_window
            q = urllib.parse.urlencode(p)
            if signed:
                q += "&signature=" + hmac.new(self.secret.encode(), q.encode(), hashlib.sha256).hexdigest()
            url = f"{self.base}{path}" + (f"?{q}" if q else "")
            try:
                req = urllib.request.Request(url, method=method, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                last = BinanceError(f"{method} {path} → {e.code} {body}")
                if e.code in (418, 429):
                    time.sleep(10 * (i + 1))
                    continue
                if e.code < 500 or method != "GET":
                    raise last
            except Exception as e:                          # 網路錯誤、逾時
                last = BinanceError(f"{method} {path} → {e}")
                if method != "GET":
                    raise last
            time.sleep(2 ** i)
        raise last

    # ------------------------------------------------------------ 公開資料
    def exchange_info(self):
        return self._request("GET", "/api/v3/exchangeInfo")

    def klines(self, symbol, interval="1d", limit=220):
        """回傳 list：[open_time, open, high, low, close, volume, close_time, quote_volume, trades,
        taker_buy_base, taker_buy_quote, ignore]"""
        return self._request("GET", "/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})

    def prices(self):
        return {x["symbol"]: float(x["price"]) for x in self._request("GET", "/api/v3/ticker/price")}

    # ------------------------------------------------------------ 帳戶
    def balances(self):
        """{資產: 可用+凍結}，只回傳 > 0 的資產"""
        acc = self._request("GET", "/api/v3/account", {"omitZeroBalances": "true"}, signed=True)
        out = {}
        for b in acc["balances"]:
            v = float(b["free"]) + float(b["locked"])
            if v > 0:
                out[b["asset"]] = v
        return out

    def free(self, asset):
        acc = self._request("GET", "/api/v3/account", {"omitZeroBalances": "true"}, signed=True)
        for b in acc["balances"]:
            if b["asset"] == asset:
                return float(b["free"])
        return 0.0

    # ------------------------------------------------------------ 下單
    def market_sell(self, symbol, quantity):
        return self._request("POST", "/api/v3/order", {
            "symbol": symbol, "side": "SELL", "type": "MARKET", "quantity": quantity,
            "newOrderRespType": "FULL"}, signed=True)

    def market_buy_quote(self, symbol, quote_amount):
        return self._request("POST", "/api/v3/order", {
            "symbol": symbol, "side": "BUY", "type": "MARKET", "quoteOrderQty": quote_amount,
            "newOrderRespType": "FULL"}, signed=True)

    # ------------------------------------------------------------ 活期理財
    def earn_product_id(self, asset="USDT"):
        rows = self._request("GET", "/sapi/v1/simple-earn/flexible/list", {"asset": asset}, signed=True).get("rows", [])
        if not rows:
            raise BinanceError(f"找不到 {asset} 活期產品")
        return rows[0]["productId"]

    def earn_balance(self, asset="USDT"):
        rows = self._request("GET", "/sapi/v1/simple-earn/flexible/position", {"asset": asset}, signed=True).get("rows", [])
        return sum(float(r.get("totalAmount", 0)) for r in rows)

    def earn_subscribe(self, product_id, amount):
        return self._request("POST", "/sapi/v1/simple-earn/flexible/subscribe",
                             {"productId": product_id, "amount": amount}, signed=True)

    def earn_redeem(self, product_id, amount=None):
        p = {"productId": product_id}
        if amount is None:
            p["redeemAll"] = "true"
        else:
            p["amount"] = amount
        return self._request("POST", "/sapi/v1/simple-earn/flexible/redeem", p, signed=True)


# ================================================================ 交易規則（數量精度、最低金額）

def symbol_rules(exchange_info, quote="USDT"):
    """{symbol: dict(base, status, step, min_qty, min_notional)}，只取 quote 計價的現貨交易對"""
    out = {}
    for s in exchange_info["symbols"]:
        if s.get("quoteAsset") != quote or not s.get("isSpotTradingAllowed", True):
            continue
        r = {"base": s["baseAsset"], "status": s["status"], "step": "1", "min_qty": 0.0, "min_notional": 5.0}
        for f in s.get("filters", []):
            t = f["filterType"]
            if t in ("LOT_SIZE", "MARKET_LOT_SIZE") and float(f.get("stepSize", 0)) > 0:
                r["step"] = f["stepSize"] if t == "LOT_SIZE" else r["step"]
                r["min_qty"] = max(r["min_qty"], float(f.get("minQty", 0)))
            if t in ("NOTIONAL", "MIN_NOTIONAL"):
                r["min_notional"] = float(f.get("minNotional", r["min_notional"]))
        out[s["symbol"]] = r
    return out


def floor_step(qty, step):
    """數量往下取到 stepSize 的整數倍，回傳字串（避免浮點誤差）"""
    st = Decimal(str(step)).normalize()
    q = (Decimal(str(qty)) / st).to_integral_value(rounding=ROUND_DOWN) * st
    return format(q.normalize(), "f") if q > 0 else "0"


def fill_summary(order):
    """市價單回傳 → (成交數量, 成交金額, 以基礎幣扣的手續費)"""
    qty = float(order.get("executedQty", 0))
    quote = float(order.get("cummulativeQuoteQty", 0))
    fee_base = sum(float(f["commission"]) for f in order.get("fills", [])
                   if f.get("commissionAsset") == order["symbol"][:-4])
    return qty, quote, fee_base
