"""
策略引擎：日 K → 訊號 → 兩本帳（A 週日、B 週四）→ 訂單。

規則見 docs/strategy.md。訊號計算直接呼叫 tools/factor_research.py 的函式
（build_panel、universe、ensure_factor、composite），確保與回測一致。

時間定義：t = 最近一根「已收盤」的日 K 日期（UTC）。程式應在 UTC 00:00 之後幾分鐘執行，
此時 t = 昨天。「週日調倉」= t 是週日的那次執行（台灣時間週一早上 8 點後）。
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import factor_research as fr  # noqa: E402

PARAMS = dict(
    top_n=100, min_days=120,            # 幣池：成交額前 100、上市 ≥120 天
    k=5, pool=20,                       # 最多 5 檔，只在品質前 20 名中挑
    lv=30, liq=30, w_lv=0.5,            # 品質分數：低波動 30 日 × 0.5 + 流動性 30 日 × 0.5
    en=60, th=0.9,                      # 進場：收盤 ≥ 60 日高點 × 0.9
    rel_n=30,                           # 相對強度：幣價/BTC ≥ 30 日均線
    exit_new=20, new_weeks=2,           # 新倉（持有 < 2 次調倉）：跌破 20 日均線出場
    exit_old=40,                        # 舊倉：跌破 40 日均線出場
    vt=0.4, vt_lb=14,                   # 波動目標 40%，回看 14 天
    min_trade_frac=0.005,               # 調倉差額小於帳淨值 0.5% 就不動（省手續費）
)
LEDGER_DAYS = {"A": 6, "B": 3}          # A 帳週日、B 帳週四（0=週一 … 6=週日）
HISTORY_DAYS = 220                      # 每次抓的日 K 根數（指標最長回看 120 天 + 緩衝）


def coin_of(name):
    """面板欄位名 → 幣名（清洗時中斷後重新上架的幣會被命名為 SYM~2）"""
    return name.split("~")[0]


# ================================================================ 訊號

class Signals:
    def __init__(self, P, params=PARAMS):
        self.p = p = params
        self.P = P
        c = P["close"]
        self.U = fr.universe(P, p["top_n"], p["min_days"])
        F = {}
        lv, lq = f"lowvol_{p['lv']}", f"liq_{p['liq']}"
        fr.ensure_factor(F, P, lv)
        fr.ensure_factor(F, P, lq)
        self.S = fr.composite(F, self.U, {lv: p["w_lv"], lq: 1 - p["w_lv"]})
        B = c["BTC"]
        self.g0 = c >= p["th"] * c.rolling(p["en"]).max()
        rel = c.div(B, axis=0)
        g = self.g0 & (rel >= rel.rolling(p["rel_n"]).mean())
        g["BTC"] = self.g0["BTC"]
        self.g = g
        self.hold_new = c >= c.rolling(p["exit_new"]).mean()
        self.hold_old = c >= c.rolling(p["exit_old"]).mean()

    def targets(self, t):
        """回傳 {幣: 權重}（佔該帳淨值，未乘波動縮放），與回測 fill 模式 + 補 BTC 相同"""
        p = self.p
        s = self.S.loc[t].dropna().nlargest(p["pool"])
        s = s[self.g.loc[t].reindex(s.index).fillna(False).astype(bool)]
        pick = list(s.nlargest(p["k"]).index)
        w = {coin_of(x): 1.0 / p["k"] for x in pick}
        if sum(w.values()) < 1 - 1e-9 and bool(self.g0.loc[t].get("BTC", False)):
            w["BTC"] = w.get("BTC", 0.0) + (1 - sum(w.values()))
        return w

    def keep(self, t, coin, age):
        """持有中的幣今天是否續抱（False = 出場）"""
        col = coin if coin in self.P["close"].columns else None
        if col is None or pd.isna(self.P["close"].at[t, col]):
            return False                                    # 沒有今天的資料（下市、暫停）→ 出場
        gate = self.hold_new if age < self.p["new_weeks"] else self.hold_old
        return bool(gate.at[t, col])


def vol_scale(nav_hist, params=PARAMS, idle_rate=0.0):
    """
    依「未縮放」的策略報酬估計近 lb 天波動，回傳 min(1, 目標 / 實際波動)。
    nav_hist：[{date, nav, scale}]，scale 為當天生效的平均縮放比例；
    未縮放報酬 ≈ (帳戶報酬 − (1−scale) × 閒置收益) / scale，與回測 vol_target 用未縮放報酬一致。
    """
    lb = params["vt_lb"]
    if len(nav_hist) < lb + 1:
        return 1.0
    h = nav_hist[-(lb + 1):]
    raw = []
    for a, b in zip(h[:-1], h[1:]):
        s = max(a.get("scale", 1.0), 1e-3)
        r = b["nav"] / a["nav"] - 1
        raw.append((r - (1 - s) * idle_rate / 365) / s)
    rv = float(np.std(raw, ddof=1)) * math.sqrt(365)
    return 1.0 if rv <= 0 else min(1.0, params["vt"] / rv)


# ================================================================ 兩本帳

def new_state(capital, ledger_days=LEDGER_DAYS):
    n = len(ledger_days)
    return {
        "ledgers": {k: {"day": d, "usdt": capital / n, "pos": {}, "scale": 1.0} for k, d in ledger_days.items()},
        "nav_history": [],
        "last_t": None,
        "dust": {},
    }


def ledger_nav(L, prices):
    return L["usdt"] + sum(p["qty"] * prices.get(c, 0.0) for c, p in L["pos"].items())


def plan_day(state, sig, t, prices, params=PARAMS, idle_rate=0.0, log=print):
    """
    決定今天的交易，不執行。回傳 (sells, buys, new_ages)：
      sells: [(帳, 幣, 數量, 原因, 是否全部出清)]
      buys:  [(帳, 幣, USDT 金額)]
      new_ages: {帳: {幣: 年齡}}（只有今天調倉的帳）
    同一天被每日出場、又被調倉選中的幣，直接調整到目標部位，不先賣再買（省一次手續費）。
    """
    sells, buys, new_ages = [], [], {}
    wd = t.dayofweek
    for name, L in state["ledgers"].items():
        exiting = {}
        for coin, pos in L["pos"].items():
            if not sig.keep(t, coin, pos.get("age", 0)):
                ma = params["exit_new"] if pos.get("age", 0) < params["new_weeks"] else params["exit_old"]
                exiting[coin] = f"跌破 {ma} 日均線"
        if L["day"] != wd:                                   # 非調倉日：只做每日出場
            sells += [(name, c, L["pos"][c]["qty"], why, True) for c, why in exiting.items()]
            continue
        # 調倉日
        nav = ledger_nav(L, prices)
        scale = vol_scale(state["nav_history"], params, idle_rate)
        L["scale"] = scale
        w = sig.targets(t)
        log(f"[{name} 帳調倉] 淨值 {nav:,.2f}  波動縮放 {scale:.0%}  目標："
            + (", ".join(f"{c} {v * scale:.0%}" for c, v in w.items()) or "全部 USDT"))
        min_trade = params["min_trade_frac"] * nav
        for coin, pos in L["pos"].items():
            cur = pos["qty"] * prices.get(coin, 0.0)
            if coin not in w:
                sells.append((name, coin, pos["qty"], exiting.get(coin, "調倉移出"), True))
                continue
            tgt = w[coin] * scale * nav
            if cur - tgt > min_trade:
                sells.append((name, coin, (cur - tgt) / prices[coin], "調倉減碼", False))
        for coin, wt in w.items():
            cur = L["pos"][coin]["qty"] * prices.get(coin, 0.0) if coin in L["pos"] else 0.0
            tgt = wt * scale * nav
            if tgt - cur > min_trade:
                buys.append((name, coin, tgt - cur))
        # 年齡：續抱 +1；新買進、或今天本該出場卻又被選中（回測中等同出場後重買）→ 0
        new_ages[name] = {c: (L["pos"][c].get("age", 0) + 1 if c in L["pos"] and c not in exiting else 0)
                          for c in w}
    return sells, buys, new_ages


def apply_fills(state, sells, buys, sell_fills, buy_fills, new_ages, prices, log=print):
    """
    把實際成交分配回各帳。
      sell_fills: {幣: (成交數量, 成交金額)}；buy_fills: {幣: (到手數量, 花費金額)}
    同一幣多本帳同時交易時，依各帳下單量比例分配。全部出清但剩餘價值 < 1 USDT（或低於最低
    下單金額而賣不掉）的部分記為零頭（dust），從帳上移除。
    """
    L_ = state["ledgers"]
    by_coin = {}
    for name, coin, qty, _, full in sells:
        by_coin.setdefault(coin, []).append((name, qty, full))
    for coin, parts in by_coin.items():
        req = sum(q for _, q, _ in parts)
        got_q, got_usdt = sell_fills.get(coin, (0.0, 0.0))
        for name, q, full in parts:
            pos = L_[name]["pos"].get(coin)
            if pos is None:
                continue
            share = q / req if req else 0.0
            pos["qty"] = max(0.0, pos["qty"] - got_q * share)
            L_[name]["usdt"] += got_usdt * share
            left_val = pos["qty"] * prices.get(coin, 0.0)
            if left_val < 1.0 or (full and got_q > 0):
                if pos["qty"] > 0:
                    state["dust"][coin] = state["dust"].get(coin, 0.0) + pos["qty"]
                del L_[name]["pos"][coin]
    by_coin = {}
    for name, coin, usdt in buys:
        by_coin.setdefault(coin, []).append((name, usdt))
    for coin, parts in by_coin.items():
        req = sum(u for _, u in parts)
        got_q, spent = buy_fills.get(coin, (0.0, 0.0))
        for name, u in parts:
            share = u / req if req else 0.0
            if got_q * share <= 0:
                continue
            pos = L_[name]["pos"].setdefault(coin, {"qty": 0.0, "age": 0})
            pos["qty"] += got_q * share
            L_[name]["usdt"] -= spent * share
    for name, ages in new_ages.items():
        for c, pos in L_[name]["pos"].items():
            if c in ages:
                pos["age"] = ages[c]


def record_nav(state, t, prices):
    navs = {k: ledger_nav(L, prices) for k, L in state["ledgers"].items()}
    total = sum(navs.values())
    scale = sum(L["scale"] * navs[k] for k, L in state["ledgers"].items()) / total if total else 1.0
    state["nav_history"] = [h for h in state["nav_history"] if h["date"] != str(t.date())]
    state["nav_history"].append({"date": str(t.date()), "nav": total, "scale": scale,
                                 **{f"nav_{k}": v for k, v in navs.items()}})
    state["nav_history"] = state["nav_history"][-400:]
    state["last_t"] = str(t.date())
    return total, navs


# ================================================================ 執行（共用流程）

def execute(state, sig, t, prices, broker, params=PARAMS, idle_rate=0.0, log=print):
    """一天的完整流程：規劃 → 先賣後買 → 分配成交 → 停泊閒置 USDT → 記錄淨值"""
    sells, buys, new_ages = plan_day(state, sig, t, prices, params, idle_rate, log)

    # 合併同一幣的賣單
    agg_sell = {}
    for name, coin, qty, why, _ in sells:
        agg_sell[coin] = agg_sell.get(coin, 0.0) + qty
        log(f"  賣 {coin:8s} {qty:.8g}（{name} 帳，{why}）")
    sell_fills = {}
    for coin, qty in agg_sell.items():
        sell_fills[coin] = broker.sell(coin, qty, prices.get(coin))

    # 買單：可用 USDT 不足時等比例縮小
    agg_buy = {}
    for name, coin, usdt in buys:
        agg_buy[coin] = agg_buy.get(coin, 0.0) + usdt
    need = sum(agg_buy.values())
    have = broker.ensure_usdt(need)
    ratio = min(1.0, have / need) if need > 0 else 1.0
    if ratio < 0.999:
        log(f"  可用 USDT {have:,.2f} < 需求 {need:,.2f}，買單縮小為 {ratio:.1%}")
    buys = [(n, c, u * ratio) for n, c, u in buys]
    buy_fills = {}
    for coin, usdt in agg_buy.items():
        log(f"  買 {coin:8s} {usdt * ratio:,.2f} USDT")
        buy_fills[coin] = broker.buy(coin, usdt * ratio, prices.get(coin))

    apply_fills(state, sells, buys, sell_fills, buy_fills, new_ages, prices, log)
    broker.park_usdt()
    total, navs = record_nav(state, t, prices)
    log(f"[{t.date()}] 總淨值 {total:,.2f}（" + "，".join(f"{k} 帳 {v:,.2f}" for k, v in navs.items()) + "）")
    return total
