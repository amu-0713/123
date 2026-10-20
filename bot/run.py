#!/usr/bin/env python3
"""
每日執行一次（建議 UTC 00:05，台灣時間 08:05）。

模式：
  paper  紙上交易：不需要 API key，以收盤價模擬成交，狀態存在 --state 檔
  dry    演練：讀取真實帳戶與行情，只印出今天要下的單，不下單、不存狀態（需要 API key）
  live   實盤：真的下單，並把閒置 USDT 存入活期

用法：
  python bot/run.py --mode paper --capital 10000              # 第一次會建立 state_paper.json
  BINANCE_API_KEY=... BINANCE_API_SECRET=... python bot/run.py --mode dry --capital 1000
  BINANCE_API_KEY=... BINANCE_API_SECRET=... python bot/run.py --mode live --capital 1000

安全機制：
  - 同一根日 K 只會處理一次（state.last_t），重複執行不會重複下單；--force 可強制重跑
  - bot 資料夾下有 STOP 檔案時直接結束（緊急停止）
  - 實盤每次執行前先對帳：以實際餘額修正兩本帳（活期利息、手續費誤差）
  - --backup-dir（或環境變數 BOT_BACKUP_DIR）：每次存檔後把狀態檔與紀錄複製一份過去，
    建議指向 OneDrive / Google 雲端硬碟資料夾，電腦壞掉時可在另一台機器接著跑
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

import engine  # noqa: E402
import factor_research as fr  # noqa: E402
from binance_api import Client, symbol_rules  # noqa: E402
from brokers import LiveBroker, PaperBroker  # noqa: E402
from download_binance import filter_symbols  # noqa: E402


# ================================================================ 資料

def fetch_panel(client, now, extra_coins=(), log=print, workers=8):
    """抓所有可交易 USDT 現貨的日 K，回傳 (面板, 已收盤日 t, 交易規則)"""
    info = client.exchange_info()
    rules = symbol_rules(info)
    syms = sorted(s for s, r in rules.items() if r["status"] == "TRADING")
    syms, _ = filter_symbols(syms, "USDT")                       # 排除穩定幣、槓桿代幣
    syms = sorted(set(syms) | {c + "USDT" for c in extra_coins if c + "USDT" in rules})
    today = pd.Timestamp(now).tz_convert("UTC").normalize()     # 今天 00:00 UTC，之前的 K 線才算收盤

    def one(sym):
        try:
            rows = client.klines(sym, "1d", engine.HISTORY_DAYS)
        except Exception as e:                                    # 單一幣失敗不影響整體
            log(f"  抓取 {sym} 失敗：{e}")
            return sym, None
        d = pd.DataFrame([(int(r[0]), float(r[2]), float(r[3]), float(r[4]), float(r[7]), float(r[10]))
                          for r in rows],
                         columns=["open_time", "high", "low", "close", "quote_volume", "taker_buy_quote"])
        if d.empty:
            return sym, None
        d.index = pd.to_datetime(d.pop("open_time"), unit="ms", utc=True)
        return sym, d[d.index < today]

    with ThreadPoolExecutor(workers) as ex:
        frames = {s[:-4]: d for s, d in ex.map(one, syms) if d is not None and len(d)}
    P = fr.build_panel(frames, verbose=False, window_start=today - pd.Timedelta(days=engine.HISTORY_DAYS))
    t = P["close"].index.max()
    if t != today - pd.Timedelta(days=1):
        log(f"  注意：最新已收盤日 K 為 {t.date()}，不是昨天 {(today - pd.Timedelta(days=1)).date()}")
    return P, t, rules


# ================================================================ 對帳與閒置收益

def accrue_paper_interest(state, wallet, t, rate):
    """紙上交易：未投入的 USDT 依天數計息（模擬活期）"""
    if not rate or not state.get("last_t"):
        return
    days = (t - pd.Timestamp(state["last_t"], tz="UTC")).days
    if days <= 0:
        return
    g = (1 + rate / 365) ** days
    for L in state["ledgers"].values():
        L["usdt"] *= g
    wallet["USDT"] = wallet.get("USDT", 0.0) * g


def real_usdt(client):
    return client.free("USDT") + client.earn_balance("USDT")


def reconcile_live(state, client, log=print):
    """
    以實際餘額修正帳本：
      - USDT：只把「上次執行結束後」到現在的變化（活期利息）依比例分給各帳。
        帳戶裡策略以外的 USDT 不會被算進來。
      - 幣：實際數量少於帳上（例如手續費以該幣扣除）時等比例下修。
    """
    Ls = state["ledgers"]
    now = real_usdt(client)
    last = state.get("usdt_real_last")
    if last is not None:
        diff = now - last
        book = sum(L["usdt"] for L in Ls.values())
        if diff < -1.0:
            log(f"  !! 帳戶 USDT 比上次執行後少了 {-diff:,.2f}，請確認是否有手動提領或交易")
        if book > 0 and abs(diff) > 0.01:
            for L in Ls.values():
                L["usdt"] += diff * (L["usdt"] / book)
    elif now < sum(L["usdt"] for L in Ls.values()) - 1:
        raise SystemExit(f"帳戶 USDT（現貨+活期）{now:,.2f} 少於投入資金，請先存入或調低 --capital")
    bal = client.balances()
    coins = {c for L in Ls.values() for c in L["pos"]}
    for c in coins:
        book = sum(L["pos"][c]["qty"] for L in Ls.values() if c in L["pos"])
        real = bal.get(c, 0.0)
        if book > 0 and real < book * 0.999:
            log(f"  {c} 實際 {real:.8g} < 帳上 {book:.8g}，等比例下修")
            for L in Ls.values():
                if c in L["pos"]:
                    L["pos"][c]["qty"] *= real / book


# ================================================================ 主流程

def run_once(client, state, mode, now, capital=None, idle_rate=0.02, fee=0.001, force=False, log=print):
    """執行一天。回傳更新後的 state（dry 模式不應存檔）。"""
    held = {c for L in state["ledgers"].values() for c in L["pos"]} if state else set()
    P, t, rules = fetch_panel(client, now, held, log)
    if state and state.get("last_t") == str(t.date()) and not force:
        log(f"{t.date()} 已處理過，略過（--force 可強制重跑）")
        return state
    sig = engine.Signals(P)

    if mode == "paper":
        prices = P["close"].loc[t].dropna().to_dict()
        prices = {engine.coin_of(k): v for k, v in prices.items()}
        wallet = state.setdefault("paper_wallet", {"USDT": capital})
        accrue_paper_interest(state, wallet, t, idle_rate)
        broker = PaperBroker(wallet, fee, log)
    else:
        allp = client.prices()
        prices = {s[:-4]: p for s, p in allp.items() if s.endswith("USDT")}
        if mode == "live":
            reconcile_live(state, client, log)
            broker = LiveBroker(client, rules, log)
        else:                                                    # dry：用真實餘額模擬，不下單
            bal = client.balances()
            wallet = {"USDT": sum(L["usdt"] for L in state["ledgers"].values())}
            for L in state["ledgers"].values():
                for c, p in L["pos"].items():
                    wallet[c] = wallet.get(c, 0.0) + p["qty"]
            log(f"  實際帳戶 USDT（現貨）{bal.get('USDT', 0):,.2f}；以帳本模擬今天的訂單")
            broker = PaperBroker(wallet, fee, log)

    log(f"===== {t.date()}（{'一二三四五六日'[t.dayofweek]}）模式：{mode} =====")
    engine.execute(state, sig, t, prices, broker, idle_rate=idle_rate if mode == "paper" else 0.0, log=log)
    if mode == "live":
        state["usdt_real_last"] = real_usdt(client)              # 下次對帳的基準
    for name, L in state["ledgers"].items():
        pos = ", ".join(f"{c} {p['qty'] * prices.get(c, 0):,.0f}" for c, p in L["pos"].items()) or "無"
        log(f"  {name} 帳：持倉 {pos}；USDT {L['usdt']:,.2f}")
    return state


def backup_state(state_path, log_path, backup_dir, today, keep=90, log=print):
    """複製狀態檔（最新一份 + 依日期的歷史）與當天紀錄到備份資料夾；失敗只警告，不影響交易。"""
    try:
        hist = os.path.join(backup_dir, "history")
        os.makedirs(hist, exist_ok=True)
        name = os.path.basename(state_path)
        shutil.copy2(state_path, os.path.join(backup_dir, name))
        shutil.copy2(state_path, os.path.join(hist, f"{name[:-5]}_{today}.json"))
        if os.path.exists(log_path):
            shutil.copy2(log_path, os.path.join(backup_dir, os.path.basename(log_path)))
        old = sorted(f for f in os.listdir(hist) if f.startswith(name[:-5] + "_"))
        for f in old[:-keep]:
            os.remove(os.path.join(hist, f))
        log(f"已備份到 {backup_dir}")
    except OSError as e:
        log(f"⚠️ 備份失敗（交易不受影響）：{e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["paper", "dry", "live"], default="paper")
    ap.add_argument("--state", default=None, help="狀態檔（預設 bot/state_<mode>.json）")
    ap.add_argument("--capital", type=float, default=None, help="第一次執行時投入的 USDT")
    ap.add_argument("--idle-rate", type=float, default=0.02, help="紙上交易的活期年化利率假設")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--backup-dir", default=os.environ.get("BOT_BACKUP_DIR"),
                    help="每次存檔後複製狀態檔與紀錄到這個資料夾（例如雲端硬碟）")
    a, _ = ap.parse_known_args(argv)

    if os.path.exists(os.path.join(HERE, "STOP")):
        print("偵測到 bot/STOP，緊急停止，不執行。")
        return
    state_path = a.state or os.path.join(HERE, f"state_{'live' if a.mode != 'paper' else 'paper'}.json")
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    log_path = os.path.join(HERE, "logs", f"{dt.datetime.utcnow():%Y-%m-%d}_{a.mode}.log")
    logf = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    else:
        if not a.capital:
            sys.exit(f"找不到狀態檔 {state_path}；第一次執行請加 --capital <USDT>")
        state = engine.new_state(a.capital)
        log(f"建立新狀態：{a.capital:,.2f} USDT，A 帳（週日）/ B 帳（週四）各半")

    key, secret = os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET")
    if a.mode != "paper" and not (key and secret):
        sys.exit("dry / live 模式需要環境變數 BINANCE_API_KEY、BINANCE_API_SECRET")
    client = Client(key, secret)
    now = pd.Timestamp.now(tz="UTC")
    state = run_once(client, state, a.mode, now, a.capital, a.idle_rate, force=a.force, log=log)

    if a.mode == "dry":
        log("（dry 模式：未下單、未存檔）")
    else:
        tmp = state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        os.replace(tmp, state_path)
        log(f"狀態已存：{state_path}")
        if a.backup_dir:
            backup_state(state_path, log_path, a.backup_dir, f"{dt.datetime.utcnow():%Y-%m-%d}", log=log)
    logf.close()


if __name__ == "__main__":
    main()
