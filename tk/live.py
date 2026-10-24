"""Live trading loop, run once a minute on the server.

MODE=paper  -> no orders, fills simulated at market price (for dry runs)
MODE=exchange with BINANCE_TESTNET=1 -> Binance futures testnet
MODE=exchange with BINANCE_TESTNET=0 -> real money

One trade per UTC day, at most one open position, flat by config.FLAT_AT. Entries come only from
today's plan (data/plan/latest.json) written by the nightly research run, using the same
plan_entry() code as the backtest.
"""
import json
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tk import config, features, notify, strategies
from tk.exchange import Futures

ROOT = os.path.join(os.path.dirname(__file__), "..")
LIVE_DIR = os.path.join(ROOT, "data", "live")
STATE = os.path.join(LIVE_DIR, "state.json")
TRADES = os.path.join(LIVE_DIR, "trades.csv")
EVENTS = os.path.join(LIVE_DIR, "events.log")
PLAN = os.path.join(ROOT, "data", "plan", "latest.json")
MODE = os.environ.get("MODE", "paper")
MAX_RISK = 1.5 * config.RISK_PER_TRADE   # allowed only to reach the exchange minimum order size


def now():
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def log(msg):
    os.makedirs(LIVE_DIR, exist_ok=True)
    line = f"{now():%Y-%m-%d %H:%M:%S} [{MODE}] {msg}"
    print(line)
    with open(EVENTS, "a") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return dict(day=None, traded_today=False, open=None, day_start_equity=None, start_equity=None,
                halted=None, paper_equity=config.CAPITAL, rows={})


def save_state(s):
    os.makedirs(LIVE_DIR, exist_ok=True)
    with open(STATE, "w") as f:
        json.dump(s, f, indent=1, default=str)


def record_trade(t):
    new = not os.path.exists(TRADES)
    pd.DataFrame([t]).to_csv(TRADES, mode="a", header=new, index=False)


def equity(x, s):
    return s["paper_equity"] if MODE == "paper" else x.balance()[0]


def session_start(day, session):
    return pd.Timestamp(f"{day} {config.SESSIONS[session]}")


def decision_row(sym, session, day, s):
    """Features at the session open, computed once and cached in the state."""
    key = f"{day}|{sym}|{session}"
    if key not in s["rows"]:
        btc = features.data.load("1h", "BTCUSDT")
        rows = features.build(sym, btc)
        r = rows[(rows["day"] == pd.Timestamp(day)) & (rows["session"] == session)]
        if r.empty:
            return None
        s["rows"][key] = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                          for k, v in r.iloc[0].to_dict().items()}
    row = dict(s["rows"][key])
    return {k: (np.nan if v is None else v) for k, v in row.items()}


def session_path(x, sym, start):
    k = x.klines(sym, "5m", limit=300, start=start)
    k = k[k.index >= start]
    if k.empty:
        return None
    return dict(open=k["open"].values, high=k["high"].values, low=k["low"].values,
                close=k["close"].values, time=k.index.values)


# ------------------------------------------------------------------ position management

def check_closed(x, s):
    """Detect that the exchange stop/target closed the position; record the result."""
    o = s["open"]
    if o is None:
        return
    if MODE == "paper":
        k = x.klines(o["sym"], "5m", limit=300, start=o["opened_at"])
        k = k[k.index >= pd.Timestamp(o["opened_at"]).ceil("5min")]  # bars fully after the fill
        hit = None
        for _, b in k.iterrows():
            if (b["low"] <= o["sl"]) if o["side"] > 0 else (b["high"] >= o["sl"]):
                hit = ("stop", o["sl"])
                break
            if (b["high"] >= o["tp"]) if o["side"] > 0 else (b["low"] <= o["tp"]):
                hit = ("target", o["tp"])
                break
        if hit is None:
            return
        exit_reason, exit_px = hit
        pnl = o["side"] * (exit_px - o["fill"]) * o["qty"] - strategies.COST * o["fill"] * o["qty"]
        s["paper_equity"] += pnl
    else:
        if o["sym"] in x.positions():
            return
        x.cancel_all(o["sym"])
        inc = x.income(int(pd.Timestamp(o["opened_at"]).timestamp() * 1000))
        pnl = sum(float(i["income"]) for i in inc if i["symbol"] == o["sym"]
                  and i["incomeType"] in ("REALIZED_PNL", "COMMISSION", "FUNDING_FEE"))
        exit_reason, exit_px = "exchange", x.price(o["sym"])
    finish(s, o, pnl, exit_reason, exit_px)


def finish(s, o, pnl, reason, exit_px):
    R = pnl / o["risk_usd"] if o["risk_usd"] else float("nan")
    t = dict(o, closed_at=str(now()), exit_reason=reason, exit_px=exit_px, pnl=round(pnl, 4), R=round(R, 3))
    record_trade(t)
    s["open"] = None
    log(f"CLOSED {o['sym']} {reason} pnl={pnl:+.2f} USDT R={R:+.2f}")
    notify.send(f"{'✅' if pnl > 0 else '❌'} {o['sym']} closed: {pnl:+.2f} USDT ({R:+.2f}R)",
                f"Strategy {o['variant']} ({o['session']})\nEntry {o['fill']} → exit ~{exit_px} ({reason})")


def flatten(x, s, why):
    o = s["open"]
    if o is None:
        return
    if MODE == "paper":
        px = x.price(o["sym"])
        pnl = o["side"] * (px - o["fill"]) * o["qty"] - strategies.COST * o["fill"] * o["qty"]
        s["paper_equity"] += pnl
        finish(s, o, pnl, why, px)
        return
    pos = x.positions().get(o["sym"])
    if pos:
        x.market(o["sym"], -int(np.sign(pos["qty"])), abs(pos["qty"]), reduce_only=True)
    x.cancel_all(o["sym"])
    check_closed(x, s)


# ------------------------------------------------------------------ entries

def try_enter(x, s, plan, t):
    day = str(t.date())
    eq = equity(x, s)
    for c in plan["candidates"]:
        p, sym = c["params"], c["sym"]
        start = session_start(day, p["session"])
        if t < start:
            continue
        row = decision_row(sym, p["session"], day, s)
        if row is None:
            continue
        path = session_path(x, sym, start)
        if path is None:
            continue
        e = strategies.plan_entry(p, row, path)
        if e is None:
            continue
        fresh = e["i"] == len(path["close"]) - 1
        if p["family"] in ("funding", "trend"):
            fresh = e["i"] == 0 and (t - start) < pd.Timedelta(minutes=10)
        if not fresh:
            continue
        return enter(x, s, c, e, eq, t)
    return False


def enter(x, s, c, e, eq, t):
    sym, side, sl, tp = c["sym"], e["side"], e["sl"], e["tp"]
    px = x.price(sym)
    if (side > 0 and px <= sl) or (side < 0 and px >= sl):
        log(f"skip {sym}: price {px} already beyond stop {sl}")
        return False
    stop_dist = abs(px - sl)
    risk_usd = config.RISK_PER_TRADE * eq
    f = x.filters(sym)
    qty = min(risk_usd / stop_dist, config.MAX_LEVERAGE * eq / px)
    qty = x.round_qty(sym, qty)
    if qty * px < f["min_notional"]:
        qty = x.round_qty(sym, f["min_notional"] / px + f["step"])
        if qty * stop_dist > MAX_RISK * eq or qty * px > config.MAX_LEVERAGE * eq:
            log(f"skip {sym}: minimum order {f['min_notional']} USDT would risk {qty * stop_dist:.2f}")
            return False
    risk_usd = qty * stop_dist
    o = dict(sym=sym, side=side, qty=qty, signal_entry=e["entry"], sl=sl, tp=tp, variant=c["variant"],
             session=c["params"]["session"], opened_at=str(t), risk_usd=round(risk_usd, 4), mode=MODE)
    if MODE == "paper":
        o["fill"] = px
    else:
        x.setup_symbol(sym, config.MAX_LEVERAGE)
        r = x.market(sym, side, qty)
        o["fill"] = float(r.get("avgPrice") or px)
        try:
            x.protect(sym, side, sl, tp)
        except Exception as err:
            x.market(sym, -side, qty, reduce_only=True)
            log(f"protective orders failed, position closed: {err}")
            notify.send("⚠️ T-Killer: stop/target could not be placed", f"{sym}: position closed at once.\n{err}")
            s["traded_today"] = True
            return False
    s["open"], s["traded_today"] = o, True
    log(f"OPEN {sym} {'LONG' if side > 0 else 'SHORT'} qty={qty} fill={o['fill']} sl={sl:.6g} tp={tp:.6g} "
        f"risk={risk_usd:.2f} via {c['variant']}")
    notify.send(
        f"🔔 {sym} {'LONG' if side > 0 else 'SHORT'} ({MODE})",
        f"Strategy: {c['variant']} — {c['params']['family']}, {c['params']['session']} session\n"
        f"Entry {o['fill']} · stop {sl:.6g} · target {tp:.6g}\n"
        f"Size {qty} ({qty * o['fill']:.2f} USDT notional), risk {risk_usd:.2f} USDT\n"
        f"History: {c['n_train']} trades, avg {c['mean_train']:+.2f}R in training, "
        f"{c['mean_val']:+.2f}R in the last 30 days")
    return True


# ------------------------------------------------------------------ main

def run():
    # paper mode follows the real market; exchange mode trades where its keys point
    x = Futures(testnet=False) if MODE == "paper" else Futures()
    s = load_state()
    t = now()
    day = str(t.date())
    if s["day"] != day:
        eq = equity(x, s)
        s.update(day=day, traded_today=False, day_start_equity=eq, rows={},
                 start_equity=s.get("start_equity") or eq)
        if s.get("halted") == "daily":
            s["halted"] = None
        log(f"new day, equity {eq:.2f}")

    check_closed(x, s)

    flat_at = pd.Timestamp(f"{day} {config.FLAT_AT}")
    if t >= flat_at and s["open"]:
        flatten(x, s, "time")

    eq = equity(x, s)
    if s["halted"] is None and s["day_start_equity"] and eq <= s["day_start_equity"] * (1 - config.DAILY_LOSS_STOP):
        s["halted"] = "daily"
        notify.send("⛔ T-Killer: daily loss limit", f"Equity {eq:.2f}; no new trades until tomorrow.")
    if s["start_equity"] and eq <= s["start_equity"] * (1 - config.TOTAL_DRAWDOWN_STOP) and s["halted"] != "total":
        s["halted"] = "total"
        flatten(x, s, "drawdown stop")
        notify.send("🛑 T-Killer: drawdown stop", f"Equity {eq:.2f}. Trading halted until reviewed.")

    if (s["halted"] is None and s["open"] is None and not s["traded_today"] and t < flat_at
            and os.path.exists(PLAN)):
        with open(PLAN) as f:
            plan = json.load(f)
        if plan.get("day") == day and plan.get("candidates"):
            try_enter(x, s, plan, t)
    save_state(s)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        log("ERROR " + repr(e))
        traceback.print_exc()
        # alert once per distinct error per hour, not every minute
        mark = os.path.join(LIVE_DIR, ".last_error")
        key = f"{now():%Y-%m-%d %H} {type(e).__name__}: {str(e)[:120]}"
        if not os.path.exists(mark) or open(mark).read() != key:
            open(mark, "w").write(key)
            notify.send("⚠️ T-Killer error", traceback.format_exc()[-1500:], mail=False)
        sys.exit(1)
