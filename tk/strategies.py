"""Strategy families, each with a small coarse parameter grid.

A strategy turns one decision row (features at a session open) plus the 5-minute path of that
session into at most one trade. All of them share the same exit rules: stop, take-profit, or the
daily flat time. Every family has an economic story; none is a blind indicator search.
"""
import itertools

import numpy as np
import pandas as pd

from tk import config

MIN_STOP = 0.003   # stops tighter than 0.3% are mostly fees and noise
MAX_STOP = 0.06
COST = 2 * (config.TAKER_FEE + config.SLIPPAGE)


def grid():
    out = []
    # 1. Opening range breakout: a session's first move often sets its direction
    for s, orm, tp, sq in itertools.product(config.SESSIONS, [30, 60], [1.0, 1.5, 2.0, 3.0], [None, 0.8]):
        out.append(dict(family="orb", session=s, or_min=orm, tp_r=tp, squeeze_max=sq))
    # 2. Volatility squeeze: after quiet periods the next break of the 24h range tends to run
    for s, vp, sl, tp in itertools.product(config.SESSIONS, [0.2, 0.35], ["mid", "atr2"], [1.0, 1.5, 2.0, 3.0]):
        out.append(dict(family="squeeze", session=s, vol_pct_max=vp, sl=sl, tp_r=tp))
    # 3. Crowded positioning: extreme funding with rising open interest, fade or follow the crowd
    for s, z, oi, mode, k, tp in itertools.product(config.SESSIONS, [1.5, 2.0], [0.0, 0.05], ["fade", "follow"],
                                                   [1.0, 2.0], [1.0, 2.0]):
        out.append(dict(family="funding", session=s, z=z, oi_min=oi, mode=mode, atr_k=k, tp_r=tp))
    # 4. Trend day: a strong prior day confirmed by BTC continues into the next session
    for s, th, k, tp in itertools.product(config.SESSIONS, [0.03, 0.06], [1.0, 2.0], [1.0, 2.0, 3.0]):
        out.append(dict(family="trend", session=s, ret_min=th, atr_k=k, tp_r=tp))
    for i, p in enumerate(out):
        p["id"] = f"{p['family']}-{i:03d}"
    return out


def _run(path, entry_i, side, entry, sl, tp, funding_rate):
    """Walk the 5m path from entry_i. Returns (R, exit_i). Same-bar stop and target counts as stop."""
    hi, lo, cl, t = path["high"], path["low"], path["close"], path["time"]
    risk = abs(entry - sl) / entry
    if not (MIN_STOP <= risk <= MAX_STOP):
        return None
    if side > 0:
        hit_sl = lo[entry_i:] <= sl
        hit_tp = hi[entry_i + 1:] >= tp
    else:
        hit_sl = hi[entry_i:] >= sl
        hit_tp = lo[entry_i + 1:] <= tp
    i_sl = np.argmax(hit_sl) if hit_sl.any() else len(hit_sl)
    i_tp = (np.argmax(hit_tp) + 1) if hit_tp.any() else len(hit_sl)
    if i_sl <= i_tp and i_sl < len(hit_sl):
        exit_p, k = sl, entry_i + i_sl
    elif i_tp < len(hit_sl):
        exit_p, k = tp, entry_i + i_tp
    else:
        exit_p, k = cl[-1], len(cl) - 1
    gross = side * (exit_p - entry) / entry
    # funding is settled at 00/08/16 UTC; a position open across a settlement pays side * rate
    hours = pd.DatetimeIndex(t[entry_i:k + 1])
    crosses = ((hours.minute == 0) & (hours.hour % 8 == 0) & (np.arange(len(hours)) > 0)).sum()
    fund = side * funding_rate * crosses if funding_rate == funding_rate else 0.0
    return (gross - COST - fund) / risk, k


def trade(p, row, path):
    """Backtest one strategy on one session: plan the entry, then walk the path to the exit."""
    e = plan_entry(p, row, path)
    if e is None:
        return None
    res = _run(path, e["i"], e["side"], e["entry"], e["sl"], e["tp"], row.get("funding", np.nan))
    if res is None:
        return None
    return dict(side=e["side"], R=res[0], entry_time=path["time"][e["i"]])


def plan_entry(p, row, path):
    """Where and how a strategy enters in this session's path (possibly a partial live path).
    Returns dict(i, side, entry, sl, tp) or None. Shared by the backtest and the live bot."""
    if row["session"] != p["session"] or path is None or len(path["close"]) < 1:
        return None
    o0 = path["open"][0]
    fam = p["family"]

    if fam == "orb":
        if p["squeeze_max"] is not None and not (row["squeeze"] <= p["squeeze_max"]):
            return None
        n = p["or_min"] // 5
        if len(path["close"]) <= n:
            return None
        rh, rl = path["high"][:n].max(), path["low"][:n].min()
        return _breakout(path, n, rh, rl, None, p, "range")

    if fam == "squeeze":
        if not (row["vol_pct"] <= p["vol_pct_max"]):
            return None
        rh, rl = row["hi24"], row["lo24"]
        if not (rl < o0 < rh):
            return None
        return _breakout(path, 0, rh, rl, row["atr_pct"], p, p["sl"])

    if fam == "funding":
        z = row["funding_z"]
        if not (abs(z) >= p["z"]) or not (row["oi_chg24"] >= p["oi_min"]):
            return None
        crowd = 1 if z > 0 else -1          # positive funding: longs are crowded
        side = -crowd if p["mode"] == "fade" else crowd
        return _market(path, side, row["atr_pct"] * p["atr_k"], p)

    if fam == "trend":
        r, b = row["ret24"], row.get("btc_ret24", np.nan)
        if not (abs(r) >= p["ret_min"]) or not (np.sign(b) == np.sign(r)):
            return None
        return _market(path, int(np.sign(r)), row["atr_pct"] * p["atr_k"], p)
    return None


def _levels(i, side, entry, sl, p):
    risk = abs(entry - sl) / entry
    if not (MIN_STOP <= risk <= MAX_STOP):
        return None
    return dict(i=i, side=side, entry=entry, sl=sl, tp=entry + side * abs(entry - sl) * p["tp_r"])


def _market(path, side, stop_pct, p):
    if not stop_pct == stop_pct:
        return None
    entry = path["open"][0]
    return _levels(0, side, entry, entry * (1 - side * stop_pct), p)


def _breakout(path, start, rh, rl, atr_pct, p, sl_mode):
    hi, lo = path["high"][start:], path["low"][start:]
    up, dn = hi > rh, lo < rl
    if not (up.any() or dn.any()):
        return None
    iu = np.argmax(up) if up.any() else 10**9
    idn = np.argmax(dn) if dn.any() else 10**9
    if iu == idn:
        return None  # both sides broken in one bar: direction unknowable, skip
    side = 1 if iu < idn else -1
    i = start + min(iu, idn)
    entry = max(path["open"][i], rh) if side > 0 else min(path["open"][i], rl)
    if sl_mode == "range":
        sl = rl if side > 0 else rh
    elif sl_mode == "mid":
        sl = (rh + rl) / 2
    else:
        if not atr_pct == atr_pct:
            return None
        sl = entry * (1 - side * 2 * atr_pct)
    return _levels(i, side, entry, sl, p)
