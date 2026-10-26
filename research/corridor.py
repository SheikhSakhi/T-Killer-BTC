"""Corridor study for 10x without a stop: coins whose price stays inside a band no wider than 9%.

For every coin and every hour t:
  dur(t)   = how many hours back the price has stayed inside a band of width <= WIDTH
             ((max high - min low) / min low), i.e. the age of the current corridor
  forward  = what happened in the next 7 days after an entry at close(t), long and short:
             10x liquidation (adverse move >= LIQ), corridor survival, P&L of a simple rule

Questions:
  1. How long did corridors last in the past (per coin)?
  2. Does a long-lived corridor predict that it continues (survival curve by age)?
  3. Does entering inside an old corridor lower the 10x liquidation rate, out of sample?
  4. Which coins are in a corridor right now, and how do they rank?

python research/corridor.py
"""
import os
import sys
from collections import deque

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as sw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tk import data  # noqa: E402

WIDTH = 0.09      # corridor: max/min band of 9%
LIQ = 0.09        # 10x isolated liquidation ~ 9% adverse move
H = 168           # forward window: 7 days
FEE = 0.01        # 0.1% of notional per round trip = 1% of margin at 10x
FUND = 0.02       # ~2% of margin per week of holding at 10x, longs pay on average
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def corridor_age(hi, lo, width=WIDTH):
    """Two-pointer with monotonic deques: for each t, earliest s with (max hi[s..t]) / (min lo[s..t]) - 1 <= width."""
    n = len(hi)
    age = np.zeros(n, dtype=np.int32)
    top = np.zeros(n)
    bot = np.zeros(n)
    qmax, qmin = deque(), deque()
    s = 0
    for t in range(n):
        while qmax and hi[qmax[-1]] <= hi[t]:
            qmax.pop()
        qmax.append(t)
        while qmin and lo[qmin[-1]] >= lo[t]:
            qmin.pop()
        qmin.append(t)
        while qmax and qmin and hi[qmax[0]] / lo[qmin[0]] - 1 > width:
            s += 1
            if qmax[0] < s:
                qmax.popleft()
            if qmin[0] < s:
                qmin.popleft()
        if not (qmax and qmin):       # this bar alone is wider than the corridor
            age[t], top[t], bot[t] = 0, hi[t], lo[t]
            qmax.clear(), qmin.clear()
            s = t + 1
            continue
        age[t] = t - s + 1
        top[t], bot[t] = hi[qmax[0]], lo[qmin[0]]
    return age, top, bot


def episodes(age):
    """Completed corridor lengths: the age just before it drops."""
    drops = np.where(np.diff(age) < 0)[0]
    return age[drops]


def analyse(sym):
    h = data.load("1h", sym)
    hi, lo, cl = h["high"].values, h["low"].values, h["close"].values
    age, top, bot = corridor_age(hi, lo)
    n = len(cl) - H
    HI, LO = sw(hi[1:], H)[:n], sw(lo[1:], H)[:n]
    e = cl[:n, None]
    out = pd.DataFrame(index=h.index[:n])
    out["sym"] = sym
    out["age"] = age[:n]
    out["pos"] = (cl[:n] - bot[:n]) / np.maximum(top[:n] - bot[:n], 1e-12)   # 0 = corridor bottom, 1 = top
    out["width"] = top[:n] / bot[:n] - 1
    for side, name in ((1, "long"), (-1, "short")):
        adv = (LO / e - 1) if side > 0 else (1 - HI / e)
        liq = adv <= -LIQ
        out[f"liq_{name}"] = liq.any(1)
    # does the current corridor (same band) survive the next 1/3/7 days?
    for days in (1, 3, 7):
        k = 24 * days
        fh, fl = sw(hi[1:], k)[:n].max(1), sw(lo[1:], k)[:n].min(1)
        band_hi = np.maximum(top[:n], fh)
        band_lo = np.minimum(bot[:n], fl)
        out[f"survive_{days}d"] = band_hi / band_lo - 1 <= WIDTH
    return out, episodes(age), h.index[-1], dict(age=int(age[-1]), top=top[-1], bot=bot[-1], close=cl[-1])


def main():
    syms = sorted({p.split("-")[0] for p in os.listdir(data.CACHE) if p.endswith("-1h.parquet")})
    frames, eps, now = [], {}, {}
    for s in syms:
        try:
            f, ep, last, cur = analyse(s)
        except Exception as e:
            print(s, "skipped", e)
            continue
        frames.append(f)
        eps[s] = ep
        now[s] = dict(cur, last=last)
    df = pd.concat(frames)
    pd.set_option("display.width", 250)

    # 1. past corridor lengths (completed episodes of at least 24h)
    rows = []
    for s, ep in eps.items():
        ep = ep[ep >= 24] / 24
        if len(ep):
            rows.append(dict(sym=s, corridors=len(ep), median_days=np.median(ep), p75_days=np.percentile(ep, 75),
                             max_days=ep.max(), time_in_corridor_7d_plus=(df[df.sym == s]["age"] >= 168).mean()))
    hist = pd.DataFrame(rows).sort_values("median_days", ascending=False)
    print("== 1. Past corridors (band <= 9%, lasting >= 1 day)")
    print(hist.round(2).to_string(index=False))

    # 2 + 3. survival and liquidation by corridor age, split in two halves (in sample / out of sample)
    cut = df.index.min() + (df.index.max() - df.index.min()) / 2
    bins = [0, 24, 72, 168, 336, 720, 10**6]
    labels = ["<1d", "1-3d", "3-7d", "7-14d", "14-30d", "30d+"]
    df["age_bin"] = pd.cut(df["age"], bins, labels=labels, right=False)
    for name, part in (("first half", df[df.index < cut]), ("second half (out of sample)", df[df.index >= cut])):
        g = part.groupby("age_bin", observed=True).agg(
            n=("sym", "size"), survive_1d=("survive_1d", "mean"), survive_3d=("survive_3d", "mean"),
            survive_7d=("survive_7d", "mean"), liq_long=("liq_long", "mean"), liq_short=("liq_short", "mean"))
        print(f"\n== 2-3. By corridor age, {name}: corridor survival and 10x liquidation within 7 days")
        print(g.round(3).to_string())
    base = df.groupby(df.index >= cut)[["liq_long", "liq_short"]].mean()
    print("\nunconditional liquidation (first half / second half):")
    print(base.round(3).to_string())

    # entry position inside an old corridor: bottom third long, top third short
    old = df[df["age"] >= 168]
    for name, part in (("first half", old[old.index < cut]), ("second half", old[old.index >= cut])):
        lb = part[part["pos"] <= 1 / 3]["liq_long"].mean()
        st = part[part["pos"] >= 2 / 3]["liq_short"].mean()
        print(f"corridor >=7d, {name}: long from bottom third liq {lb:.3f} | short from top third liq {st:.3f}")

    # 4. right now
    rows = []
    for s, c in now.items():
        hd = hist.set_index("sym").reindex([s])
        rows.append(dict(sym=s, age_days=c["age"] / 24, width=c["top"] / c["bot"] - 1,
                         pos=(c["close"] - c["bot"]) / max(c["top"] - c["bot"], 1e-12),
                         bot=c["bot"], top=c["top"], close=c["close"],
                         past_median_days=float(hd["median_days"].iloc[0]) if len(hd) else np.nan,
                         past_p75_days=float(hd["p75_days"].iloc[0]) if len(hd) else np.nan))
    cur = pd.DataFrame(rows).sort_values("age_days", ascending=False)
    print("\n== 4. Now (", max(c['last'] for c in now.values()), "UTC )")
    print(cur.round(4).to_string(index=False))
    df.to_parquet(os.path.join(data.CACHE, "corridor_panel.parquet"))
    cur.to_csv(os.path.join(OUT, "corridor_now.csv"), index=False)
    hist.to_csv(os.path.join(OUT, "corridor_history.csv"), index=False)


if __name__ == "__main__":
    main()
