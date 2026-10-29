"""One-day signals for gold: is there any rule that works on a 24-hour horizon?

Decision at 00:00 UTC (Asia open) or 07:00 UTC (London open); exit at stop, target, or 24 hours.
Direction rules (all from data known at the decision time):
  vote3      majority of trend20, trend50, dollar20 (the weekly rule)
  trend20 / trend50
  mom1d      direction of the previous 24 hours
  rev1d      opposite of the previous 24 hours
  asia       07:00 entries only: direction of the 00:00-07:00 move
  rsi_rev    against an hourly RSI(14) extreme (>65 short, <35 long), else no trade
Costs 0.08% of the position per round trip (limit entry, market exit). Both halves reported.

python research/xau_1day.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tk import data, xau  # noqa: E402

COST = 0.0008
EXITS = [(sl, k) for sl in (0.005, 0.0075, 0.01) for k in (1.0, 1.5, 2.0)]


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn)


def run(sym):
    h = data.load("1h", sym)
    usd = xau.dollar()
    r14 = rsi(h["close"])
    hi, lo, cl, idx = h["high"].values, h["low"].values, h["close"].values, h.index
    rows = []
    for day in pd.date_range(idx[0].normalize() + pd.Timedelta(days=51), idx[-1].normalize() - pd.Timedelta(days=1)):
        for hour in (0, 7):
            t = day + pd.Timedelta(hours=hour)
            i = idx.searchsorted(t)
            if i >= len(idx) - 24 or idx[i] != t:
                continue
            e = cl[i - 1]
            side3, _, _, v = xau.side_at(h, t, usd)
            sig = {"vote3": side3, "trend20": v["trend20"], "trend50": v.get("trend50", 0),
                   "mom1d": int(np.sign(e - cl[i - 25])), "rev1d": -int(np.sign(e - cl[i - 25])),
                   "rsi_rev": -1 if r14.iloc[i - 1] > 65 else (1 if r14.iloc[i - 1] < 35 else 0)}
            if hour == 7:
                sig["asia"] = int(np.sign(e - cl[i - 8]))
            fh, fl, fc = hi[i:i + 24], lo[i:i + 24], cl[i + 23]
            for name, side in sig.items():
                if side == 0:
                    continue
                adv = (fl / e - 1) if side > 0 else (1 - fh / e)
                fav = (fh / e - 1) if side > 0 else (1 - fl / e)
                for sl, k in EXITS:
                    tp = sl * k
                    ts = np.argmax(adv <= -sl) if (adv <= -sl).any() else 24
                    tt = np.argmax(fav >= tp) if (fav >= tp).any() else 24
                    if ts < 24 and ts <= tt:
                        r = -sl
                    elif tt < 24:
                        r = tp
                    else:
                        r = side * (fc / e - 1)
                    rows.append((sym, t, hour, name, sl, k, (r - COST) / sl, r - COST))
    return pd.DataFrame(rows, columns=["sym", "t", "hour", "rule", "sl", "k", "R", "ret"])


def main():
    pd.set_option("display.width", 220)
    r = pd.concat([run("XAUUSDT"), run("PAXGUSDT")])
    r["half"] = np.where(r["t"] < r.groupby("sym")["t"].transform(lambda s: s.min() + (s.max() - s.min()) / 2), "1st", "2nd")
    g = r.groupby(["rule", "hour", "sl", "k", "half"])["R"].mean().unstack("half")
    n = r.groupby(["rule", "hour", "sl", "k"])["R"].size()
    w = r.groupby(["rule", "hour", "sl", "k"])["R"].apply(lambda x: (x > 0).mean())
    g = g.assign(trades=n, win=w)
    both = g[(g["1st"] > 0) & (g["2nd"] > 0)].sort_values("2nd", ascending=False)
    print(f"combinations tested: {len(g)}; positive in both halves: {len(both)}")
    print(both.round(3).head(20).to_string())
    print("\nAverage R by rule and entry hour (all exits pooled):")
    print(r.groupby(["rule", "hour", "half"])["R"].mean().unstack("half").round(3).to_string())
    r.to_parquet(os.path.join(data.CACHE, "xau_1day.parquet"))


if __name__ == "__main__":
    main()
