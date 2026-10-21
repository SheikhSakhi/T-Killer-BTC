"""Historical check: Kronos vs "price stays flat" on non-overlapping 24h windows.

Run: python forecast/backtest.py --days 200 --windows 180 --samples 5
"""
import argparse
import json

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=200)
ap.add_argument("--windows", type=int, default=180)
ap.add_argument("--samples", type=int, default=5)
ap.add_argument("--cost", type=float, default=0.001, help="round-trip cost per trade")
a = ap.parse_args()

df, _ = core.load_candles(days=a.days)
H = core.HORIZON
ends = [e for e in range(len(df) - H, core.LOOKBACK, -H)][:a.windows][::-1]

rows = []
for e in ends:
    last, actual = df["close"].iloc[e - 1], df["close"].iloc[e + H - 1]
    fin = core.forecast(df, e, a.samples)[:, -1]
    rows.append(dict(origin=str(df["timestamps"].iloc[e - 1]), last=last, actual=actual,
                     pred=float(fin.mean()), lo=float(np.percentile(fin, 10)),
                     hi=float(np.percentile(fin, 90))))
r = pd.DataFrame(rows)
sig = np.sign(r.pred - r["last"])
ret = sig * (r.actual / r["last"] - 1) - a.cost
hits = int((sig == np.sign(r.actual - r["last"])).sum())
print(json.dumps(dict(
    windows=len(r), first=r.origin.iloc[0], last=r.origin.iloc[-1],
    dir_acc=hits / len(r), up_days=float((r.actual > r["last"]).mean()),
    mae_kronos=float((r.pred - r.actual).abs().mean()), mae_naive=float((r["last"] - r.actual).abs().mean()),
    band80_coverage=float(((r.actual >= r.lo) & (r.actual <= r.hi)).mean()),
    strategy_total=float((1 + ret).prod() - 1), buy_hold_total=float((r.actual / r["last"]).prod() - 1),
), indent=1))
