"""Historical check: Kronos vs "price stays flat" on non-overlapping 24h windows,
plus a confidence filter that only trades when most sampled paths agree.

Run: python forecast/backtest.py --days 205 --windows 180 --samples 30 --out results/bt
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=205)
ap.add_argument("--windows", type=int, default=180)
ap.add_argument("--samples", type=int, default=30)
ap.add_argument("--group", type=int, default=4, help="windows per model batch")
ap.add_argument("--cost", type=float, default=0.001, help="round-trip cost per trade")
ap.add_argument("--out", default="results/backtest")
a = ap.parse_args()

df, _ = core.load_candles(days=a.days)
H = core.HORIZON
# Windows start at the 23:00 UTC candle so a resumed run picks the same windows.
ends = [e for e in range(core.LOOKBACK, len(df) - H + 1)
        if df["timestamps"].iloc[e - 1].hour == 23][-a.windows:]

# Resume: windows already in the CSV from an earlier (possibly timed-out) run are reused.
csv_path = a.out + "-windows.csv"
done = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
rows = done.to_dict("records")
seen = set(done["origin"]) if len(done) else set()
todo = [e for e in ends if str(df["timestamps"].iloc[e - 1]) not in seen]
print(f"{len(rows)} windows reused, {len(todo)} to run", file=sys.stderr, flush=True)

for i in range(0, len(todo), a.group):
    chunk = todo[i:i + a.group]
    fins = core.forecast_many(df, chunk, a.samples)[:, :, -1]
    for e, fin in zip(chunk, fins):
        last = df["close"].iloc[e - 1]
        rows.append(dict(origin=str(df["timestamps"].iloc[e - 1]), last=last,
                         actual=df["close"].iloc[e + H - 1], p50=float(np.median(fin)),
                         p10=float(np.percentile(fin, 10)), p90=float(np.percentile(fin, 90)),
                         up_share=float((fin > last).mean())))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"{len(rows)}/{len(ends)}", file=sys.stderr, flush=True)

r = pd.DataFrame(rows).sort_values("origin").reset_index(drop=True)
r["move"] = r.actual / r["last"] - 1


def trade_stats(mask, side):
    t = r[mask]
    if t.empty:
        return dict(trades=0)
    ret = side[mask] * t.move - a.cost
    return dict(trades=int(len(t)), hit=float((ret + a.cost > 0).mean()),
                avg_ret=float(ret.mean()), total=float((1 + ret).prod() - 1))


side = np.sign(r.p50 - r["last"])
summary = dict(
    windows=len(r), samples=a.samples, first=r.origin.iloc[0], last=r.origin.iloc[-1],
    dir_acc=float((side == np.sign(r.move)).mean()), up_days=float((r.move > 0).mean()),
    mae_kronos=float((r.p50 - r.actual).abs().mean()), mae_naive=float((r["last"] - r.actual).abs().mean()),
    band80_coverage=float(((r.actual >= r.p10) & (r.actual <= r.p90)).mean()),
    buy_hold_total=float((1 + r.move).prod() - 1),
    all_days=trade_stats(side != 0, side),
    filters={},
)
agree = np.where(r.up_share >= 0.5, 1, -1)
for th in (0.6, 0.7, 0.8, 0.9):
    conf = np.maximum(r.up_share, 1 - r.up_share) >= th
    summary["filters"][f">={th:.0%}"] = trade_stats(conf, pd.Series(agree, index=r.index))
with open(a.out + ".json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
