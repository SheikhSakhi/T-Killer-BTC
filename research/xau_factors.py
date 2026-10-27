"""Which signals tell the direction of gold? Same test for every rule.

Gold: PAXGUSDT hourly (Mar 2025 -), XAUUSDT as a cross-check. Macro: FRED real 10y yield (DFII10),
broad dollar index (DTWEXBGS); CFTC disaggregated COT, managed money in COMEX gold (088691).
Every signal is computed from data published before the 00:00 UTC decision:
FRED values lag 2 days (published next business day), COT is usable from the Friday after its Tuesday date.

For each rule: (1) next-5-day direction hit rate and average move, every 5th day (non-overlapping);
(2) the trade the analyst uses: stop 2%, target 4%, max 7 days, non-overlapping; both split in halves.

python research/xau_factors.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from tk import data, xau  # noqa: E402

M = os.path.join(ROOT, "cache", "macro")


def fred(name):
    s = pd.read_csv(os.path.join(M, f"fred_{name}.csv"), na_values=".")
    s.columns = ["date", "v"]
    s["date"] = pd.to_datetime(s["date"])
    return s.dropna().set_index("date")["v"]


def asof(series, when, lag_days):
    s = series[series.index <= when - pd.Timedelta(days=lag_days)]
    return s


def signals(h):
    daily = h["close"].resample("1D").last().dropna()
    ry, usd = fred("DFII10"), fred("DTWEXBGS")
    cot = pd.read_csv(os.path.join(M, "cot_gold.csv"), parse_dates=["date"])
    cot["usable"] = cot["date"] + pd.Timedelta(days=3)
    rows = []
    for d in daily.index[100:]:
        past = daily[daily.index < d]
        px = past.iloc[-1]
        sig = {}
        for n in (10, 20, 50, 100):
            sig[f"trend{n}"] = 1 if px > past.iloc[-n:].mean() else -1
        sig["trend_vote"] = int(np.sign(sum(sig[f"trend{n}"] for n in (10, 20, 50, 100)))) or sig["trend20"]
        r = asof(ry, d, 2)
        if len(r) > 25:
            sig["real_yield_20d"] = -1 if r.iloc[-1] > r.iloc[-21] else 1
            sig["real_yield_5d"] = -1 if r.iloc[-1] > r.iloc[-6] else 1
        u = asof(usd, d, 2)
        if len(u) > 25:
            sig["dollar_20d"] = -1 if u.iloc[-1] > u.iloc[-21] else 1
        c = cot[cot["usable"] <= d]
        if len(c) >= 26:
            pct = (c["mm_net"].iloc[-26:] <= c["mm_net"].iloc[-1]).mean()
            sig["cot_contrarian"] = -1 if pct >= 0.8 else (1 if pct <= 0.2 else 0)
        rows.append(dict(day=d, **sig))
    return pd.DataFrame(rows).set_index("day")


def evaluate(h, S):
    daily = h["close"].resample("1D").last().dropna()
    fwd5 = daily.shift(-5) / daily - 1
    out = []
    half = S.index[0] + (S.index[-1] - S.index[0]) / 2
    for name in S.columns:
        s = S[name].dropna()
        s = s[s != 0]
        # (1) 5-day direction, every 5th signal day
        sub = s.iloc[::5]
        m = (sub * fwd5.reindex(sub.index - pd.Timedelta(days=1)).values).dropna()
        # (2) analyst trade, non-overlapping
        res, t = [], s.index[0]
        while t < h.index[-1] - pd.Timedelta(hours=xau.HOLD_H):
            if t not in s.index:
                t += pd.Timedelta(days=1)
                continue
            px = h["close"][h.index < t].iloc[-1]
            o, r, end = xau.simulate(h, t, int(s[t]), px)
            res.append((t, (r - xau.FEE) / xau.SL))
            t = (end + pd.Timedelta(hours=1)).ceil("D") if end is not None else t + pd.Timedelta(days=1)
        R = pd.Series([x[1] for x in res], index=[x[0] for x in res])
        out.append(dict(rule=name, days=len(s), long_share=float((s > 0).mean()),
                        hit5_1=float((m[m.index < half] > 0).mean()), hit5_2=float((m[m.index >= half] > 0).mean()),
                        trades=len(R), R_1=float(R[R.index < half].mean()), R_2=float(R[R.index >= half].mean()),
                        t=float(R.mean() / (R.std() / np.sqrt(len(R)))) if len(R) > 2 else np.nan))
    return pd.DataFrame(out).set_index("rule")


def weekend(h):
    """Fade the weekend drift: Friday 21:00 -> Sunday 23:00 UTC move, traded Sunday 23:00 -> Monday 21:00."""
    c = h["close"]
    res = []
    for fri in c.index[(c.index.dayofweek == 4) & (c.index.hour == 21)]:
        sun, mon = fri + pd.Timedelta(hours=50), fri + pd.Timedelta(hours=72)
        if sun in c.index and mon in c.index:
            drift = c[sun] / c[fri] - 1
            res.append((fri, drift, c[mon] / c[sun] - 1))
    w = pd.DataFrame(res, columns=["fri", "drift", "next"]).set_index("fri")
    big = w[w["drift"].abs() >= 0.003]
    return dict(weeks=len(w), corr=float(w["drift"].corr(w["next"])),
                fade_hit=float((np.sign(-big["drift"]) == np.sign(big["next"])).mean()), big_weeks=len(big),
                fade_avg=float((-np.sign(big["drift"]) * big["next"]).mean()))


def main():
    pd.set_option("display.width", 220)
    for sym in ("PAXGUSDT", "XAUUSDT"):
        h = data.load("1h", sym)
        S = signals(h)
        print(f"\n=== {sym} {h.index[0].date()}..{h.index[-1].date()} (decisions {S.index[0].date()}..)")
        print(evaluate(h, S).round(3).to_string())
        print("weekend fade:", {k: round(v, 3) if isinstance(v, float) else v for k, v in weekend(h).items()})


if __name__ == "__main__":
    main()
