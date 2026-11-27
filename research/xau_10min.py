"""Next-10-minutes model on the live tape: gradient-boosted trees learn combinations of tape patterns.

Every minute t is described by the minutes up to t only (flow, large trades, liquidations, book, OI, premium,
returns and volatility over 1/5/15/60 min, time of day). Target: did the price rise over the next 10 minutes.
Walk-forward by day: train on all earlier days (purged), predict the next day, never the same day.
Reported: hit rate overall and by confidence, the average move in the predicted direction vs costs, per half.
python research/xau_10min.py
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xau_tape  # noqa: E402

H = 10
COST = 0.0008          # taker in and out


def features(m):
    m = m.asfreq("1min")
    vol = (m["buy"] + m["sell"]).fillna(0)
    flow = (m["buy"] - m["sell"]).fillna(0)
    big = (m["big_buy"] - m["big_sell"]).fillna(0)
    liq = (m["liq_long"] - m["liq_short"]).fillna(0)
    c = m["close"].ffill()
    r1 = np.log(c).diff()
    X = pd.DataFrame(index=m.index)
    for w in (1, 5, 15, 60):
        v = vol.rolling(w).sum()
        X[f"flow_{w}"] = flow.rolling(w).sum() / v
        X[f"big_{w}"] = big.rolling(w).sum() / v
        X[f"liq_{w}"] = liq.rolling(w).sum() / v
        X[f"ret_{w}"] = c / c.shift(w) - 1
        X[f"vol_{w}"] = np.log(v + 1) - np.log(vol.rolling(1440, min_periods=200).mean() * w + 1)
    for w in (15, 60):
        X[f"rv_{w}"] = r1.rolling(w).std()
        X[f"oi_{w}"] = m["oi"].ffill() / m["oi"].ffill().shift(w) - 1
        X[f"b5chg_{w}"] = m["b5"].ffill() - m["b5"].ffill().shift(w)
    X["b5"] = m["b5"].ffill()
    X["b5_end"] = m["b5_end"].ffill()
    X["b20"] = m["b20"].ffill()
    X["prem"] = m["prem"].ffill()
    X["prem_chg15"] = X["prem"] - X["prem"].shift(15)
    X["pos_60"] = (c - c.rolling(60).min()) / (c.rolling(60).max() - c.rolling(60).min())
    X["hour"] = m.index.hour
    X["minute"] = m.index.minute
    y = c.shift(-H) / c - 1
    ok = m["close"].notna() & c.shift(-H).notna()
    return X[ok], y[ok]


def walk_forward(X, y, min_days=7):
    days = X.index.normalize().unique()
    p = pd.Series(np.nan, index=X.index)
    for d in days[min_days:]:
        tr = X.index < d - pd.Timedelta(minutes=H)
        te = (X.index >= d) & (X.index < d + pd.Timedelta(days=1))
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                             min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        clf.fit(X[tr], (y[tr] > 0).astype(int))
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p


def report(p, y):
    d = pd.DataFrame({"p": p, "y": y}).dropna().iloc[::H]          # non-overlapping 10-minute windows
    d["side"] = np.where(d.p >= 0.5, 1, -1)
    d["hit"] = (np.sign(d.y) == d.side)
    d["gain"] = d.side * d.y
    d["conf"] = (d.p - 0.5).abs()
    half = d.index[0] + (d.index[-1] - d.index[0]) / 2
    rows = []
    for name, part in (("1st", d[d.index < half]), ("2nd", d[d.index >= half]), ("all", d)):
        for lo in (0.0, 0.05, 0.10, 0.15):
            s = part[part.conf >= lo]
            rows.append(dict(half=name, conf_from=0.5 + lo, n=len(s), hit=s.hit.mean(), avg_gain_pct=s.gain.mean() * 100,
                             after_cost_pct=(s.gain.mean() - COST) * 100))
    return pd.DataFrame(rows), d


def main():
    pd.set_option("display.width", 200)
    m = xau_tape.minutes()
    X, y = features(m)
    print(f"minutes {len(X)}  {X.index[0]} .. {X.index[-1]}; features {X.shape[1]}; "
          f"avg |10-min move| {y.abs().mean() * 100:.3f}%  cost {COST * 100:.2f}%")
    p = walk_forward(X, y)
    t, d = report(p, y)
    print(t.round(3).to_string(index=False))
    t.to_csv(os.path.join(xau_tape.ROOT, "results", "xau_10min.csv"), index=False)


if __name__ == "__main__":
    main()
