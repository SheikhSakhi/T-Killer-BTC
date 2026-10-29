"""Order-flow study for XAUUSDT: do measurable forces predict the next 4 / 24 hours?

Every hour t (features known at the close of hour t, nothing later):
  flow     taker buy share of volume over 1/4/24h, volume vs its 7-day norm
  book     bid/ask depth imbalance at 0.2/1/2/5% from price, and its 24h change
  crowd    open interest change 1/4/24h, top-trader and global long/short ratios (z vs 7 days),
           taker long/short volume ratio
  basis    premium of the perpetual over the index, funding rate
  context  past returns 4/24h, 24h volatility, dollar 20-day change (lagged 9 days)

Tests
  1. Each feature alone: rank correlation (IC) with the forward return, on the first and the second
     half separately, sampled without overlap. A real force keeps its sign in both halves.
  2. All features together: ridge regression re-fitted every week on the past only (walk-forward),
     trading the prediction's sign at 24h with the 2% / 4% exits, compared with the trend-vote rule.

python research/xau_flow.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from tk import data, xau  # noqa: E402

F = os.path.join(ROOT, "cache", "xauflow")


def zs(s, n):
    return (s - s.rolling(n, min_periods=n // 2).mean()) / s.rolling(n, min_periods=n // 2).std()


def panel():
    k = data.load("1h", "XAUUSDT")
    book = pd.read_parquet(os.path.join(F, "book_1h.parquet"))
    met = pd.read_parquet(os.path.join(F, "metrics.parquet")).resample("1h").last()
    met.index = met.index - pd.Timedelta(hours=1)        # value at hh:55 belongs to the hour that started at hh-1... aligned to bar open
    prem = pd.read_parquet(os.path.join(F, "premium_1h.parquet"))["premium"]
    fund = pd.read_parquet(os.path.join(F, "funding.parquet"))["last_funding_rate"]
    usd = xau.dollar()

    X = pd.DataFrame(index=k.index)
    qv, tb = k["quote_volume"], k["taker_buy_quote"]
    for n in (1, 4, 24):
        X[f"flow_taker_buy_{n}h"] = tb.rolling(n).sum() / qv.rolling(n).sum() - 0.5
    X["flow_volume_z"] = zs(np.log(qv), 168)
    # book is the average of snapshots inside the hour: known at the end of the bar
    b = book.reindex(k.index)
    for p in ("0.2", "1.0", "2.0", "5.0"):
        bid, ask = b[f"n-{p}"], b[f"n+{p}"]
        X[f"book_imb_{p}"] = (bid - ask) / (bid + ask)
    X["book_imb_1.0_chg24"] = X["book_imb_1.0"] - X["book_imb_1.0"].shift(24)
    X["book_depth_1.0_z"] = zs(np.log(b["n-1.0"] + b["n+1.0"]), 168)
    m = met.reindex(k.index).ffill(limit=2)
    oi = m["sum_open_interest_value"]
    for n in (1, 4, 24):
        X[f"crowd_oi_chg_{n}h"] = oi / oi.shift(n) - 1
    X["crowd_top_ls_z"] = zs(m["sum_toptrader_long_short_ratio"], 168)
    X["crowd_global_ls_z"] = zs(m["count_long_short_ratio"], 168)
    X["crowd_taker_ls_4h"] = np.log(m["sum_taker_long_short_vol_ratio"]).rolling(4).mean()
    X["basis_premium"] = prem.reindex(k.index)
    X["basis_premium_24h"] = X["basis_premium"].rolling(24).mean()
    X["basis_funding"] = fund.reindex(k.index).ffill()
    lr = np.log(k["close"]).diff()
    X["ctx_ret_4h"] = k["close"].pct_change(4)
    X["ctx_ret_24h"] = k["close"].pct_change(24)
    X["ctx_vol_24h"] = lr.rolling(24).std()
    u = usd.reindex(pd.date_range(usd.index.min(), k.index.max(), freq="D")).ffill()
    uchg = (u / u.shift(20) - 1).shift(9)
    X["ctx_dollar_20d"] = uchg.reindex(k.index.normalize()).values
    Y = pd.DataFrame({f"fwd_{n}h": k["close"].shift(-n) / k["close"] - 1 for n in (4, 24)}, index=k.index)
    # features describe bar t; the trade starts at the close of bar t = open of t+1
    return X, Y, k


def ic_table(X, Y):
    rows = []
    half = X.index[0] + (X.index[-1] - X.index[0]) / 2
    for h in (4, 24):
        y = Y[f"fwd_{h}h"]
        sub = X.index[::h]                       # non-overlapping samples
        for c in X.columns:
            d = pd.concat([X[c], y], axis=1).loc[sub].dropna()
            a, b = d[d.index < half], d[d.index >= half]
            ic1 = a.iloc[:, 0].rank().corr(a.iloc[:, 1].rank())
            ic2 = b.iloc[:, 0].rank().corr(b.iloc[:, 1].rank())
            n = len(d)
            rows.append(dict(horizon=f"{h}h", feature=c, n=n, ic_1st=ic1, ic_2nd=ic2,
                             stable=np.sign(ic1) == np.sign(ic2) and min(abs(ic1), abs(ic2)) >= 0.05))
    return pd.DataFrame(rows)


def ridge_walk_forward(X, Y, k, horizon=24, lam=10.0, min_train=24 * 60):
    """Weekly re-fit on all past hours (targets fully known), predict the next week, hourly."""
    y = Y[f"fwd_{horizon}h"]
    cols = list(X.columns)
    Z = X[cols]
    pred = pd.Series(np.nan, index=X.index)
    weeks = pd.date_range(X.index[0].normalize() + pd.Timedelta(hours=min_train), X.index[-1], freq="7D")
    for w0 in weeks:
        train_end = w0 - pd.Timedelta(hours=horizon)        # purge: targets must be known at w0
        tr = pd.concat([Z, y], axis=1).loc[:train_end].dropna()
        if len(tr) < min_train // 2:
            continue
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        A = ((tr[cols] - mu) / sd).values
        b = tr[y.name].values
        beta = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]) / 100, A.T @ (b - b.mean()))
        te = Z.loc[w0:w0 + pd.Timedelta(days=7) - pd.Timedelta(hours=1)]
        pred.loc[te.index] = (((te - mu) / sd).fillna(0).values @ beta) + b.mean()
    return pred


def trade_daily(pred, k, usd):
    """Once a day at 00:00 UTC, trade the sign of the prediction with the analyst's exits."""
    res = []
    t = pred.dropna().index[0].normalize() + pd.Timedelta(days=1)
    while t < k.index[-1] - pd.Timedelta(hours=xau.HOLD_H):
        p = pred.get(t - pd.Timedelta(hours=1), np.nan)
        if p == p and p != 0:
            side = 1 if p > 0 else -1
            px = k["close"][k.index < t].iloc[-1]
            o, r, end = xau.simulate(k, t, side, px)
            vote = xau.side_at(k, t, usd)[0]
            o2, r2, _ = xau.simulate(k, t, vote, px)
            res.append(dict(t=t, side=side, R=(r - xau.FEE) / xau.SL, vote=vote, R_vote=(r2 - xau.FEE) / xau.SL))
            t = (end + pd.Timedelta(hours=1)).ceil("D") if end is not None else t + pd.Timedelta(days=1)
        else:
            t += pd.Timedelta(days=1)
    return pd.DataFrame(res)


def main():
    pd.set_option("display.width", 230)
    X, Y, k = panel()
    print(f"hours {len(X)}  {X.index[0]} .. {X.index[-1]}; features {X.shape[1]}")
    ic = ic_table(X, Y)
    print("\n== 1. Each force alone: rank correlation with the next 4h / 24h move (1st half | 2nd half)")
    print(ic.sort_values(["horizon", "stable", "ic_2nd"], ascending=[True, False, False]).round(3).to_string(index=False))
    print("stable in both halves (|IC| >= 0.05, same sign):", int(ic["stable"].sum()), "of", len(ic))

    usd = xau.dollar()
    for h in (24, 4):
        pred = ridge_walk_forward(X, Y, k, horizon=h)
        y = Y[f"fwd_{h}h"]
        d = pd.concat([pred, y], axis=1).dropna().iloc[::h]
        hit = (np.sign(d.iloc[:, 0]) == np.sign(d.iloc[:, 1])).mean()
        icp = d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())
        print(f"\n== 2. All forces combined, walk-forward ridge, horizon {h}h: direction hit {hit:.1%}, IC {icp:+.3f}, n={len(d)}")
        if h == 24:
            tr = trade_daily(pred, k, usd)
            half = tr["t"].min() + (tr["t"].max() - tr["t"].min()) / 2
            for name, part in (("1st", tr[tr["t"] < half]), ("2nd", tr[tr["t"] >= half])):
                print(f"   trades {name} half: n={len(part)}  flow model {part['R'].mean():+.3f}R  "
                      f"trend-vote rule {part['R_vote'].mean():+.3f}R  agree {np.mean(part['side'] == part['vote']):.0%}")
    ic.to_csv(os.path.join(ROOT, "results", "xau_flow_ic.csv"), index=False)


if __name__ == "__main__":
    main()
