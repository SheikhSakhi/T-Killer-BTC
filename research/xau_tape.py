"""Intra-hour tape dynamics from the live collector (data/hourly/minutes) vs the next 1h / 4h move.

Each hour h is described only by its own minutes; the target starts at the close of h.
Ideas tested (each counted in MEMORY.md "Tried"):
  flow        aggressive buy - sell over the hour, share of volume
  flow_last15 same over the last 15 minutes
  absorb      flow the price did not follow: residual of flow after the hour's own return (rolling fit, past only)
  big         large trades (>= 50k) net, share of volume
  liq         long minus short liquidations, share of volume (positive = longs were flushed)
  book5_end   top-5 levels bid/ask imbalance at the hour's end
  book20_avg  top-20 imbalance averaged over the hour
  book_chg    top-5 imbalance, last 10 minutes minus first 10 minutes
  oi_price    open-interest change times the sign of the hour's move (+ = positions built with the move)
  prem_chg    change of mark premium over the hour
python research/xau_tape.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def minutes():
    m = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "hourly", "minutes", "*.csv"))))
    m["minute"] = pd.to_datetime(m["minute"])
    g = m.groupby("minute")           # a late trade can produce a second row for the same minute
    out = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                buy=("buy_usd", "sum"), sell=("sell_usd", "sum"), big_buy=("big_buy_usd", "sum"),
                big_sell=("big_sell_usd", "sum"), liq_long=("liq_long_usd", "sum"), liq_short=("liq_short_usd", "sum"),
                b5=("book5_imb", "mean"), b20=("book20_imb", "mean"), b5_end=("book5_imb_end", "last"),
                prem=("premium", "last"), oi=("oi", "last"))
    return out


def hourly(m):
    h = m.index.floor("h")
    G = m.groupby(h)
    H = pd.DataFrame({"close": G["close"].last(), "open": G["open"].first(), "n": G.size()})
    vol = G["buy"].sum() + G["sell"].sum()
    H["flow"] = (G["buy"].sum() - G["sell"].sum()) / vol
    last = m[m.index.minute >= 45].groupby(m[m.index.minute >= 45].index.floor("h"))
    H["flow_last15"] = (last["buy"].sum() - last["sell"].sum()) / (last["buy"].sum() + last["sell"].sum())
    H["big"] = (G["big_buy"].sum() - G["big_sell"].sum()) / vol
    H["liq"] = (G["liq_long"].sum() - G["liq_short"].sum()) / vol
    H["book5_end"] = G["b5_end"].last()
    H["book20_avg"] = G["b20"].mean()
    first = m[m.index.minute < 10].groupby(m[m.index.minute < 10].index.floor("h"))["b5"].mean()
    lastb = m[m.index.minute >= 50].groupby(m[m.index.minute >= 50].index.floor("h"))["b5"].mean()
    H["book_chg"] = lastb - first
    ret = H["close"] / H["open"] - 1
    H["oi_price"] = (G["oi"].last() / G["oi"].first() - 1) * np.sign(ret)
    H["prem_chg"] = G["prem"].last() - G["prem"].first()
    z = lambda s: (s - s.rolling(168, min_periods=48).mean()) / s.rolling(168, min_periods=48).std()
    rz, fz = z(ret), z(H["flow"])
    beta = (rz * fz).rolling(168, min_periods=48).mean()        # past-only slope of flow on return
    H["absorb"] = fz - beta * rz
    H = H[H["n"] >= 50]                                          # hours with most minutes present
    full = pd.date_range(H.index.min(), H.index.max(), freq="h")
    c = H["close"].reindex(full)
    H["fwd_1h"] = (c.shift(-1) / c - 1).reindex(H.index)
    H["fwd_4h"] = (c.shift(-4) / c - 1).reindex(H.index)
    return H


FEATS = ["flow", "flow_last15", "absorb", "big", "liq", "book5_end", "book20_avg", "book_chg", "oi_price", "prem_chg"]


def table(H):
    rows = []
    half = H.index[0] + (H.index[-1] - H.index[0]) / 2
    for hz in (1, 4):
        y = H[f"fwd_{hz}h"]
        sub = H.index[::hz]
        for f in FEATS:
            d = pd.concat([H[f], y], axis=1).loc[sub].dropna()
            a, b = d[d.index < half], d[d.index >= half]
            ic = lambda x: x.iloc[:, 0].rank().corr(x.iloc[:, 1].rank())
            q = d[d.iloc[:, 0] >= d.iloc[:, 0].quantile(0.8)].iloc[:, 1]
            lo = d[d.iloc[:, 0] <= d.iloc[:, 0].quantile(0.2)].iloc[:, 1]
            rows.append(dict(h=f"{hz}h", feature=f, n=len(d), ic_1st=ic(a), ic_2nd=ic(b), ic_all=ic(d),
                             top20_up=(q > 0).mean(), bot20_up=(lo > 0).mean()))
    t = pd.DataFrame(rows)
    se = 1 / np.sqrt(t["n"] / 2)
    t["stable"] = (np.sign(t.ic_1st) == np.sign(t.ic_2nd)) & (t[["ic_1st", "ic_2nd"]].abs().min(axis=1) >= 2 * se)
    return t


def main():
    pd.set_option("display.width", 200)
    H = hourly(minutes())
    print(f"hours {len(H)}  {H.index[0]} .. {H.index[-1]}")
    t = table(H)
    print(t.round(3).to_string(index=False))
    print("\nnow (last full hour", H.index[-1], "):")
    pct = H[FEATS].rank(pct=True).iloc[-1]
    print(pd.DataFrame({"value": H[FEATS].iloc[-1], "percentile": pct}).round(3).to_string())
    t.to_csv(os.path.join(ROOT, "results", "xau_tape_ic.csv"), index=False)


if __name__ == "__main__":
    main()
