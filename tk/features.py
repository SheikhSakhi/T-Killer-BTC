"""Decision-time features: one row per (day, session) per coin, using only data before the session opens."""
import numpy as np
import pandas as pd

from tk import config, data


def session_starts(days):
    rows = []
    for d in days:
        for name, hhmm in config.SESSIONS.items():
            rows.append((d, name, pd.Timestamp(f"{d.date()} {hhmm}")))
    return pd.DataFrame(rows, columns=["day", "session", "start"])


def build(sym, btc_1h=None):
    h = data.load("1h", sym)
    f = data.load("funding", sym)["last_funding_rate"]
    m = data.load("metrics", sym)

    logret = np.log(h["close"]).diff()
    tr = pd.concat([h["high"] - h["low"], (h["high"] - h["close"].shift()).abs(),
                    (h["low"] - h["close"].shift()).abs()], axis=1).max(axis=1)
    feats = pd.DataFrame(index=h.index)
    # every column is known at the CLOSE of the bar, so shift by one bar to be known at the bar's open time + 1h
    feats["atr_pct"] = (tr.rolling(14).mean() / h["close"])
    feats["vol24"] = logret.rolling(24).std() * np.sqrt(24)
    feats["vol20d"] = logret.rolling(24 * 20).std() * np.sqrt(24)
    feats["vol_pct"] = feats["vol20d"].rolling(24 * 120, min_periods=24 * 60).rank(pct=True)
    feats["squeeze"] = feats["vol24"] / feats["vol20d"]
    feats["hi24"] = h["high"].rolling(24).max()
    feats["lo24"] = h["low"].rolling(24).min()
    feats["ret24"] = h["close"].pct_change(24)
    feats["close"] = h["close"]
    if btc_1h is not None:
        feats["btc_ret24"] = btc_1h["close"].pct_change(24).reindex(h.index)
    feats.index = feats.index + pd.Timedelta(hours=1)  # value available from this time on

    fz = (f - f.rolling(90, min_periods=30).mean()) / f.rolling(90, min_periods=30).std()
    fund = pd.DataFrame({"funding": f, "funding_z": fz})
    fund.index = fund.index + pd.Timedelta(minutes=1)

    oi = m["sum_open_interest_value"]
    met = pd.DataFrame({
        "oi_chg24": oi.pct_change(288),
        "lsr_top_z": (m["sum_toptrader_long_short_ratio"] - m["sum_toptrader_long_short_ratio"].rolling(288 * 30, min_periods=288 * 7).mean())
        / m["sum_toptrader_long_short_ratio"].rolling(288 * 30, min_periods=288 * 7).std(),
    })
    met.index = met.index + pd.Timedelta(minutes=5)

    days = pd.date_range(h.index[0].normalize() + pd.Timedelta(days=130), h.index[-1].normalize(), freq="D")
    ss = session_starts(days)
    t = ss["start"]
    out = ss.copy()
    for src in (feats, fund, met):
        src = src.sort_index()
        idx = src.index.searchsorted(t, side="right") - 1
        ok = idx >= 0
        vals = src.iloc[np.clip(idx, 0, None)].reset_index(drop=True)
        vals[~ok] = np.nan
        out = pd.concat([out, vals], axis=1)
    out["sym"] = sym
    return out
