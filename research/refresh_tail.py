"""Bring cached symbols up to now from a US-hosted session: archive days up to yesterday, then
today's candles and funding from the testnet (tracks the real market within ~0.05%).
Open interest and L/S stay at yesterday's archive (live endpoints are geo-blocked here)."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tk import data  # noqa: E402
from tk.exchange import Futures  # noqa: E402

x = Futures(testnet=True)


def tail(sym):
    for kind in ("5m", "1h"):
        old = data.load(kind, sym)
        start = old.index[-1].normalize().date()
        days = [(start + timedelta(days=i)).isoformat() for i in range(0, (date.today() - start).days)]
        urls = [f"{data.BASE}/daily/klines/{sym}/{kind}/{sym}-{kind}-{d}.zip" for d in days]
        parts = [p for p in map(data._zip_csv, urls) if p is not None]
        new = [old]
        if parts:
            df = pd.concat([p.set_axis(data.KCOLS, axis=1) for p in parts])
            df["time"] = data._ms(df["open_time"])
            new.append(df.set_index("time")[old.columns].astype(float))
        today = pd.Timestamp(date.today())
        k = x.klines(sym, kind, limit=1000 if kind == "5m" else 48, start=today if kind == "5m" else today - pd.Timedelta(days=1))
        new.append(k[k.index >= today][old.columns])
        out = pd.concat(new)
        out[~out.index.duplicated(keep="last")].sort_index().to_parquet(data.path(kind, sym))
    f = data.load("funding", sym)["last_funding_rate"]
    f = pd.concat([f, x.funding_history(sym, 100)])
    f[~f.index.duplicated(keep="last")].sort_index().to_frame("last_funding_rate").to_parquet(data.path("funding", sym))
    m = data.load("metrics", sym)
    start = m.index[-1].normalize().date()
    days = [(start + timedelta(days=i)).isoformat() for i in range(1, (date.today() - start).days)]
    parts = [p for p in map(data._zip_csv, [f"{data.BASE}/daily/metrics/{sym}/{sym}-metrics-{d}.zip" for d in days]) if p is not None]
    if parts:
        df = pd.concat(parts)
        df["time"] = pd.to_datetime(df["create_time"])
        df = df.set_index("time").drop(columns=["create_time", "symbol"]).astype(float)
        m = pd.concat([m, df])
        m[~m.index.duplicated(keep="last")].sort_index().to_parquet(data.path("metrics", sym))
    return sym, str(data.load("5m", sym).index[-1])


if __name__ == "__main__":
    with ThreadPoolExecutor(8) as ex:
        for s, last in ex.map(tail, sys.argv[1].split(",")):
            print(s, "->", last, flush=True)
