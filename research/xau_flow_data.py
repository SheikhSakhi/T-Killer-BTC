"""Download XAUUSDT order-flow data from data.binance.vision into cache/xauflow (parquet per kind).

bookDepth (30s snapshots of depth at +-0.2..5%), metrics (5m OI and long/short ratios),
premiumIndexKlines 1h, fundingRate monthly. Klines with taker buy volume are already in cache.
"""
import io
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "xauflow")
B = "https://data.binance.vision/data/futures/um"
SYM = "XAUUSDT"
START = date(2025, 12, 11)
S = requests.Session()


def get(url):
    r = S.get(url, timeout=60)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    return pd.read_csv(z.open(z.namelist()[0]))


def days():
    d, out = START, []
    while d < date.today():
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def book_hourly(day):
    d = get(f"{B}/daily/bookDepth/{SYM}/{SYM}-bookDepth-{day}.zip")
    if d is None:
        return None
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    w = d.pivot_table(index="timestamp", columns="percentage", values="notional")
    w.columns = [f"n{c:+.1f}" for c in w.columns]
    # hourly average of the snapshots, labelled by the hour they cover (known at the end of it)
    return w.resample("1h").mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = days()
    with ThreadPoolExecutor(12) as ex:
        books = [b for b in ex.map(book_hourly, ds) if b is not None]
    pd.concat(books).sort_index().to_parquet(os.path.join(OUT, "book_1h.parquet"))
    print("book hours", sum(len(b) for b in books), flush=True)

    with ThreadPoolExecutor(12) as ex:
        mets = [m for m in ex.map(lambda d: get(f"{B}/daily/metrics/{SYM}/{SYM}-metrics-{d}.zip"), ds) if m is not None]
    m = pd.concat(mets)
    m["time"] = pd.to_datetime(m["create_time"])
    m.drop(columns=["create_time", "symbol"]).set_index("time").sort_index().to_parquet(os.path.join(OUT, "metrics.parquet"))
    print("metrics rows", len(m), flush=True)

    with ThreadPoolExecutor(12) as ex:
        prem = [p for p in ex.map(lambda d: get(f"{B}/daily/premiumIndexKlines/{SYM}/1h/{SYM}-1h-{d}.zip"), ds) if p is not None]
    p = pd.concat([x.iloc[:, :5].set_axis(["open_time", "open", "high", "low", "close"], axis=1) for x in prem])
    p = p[pd.to_numeric(p["open_time"], errors="coerce").notna()].astype(float)
    p["time"] = pd.to_datetime(p["open_time"].where(p["open_time"] < 1e14, p["open_time"] // 1000), unit="ms")
    p.set_index("time")["close"].rename("premium").sort_index().to_frame().to_parquet(os.path.join(OUT, "premium_1h.parquet"))
    print("premium hours", len(p), flush=True)

    months = pd.period_range("2025-12", pd.Period(date.today(), "M") - 1, freq="M")
    fr = [get(f"{B}/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{mo}.zip") for mo in months]
    f = pd.concat([x for x in fr if x is not None])
    f["time"] = pd.to_datetime(f["calc_time"], unit="ms").dt.round("h")
    f.set_index("time")["last_funding_rate"].sort_index().to_frame().to_parquet(os.path.join(OUT, "funding.parquet"))
    print("funding rows", len(f), flush=True)


if __name__ == "__main__":
    main()
