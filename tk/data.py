"""Historical USD-M futures data from data.binance.vision, cached as parquet.

python -m tk.data --months 24
"""
import argparse
import io
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

from tk import config

BASE = "https://data.binance.vision/data/futures/um"
CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
S = requests.Session()


def _zip_csv(url):
    r = S.get(url, timeout=60)
    if r.status_code != 200:
        return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    raw = z.open(z.namelist()[0]).read()
    first = raw[:200].decode(errors="ignore").split("\n")[0]
    header = 0 if any(c.isalpha() for c in first.split(",")[0]) else None
    return pd.read_csv(io.BytesIO(raw), header=header)


def _months(n):
    d = date.today().replace(day=1)
    out = []
    for _ in range(n):
        d = (d - timedelta(days=1)).replace(day=1)
        out.append(d.strftime("%Y-%m"))
    return out[::-1]


def _days_this_month():
    today = date.today()
    return [(today.replace(day=1) + timedelta(days=i)).isoformat() for i in range((today - today.replace(day=1)).days)]


def _ms(s):
    s = pd.to_numeric(s)
    return pd.to_datetime(s.where(s < 1e14, s // 1000), unit="ms")


def klines(sym, interval, months):
    urls = [f"{BASE}/monthly/klines/{sym}/{interval}/{sym}-{interval}-{m}.zip" for m in _months(months)]
    urls += [f"{BASE}/daily/klines/{sym}/{interval}/{sym}-{interval}-{d}.zip" for d in _days_this_month()]
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(_zip_csv, urls) if p is not None]
    df = pd.concat([p.set_axis(KCOLS, axis=1) for p in parts])
    df["time"] = _ms(df["open_time"])
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    return df[["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]].astype(float)


def funding(sym, months):
    urls = [f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{m}.zip" for m in _months(months)]
    with ThreadPoolExecutor(8) as ex:
        parts = [p for p in ex.map(_zip_csv, urls) if p is not None]
    df = pd.concat(parts)
    df.columns = [c.strip() for c in df.columns]
    df["time"] = _ms(df["calc_time"]).dt.round("h")
    return df.drop_duplicates("time").set_index("time")["last_funding_rate"].astype(float).sort_index()


def metrics(sym, months):
    """5-minute open interest and long/short ratios."""
    start = date.fromisoformat(_months(months)[0] + "-01")
    days = [(start + timedelta(days=i)).isoformat() for i in range((date.today() - start).days)]
    urls = [f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{d}.zip" for d in days]
    with ThreadPoolExecutor(16) as ex:
        parts = [p for p in ex.map(_zip_csv, urls) if p is not None]
    df = pd.concat(parts)
    df["time"] = pd.to_datetime(df["create_time"])
    df = df.drop_duplicates("time").sort_values("time").set_index("time")
    return df.drop(columns=["create_time", "symbol"]).astype(float)


def path(kind, sym):
    return os.path.join(CACHE, f"{sym}-{kind}.parquet")


def load(kind, sym):
    return pd.read_parquet(path(kind, sym))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    for sym in config.COINS + config.CONTEXT:
        for kind, fn in [("5m", lambda s: klines(s, "5m", a.months)),
                         ("1h", lambda s: klines(s, "1h", a.months)),
                         ("funding", lambda s: funding(s, a.months).to_frame()),
                         ("metrics", lambda s: metrics(s, a.months))]:
            try:
                df = fn(sym)
                df.to_parquet(path(kind, sym))
                print(sym, kind, len(df), df.index[0], df.index[-1], flush=True)
            except Exception as e:  # keep going; report gaps
                print(sym, kind, "FAILED", repr(e)[:200], flush=True)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- incremental updates (server)

def update(sym, client):
    """Append archive days since the cache ends, then the last hours from the live API."""
    for kind, interval in (("5m", "5m"), ("1h", "1h")):
        old = load(kind, sym)
        start = old.index[-1].normalize().date()
        days = [(start + timedelta(days=i)).isoformat() for i in range(1, (date.today() - start).days)]
        urls = [f"{BASE}/daily/klines/{sym}/{interval}/{sym}-{interval}-{d}.zip" for d in days]
        parts = [p for p in map(_zip_csv, urls) if p is not None]
        new = [old]
        if parts:
            df = pd.concat([p.set_axis(KCOLS, axis=1) for p in parts])
            df["time"] = _ms(df["open_time"])
            new.append(df.set_index("time")[old.columns].astype(float))
        new.append(client.klines(sym, interval, limit=1000 if kind == "5m" else 200)[old.columns])
        out = pd.concat(new)
        out[~out.index.duplicated(keep="last")].sort_index().to_parquet(path(kind, sym))

    f = load("funding", sym)["last_funding_rate"]
    f = pd.concat([f, client.funding_history(sym, 200)])
    f[~f.index.duplicated(keep="last")].sort_index().to_frame("last_funding_rate").to_parquet(path("funding", sym))

    m = load("metrics", sym)
    recent = pd.DataFrame({"sum_open_interest_value": client.oi_history(sym),
                           "sum_toptrader_long_short_ratio": client.top_ls_history(sym)})
    m = pd.concat([m, recent])
    m[~m.index.duplicated(keep="last")].sort_index().to_parquet(path("metrics", sym))
