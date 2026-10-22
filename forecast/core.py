"""Shared pieces: candle download and Kronos forecasting for BTC/USDT 1h."""
import io
import json
import os
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LOOKBACK = 400
HORIZON = 24
MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"
FEATURES = ["open", "high", "low", "close", "volume", "amount"]
KRONOS_DIR = os.environ.get("KRONOS_DIR", os.path.join(os.path.dirname(__file__), "..", "vendor", "Kronos"))

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "amount", "trades", "tbb", "tbq", "ignore"]


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "T-Killer-BTC"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _frame(rows):
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS:
        df[c] = pd.to_numeric(df[c])
    # data.binance.vision switched to microseconds in 2025; the REST API uses milliseconds
    ms = np.where(df["open_time"] > 1e14, df["open_time"] // 1000, df["open_time"])
    df["timestamps"] = pd.to_datetime(ms, unit="ms")
    return df


API_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]


def from_api(days, host):
    """Live candles from a Binance REST host. data-api.binance.vision is the market-data-only
    mirror, which is not geo-blocked for US cloud runners the way api.binance.com is."""
    out, end = [], None
    need = days * 24
    while len(out) < need:
        url = f"{host}/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit=1000"
        if end is not None:
            url += f"&endTime={end}"
        batch = json.loads(_get(url))
        if not batch:
            break
        out = batch + out
        end = batch[0][0] - 1
    return _frame(out)


def from_archive(days):
    """Daily zips from data.binance.vision (published with ~1 day delay)."""
    rows = []
    today = datetime.now(timezone.utc).date()
    for i in range(days + 3, 0, -1):
        d = today - timedelta(days=i)
        url = (f"https://data.binance.vision/data/spot/daily/klines/{SYMBOL}/{INTERVAL}/"
               f"{SYMBOL}-{INTERVAL}-{d}.zip")
        try:
            z = zipfile.ZipFile(io.BytesIO(_get(url)))
        except Exception:
            continue
        rows += pd.read_csv(z.open(z.namelist()[0]), header=None).values.tolist()
    return _frame(rows)


def load_candles(days=25):
    """Closed hourly candles, oldest first. Tries the live API, falls back to the archive."""
    df = None
    for host in API_HOSTS:
        try:
            df, source = from_api(days, host), host.split("//")[1]
            break
        except Exception as e:
            print(f"{host} failed ({e})", file=sys.stderr)
    if df is None:
        df, source = from_archive(days), "archive"
    df = df.sort_values("timestamps").drop_duplicates("timestamps")
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    close_ms = np.where(df["close_time"] > 1e14, df["close_time"] / 1000, df["close_time"])
    df = df[close_ms < now_ms].reset_index(drop=True)
    return df, source


_predictor = None


def predictor():
    global _predictor
    if _predictor is None:
        import torch
        sys.path.insert(0, os.path.abspath(KRONOS_DIR))
        from model import Kronos, KronosPredictor, KronosTokenizer
        torch.manual_seed(0)
        np.random.seed(0)
        tok = KronosTokenizer.from_pretrained(TOKENIZER)
        mdl = Kronos.from_pretrained(MODEL)
        _predictor = KronosPredictor(mdl, tok, device="cpu", max_context=512)
    return _predictor


def forecast(df, end, samples):
    """Sample `samples` 24h close paths from the LOOKBACK candles before index `end`."""
    x = df.iloc[end - LOOKBACK:end]
    step = pd.Timedelta(hours=1)
    yts = pd.Series([x["timestamps"].iloc[-1] + step * (i + 1) for i in range(HORIZON)])
    # One batch of identical series = `samples` independent paths in a single pass
    # (sample_count would average them, which hides the spread we want to measure).
    xd, xt = x[FEATURES].reset_index(drop=True), x["timestamps"].reset_index(drop=True)
    out = predictor().predict_batch(
        df_list=[xd] * samples, x_timestamp_list=[xt] * samples, y_timestamp_list=[yts] * samples,
        pred_len=HORIZON, T=1.0, top_p=0.9, sample_count=1, verbose=False)
    return np.array([p["close"].values for p in out])
