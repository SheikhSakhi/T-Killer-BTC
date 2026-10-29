"""Live XAUUSDT collector: runs all the time on the server and writes one row per minute.

Streams (Binance USD-M websocket): every aggregated trade, top-20 order book every 500 ms,
liquidations, mark/index price every second; open interest polled by REST every minute.
Each minute row (cache/minutes/YYYY-MM-DD.csv):
  OHLC from trades, aggressive buy and sell volume (USD), trade count, large trades (>= 50k USD)
  net, book imbalance top-5 and top-20 levels (average over the minute and at its end), spread,
  long and short liquidations (USD), premium of mark over index, funding rate, open interest.

python -m tk.collect
"""
import asyncio
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone

import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "cache", "minutes")
SYM = "xauusdt"
WS = f"wss://fstream.binance.com/stream?streams={SYM}@aggTrade/{SYM}@depth20@500ms/{SYM}@forceOrder/{SYM}@markPrice@1s"
BIG_USD = 50_000
COLS = ["minute", "open", "high", "low", "close", "buy_usd", "sell_usd", "trades", "big_buy_usd", "big_sell_usd",
        "book5_imb", "book20_imb", "book5_imb_end", "book20_imb_end", "spread", "liq_long_usd", "liq_short_usd",
        "premium", "funding", "oi"]


class MinuteAgg:
    """Folds stream messages into per-minute rows; emit() is called when the minute changes."""

    def __init__(self, emit):
        self.emit = emit
        self.minute = None
        self.oi = None
        self._reset()

    def _reset(self):
        self.o = self.h = self.l = self.c = None
        self.buy = self.sell = self.big_buy = self.big_sell = 0.0
        self.n = 0
        self.b5 = []
        self.b20 = []
        self.spreads = []
        self.liq_long = self.liq_short = 0.0
        self.premium = self.funding = None

    def _roll(self, ts_ms):
        m = int(ts_ms // 60000)
        if self.minute is None:
            self.minute = m
        elif m != self.minute:
            self.flush()
            self.minute = m

    def flush(self):
        if self.minute is None or self.c is None:
            self._reset()
            return
        mean = lambda xs: sum(xs) / len(xs) if xs else None
        row = dict(minute=datetime.fromtimestamp(self.minute * 60, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                   open=self.o, high=self.h, low=self.l, close=self.c, buy_usd=round(self.buy, 2),
                   sell_usd=round(self.sell, 2), trades=self.n, big_buy_usd=round(self.big_buy, 2),
                   big_sell_usd=round(self.big_sell, 2), book5_imb=mean(self.b5), book20_imb=mean(self.b20),
                   book5_imb_end=self.b5[-1] if self.b5 else None, book20_imb_end=self.b20[-1] if self.b20 else None,
                   spread=mean(self.spreads), liq_long_usd=round(self.liq_long, 2), liq_short_usd=round(self.liq_short, 2),
                   premium=self.premium, funding=self.funding, oi=self.oi)
        self.emit(row)
        self._reset()

    def on(self, stream, d):
        if stream.endswith("@aggTrade"):
            self._roll(d["T"])
            p, q = float(d["p"]), float(d["q"])
            usd = p * q
            self.o = p if self.o is None else self.o
            self.h = p if self.h is None else max(self.h, p)
            self.l = p if self.l is None else min(self.l, p)
            self.c = p
            self.n += 1
            if d["m"]:              # buyer is the maker -> the aggressor sold
                self.sell += usd
                self.big_sell += usd if usd >= BIG_USD else 0
            else:
                self.buy += usd
                self.big_buy += usd if usd >= BIG_USD else 0
        elif "@depth" in stream:
            self._roll(d.get("E", time.time() * 1000))
            bids = [(float(p), float(q)) for p, q in d["b"]]
            asks = [(float(p), float(q)) for p, q in d["a"]]
            if bids and asks:
                for k, store in ((5, self.b5), (20, self.b20)):
                    b = sum(p * q for p, q in bids[:k])
                    a = sum(p * q for p, q in asks[:k])
                    store.append((b - a) / (b + a) if b + a else 0.0)
                self.spreads.append(asks[0][0] - bids[0][0])
        elif stream.endswith("@forceOrder"):
            o = d["o"]
            self._roll(d.get("E", o.get("T", time.time() * 1000)))
            usd = float(o["ap"] or o["p"]) * float(o["z"] or o["q"])
            if o["S"] == "SELL":        # a long position was force-sold
                self.liq_long += usd
            else:
                self.liq_short += usd
        elif "@markPrice" in stream:
            self._roll(d["E"])
            mark, index = float(d["p"]), float(d["i"])
            self.premium = mark / index - 1 if index else None
            self.funding = float(d["r"])


def writer(row):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, row["minute"][:10] + ".csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)


def poll_oi(agg):
    while True:
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/openInterest", params=dict(symbol=SYM.upper()), timeout=10)
            agg.oi = float(r.json()["openInterest"])
        except Exception as e:
            print("oi poll failed:", e, file=sys.stderr)
        time.sleep(60)


async def run():
    import websockets
    agg = MinuteAgg(writer)
    threading.Thread(target=poll_oi, args=(agg,), daemon=True).start()
    while True:
        try:
            async with websockets.connect(WS, ping_interval=20, max_size=2 ** 22) as ws:
                print("connected", WS, flush=True)
                async for raw in ws:
                    msg = json.loads(raw)
                    agg.on(msg["stream"], msg["data"])
        except Exception as e:
            print("stream dropped, reconnecting:", e, file=sys.stderr, flush=True)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(run())
