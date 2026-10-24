"""Minimal Binance USD-M futures client (public data + signed trading).

Keys come from the environment (BINANCE_KEY / BINANCE_SECRET); BINANCE_TESTNET=1 targets the testnet.
Protective orders go to the Algo Order API that Binance moved conditional orders to, with a
fallback to the classic order endpoint.
"""
import hashlib
import hmac
import math
import os
import time
from urllib.parse import urlencode

import pandas as pd
import requests

LIVE = "https://fapi.binance.com"
TEST = "https://testnet.binancefuture.com"


class BinanceError(Exception):
    pass


class Futures:
    def __init__(self, key=None, secret=None, testnet=None):
        self.key = key or os.environ.get("BINANCE_KEY", "")
        self.secret = (secret or os.environ.get("BINANCE_SECRET", "")).encode()
        testnet = os.environ.get("BINANCE_TESTNET", "1") == "1" if testnet is None else testnet
        self.base = TEST if testnet else LIVE
        self.testnet = testnet
        self.s = requests.Session()
        self.s.headers["X-MBX-APIKEY"] = self.key
        self._filters = {}

    # ------------------------------------------------------------ transport
    def _req(self, method, path, params=None, signed=False, base=None):
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = 10000
            q = urlencode(params)
            params["signature"] = hmac.new(self.secret, q.encode(), hashlib.sha256).hexdigest()
        r = self.s.request(method, (base or self.base) + path, params=params, timeout=20)
        if r.status_code >= 400:
            raise BinanceError(f"{method} {path} {r.status_code}: {r.text[:300]}")
        return r.json()

    # ------------------------------------------------------------ public data
    def klines(self, sym, interval="5m", limit=500, start=None):
        p = dict(symbol=sym, interval=interval, limit=limit)
        if start is not None:
            p["startTime"] = int(pd.Timestamp(start).timestamp() * 1000)
        rows = self._req("GET", "/fapi/v1/klines", p)
        cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]
        if not rows:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="time"), dtype=float)
        df = pd.DataFrame(rows).iloc[:, :11]
        df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                      "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
        df["time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("time")
        return df[["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]].astype(float)

    def funding_history(self, sym, limit=200):
        rows = self._req("GET", "/fapi/v1/fundingRate", dict(symbol=sym, limit=limit))
        s = pd.Series({pd.to_datetime(r["fundingTime"], unit="ms").round("h"): float(r["fundingRate"]) for r in rows})
        return s.sort_index()

    def oi_history(self, sym, period="5m", limit=500):
        # the testnet has no /futures/data endpoints; positioning data always comes from the live market
        rows = self._req("GET", "/futures/data/openInterestHist", dict(symbol=sym, period=period, limit=limit), base=LIVE)
        return pd.Series({pd.to_datetime(r["timestamp"], unit="ms"): float(r["sumOpenInterestValue"]) for r in rows}).sort_index()

    def top_ls_history(self, sym, period="5m", limit=500):
        rows = self._req("GET", "/futures/data/topLongShortPositionRatio", dict(symbol=sym, period=period, limit=limit),
                         base=LIVE)
        return pd.Series({pd.to_datetime(r["timestamp"], unit="ms"): float(r["longShortRatio"]) for r in rows}).sort_index()

    def price(self, sym):
        return float(self._req("GET", "/fapi/v1/ticker/price", dict(symbol=sym))["price"])

    def filters(self, sym):
        if sym not in self._filters:
            info = self._req("GET", "/fapi/v1/exchangeInfo")
            for s in info["symbols"]:
                f = {x["filterType"]: x for x in s["filters"]}
                self._filters[s["symbol"]] = dict(
                    step=float(f["MARKET_LOT_SIZE"]["stepSize"]), min_qty=float(f["MARKET_LOT_SIZE"]["minQty"]),
                    tick=float(f["PRICE_FILTER"]["tickSize"]), min_notional=float(f["MIN_NOTIONAL"]["notional"]))
        return self._filters[sym]

    # ------------------------------------------------------------ account
    def balance(self):
        for a in self._req("GET", "/fapi/v2/balance", signed=True):
            if a["asset"] == "USDT":
                return float(a["balance"]), float(a["availableBalance"])
        return 0.0, 0.0

    def positions(self):
        out = {}
        for p in self._req("GET", "/fapi/v2/positionRisk", signed=True):
            amt = float(p["positionAmt"])
            if amt != 0:
                out[p["symbol"]] = dict(qty=amt, entry=float(p["entryPrice"]), upnl=float(p["unRealizedProfit"]))
        return out

    def income(self, start_ms):
        return self._req("GET", "/fapi/v1/income", dict(startTime=start_ms, limit=1000), signed=True)

    # ------------------------------------------------------------ trading
    def round_qty(self, sym, qty):
        f = self.filters(sym)
        return math.floor(qty / f["step"]) * f["step"]

    def round_price(self, sym, px):
        f = self.filters(sym)
        return round(round(px / f["tick"]) * f["tick"], 10)

    def setup_symbol(self, sym, leverage):
        try:
            self._req("POST", "/fapi/v1/marginType", dict(symbol=sym, marginType="ISOLATED"), signed=True)
        except BinanceError as e:
            if "-4046" not in str(e):  # already isolated
                raise
        self._req("POST", "/fapi/v1/leverage", dict(symbol=sym, leverage=int(leverage)), signed=True)

    def market(self, sym, side, qty, reduce_only=False):
        p = dict(symbol=sym, side="BUY" if side > 0 else "SELL", type="MARKET", quantity=self._fmt(qty),
                 newOrderRespType="RESULT")
        if reduce_only:
            p["reduceOnly"] = "true"
        return self._req("POST", "/fapi/v1/order", p, signed=True)

    def protect(self, sym, side, stop, target):
        """Exchange-side stop-loss and take-profit that close the whole position."""
        close_side = "SELL" if side > 0 else "BUY"
        out = []
        for typ, px in (("STOP_MARKET", stop), ("TAKE_PROFIT_MARKET", target)):
            px = self._fmt(self.round_price(sym, px))
            try:
                out.append(self._req("POST", "/fapi/v1/algoOrder", dict(
                    algoType="CONDITIONAL", symbol=sym, side=close_side, type=typ, triggerPrice=px,
                    closePosition="true", workingType="MARK_PRICE"), signed=True))
            except BinanceError as e:
                if " 404" not in str(e):
                    raise
                out.append(self._req("POST", "/fapi/v1/order", dict(
                    symbol=sym, side=close_side, type=typ, stopPrice=px, closePosition="true",
                    workingType="MARK_PRICE"), signed=True))
        return out

    def cancel_all(self, sym):
        for path in ("/fapi/v1/allOpenOrders", "/fapi/v1/algoOpenOrders"):
            try:
                self._req("DELETE", path, dict(symbol=sym), signed=True)
            except BinanceError:
                pass

    @staticmethod
    def _fmt(x):
        return f"{x:.10f}".rstrip("0").rstrip(".")
