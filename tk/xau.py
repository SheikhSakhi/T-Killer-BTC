"""XAUUSDT analyst: one command, one answer.

python -m tk.xau [risk_usd]

Rules (tested on XAUUSDT Dec 2025 - Oct 2026 and PAXGUSDT Mar 2025 - Oct 2026, see results/xau.md):
- Direction: majority vote of three signals (never a tie), see research/xau_factors.py:
    price vs 20-day average, price vs 50-day average, and the US dollar (FRED broad index DTWEXBGS):
    dollar down over 20 days -> gold LONG vote. The dollar value used is at least 9 days old, so the
    rule never relies on data that was not yet published.
- Stop 2%, target 4%, close after 7 days if neither is hit.
- Size so that the stop loses `risk_usd` including fees; leverage only sets the margin.
- Volatility: current daily sigma from the last 7 days of hourly returns, with ranges for 1 and 7 days.
Every call is logged to data/xau/signals.csv and earlier signals are scored, so the live record is honest.
"""
import io
import os
import sys
import zipfile
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests

SYM = "XAUUSDT"
SL, TP, HOLD_H = 0.02, 0.04, 168
FEE = 0.0014                  # taker in + out + slippage, share of notional
LEV = 10
LIMIT_BETTER = 0.003          # limit entry 0.3% better than the signal price, valid 24h (research: +0.25/+0.12R vs +0.20/+0.11R)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOG = os.path.join(ROOT, "data", "xau", "signals.csv")
CACHE = os.path.join(ROOT, "cache", "XAUUSDT-1h-archive.parquet")
SOURCES = [("live", "https://fapi.binance.com"), ("testnet", "https://testnet.binancefuture.com")]


def recent(limit=1000):
    """Last ~41 days of 1h candles: live API if reachable (server), else the testnet mirror."""
    for name, base in SOURCES:
        try:
            r = requests.get(f"{base}/fapi/v1/klines", params=dict(symbol=SYM, interval="1h", limit=limit), timeout=15)
            r.raise_for_status()
            rows = r.json()
            df = pd.DataFrame([x[:6] for x in rows], columns=["t", "open", "high", "low", "close", "volume"])
            df["t"] = pd.to_datetime(df["t"], unit="ms")
            return df.set_index("t").astype(float), name, base
        except Exception:
            continue
    raise RuntimeError("no price source reachable")


def funding(base):
    try:
        r = requests.get(f"{base}/fapi/v1/premiumIndex", params=dict(symbol=SYM), timeout=15).json()
        return float(r["lastFundingRate"])
    except Exception:
        return float("nan")


def archive():
    """Full history from data.binance.vision (monthly + this month's days), cached."""
    if os.path.exists(CACHE) and (datetime.now().timestamp() - os.path.getmtime(CACHE)) < 86400:
        return pd.read_parquet(CACHE)
    parts, today = [], date.today()
    months = pd.period_range("2025-12", pd.Period(today, "M") - 1, freq="M")
    urls = [f"https://data.binance.vision/data/futures/um/monthly/klines/{SYM}/1h/{SYM}-1h-{m}.zip" for m in months]
    urls += [f"https://data.binance.vision/data/futures/um/daily/klines/{SYM}/1h/{SYM}-1h-{d.date()}.zip"
             for d in pd.date_range(today.replace(day=1), periods=max(today.day - 1, 0))]
    for u in urls:
        r = requests.get(u, timeout=30)
        if r.status_code != 200:
            continue
        z = zipfile.ZipFile(io.BytesIO(r.content))
        d = pd.read_csv(z.open(z.namelist()[0]), header=None)
        if not str(d.iloc[0, 0]).isdigit():
            d = d.iloc[1:]
        d = d.iloc[:, :6].astype(float)
        d.columns = ["t", "open", "high", "low", "close", "volume"]
        d["t"] = pd.to_datetime(np.where(d["t"] > 1e14, d["t"] // 1000, d["t"]), unit="ms")
        parts.append(d.set_index("t"))
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    df.to_parquet(CACHE)
    return df


DOLLAR_LAG = pd.Timedelta(days=9)
_usd = None


def dollar():
    """FRED broad dollar index (daily values, published weekly); cached for a day."""
    global _usd
    if _usd is None:
        path = os.path.join(ROOT, "cache", "macro", "fred_DTWEXBGS.csv")
        if not os.path.exists(path) or (datetime.now().timestamp() - os.path.getmtime(path)) > 86400:
            try:
                r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS", timeout=30)
                r.raise_for_status()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                open(path, "w").write(r.text)
            except Exception:
                pass
        if os.path.exists(path):
            s = pd.read_csv(path, na_values=".")
            s.columns = ["date", "v"]
            _usd = s.dropna().assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")["v"]
        else:
            _usd = pd.Series(dtype=float)
    return _usd


def side_at(h, t, usd=None):
    """Rule direction at time t: majority of trend20, trend50 and the 20-day dollar change."""
    daily = h["close"][h.index < t.normalize()].resample("1D").last().dropna()
    px = h["close"][h.index <= t].iloc[-1]
    sma20, sma50 = daily.iloc[-20:].mean(), daily.iloc[-50:].mean()
    votes = {"trend20": 1 if px > sma20 else -1}
    if len(daily) >= 50:
        votes["trend50"] = 1 if px > sma50 else -1
    if usd is not None:
        u = usd[usd.index <= t - DOLLAR_LAG]
        if len(u) > 21:
            votes["dollar"] = -1 if u.iloc[-1] > u.iloc[-21] else 1
    total = sum(votes.values())
    side = int(np.sign(total)) if total != 0 else votes["trend20"]
    return side, px, sma20, dict(votes, sma50=sma50, usd_chg=(u.iloc[-1] / u.iloc[-21] - 1) if usd is not None and len(u) > 21 else None)


def simulate(h, t0, side, entry):
    """Outcome of the rule trade from t0 using hourly highs/lows; same-hour stop and target = stop."""
    f = h[h.index > t0].iloc[:HOLD_H]
    for ts, b in f.iterrows():
        adv = (b["low"] / entry - 1) if side > 0 else (1 - b["high"] / entry)
        fav = (b["high"] / entry - 1) if side > 0 else (1 - b["low"] / entry)
        if adv <= -SL:
            return "stop", -SL, ts
        if fav >= TP:
            return "target", TP, ts
    if len(f) >= HOLD_H:
        return "7 days", side * (f["close"].iloc[-1] / entry - 1), f.index[-1]
    return "open", side * (f["close"].iloc[-1] / entry - 1) if len(f) else 0.0, None


def history_stats(h):
    """Non-overlapping replay of the rule (decisions at 00:00 UTC) on the full history."""
    t = h.index[0].normalize() + pd.Timedelta(days=51)
    res = []
    while t < h.index[-1] - pd.Timedelta(hours=HOLD_H):
        side, px, _, _ = side_at(h, t, dollar())
        out, r, end = simulate(h, t, side, px)
        res.append(r - FEE)
        t = (end + pd.Timedelta(hours=1)).ceil("D") if end is not None else t + pd.Timedelta(days=1)
    R = np.array(res) / SL
    return dict(n=len(R), win=float(np.mean(R > 0)), avg_R=float(R.mean()))


def vol_calibration(h):
    """How often the next 24h close-to-close move stayed inside the +-1 sigma forecast."""
    lr = np.log(h["close"]).diff()
    sig = lr.rolling(168).std() * np.sqrt(24)
    fwd = np.log(h["close"].shift(-24) / h["close"])
    ok = sig.notna() & fwd.notna()
    return float((fwd[ok].abs() <= sig[ok]).mean())


def score_log(h):
    if not os.path.exists(LOG):
        return None
    log = pd.read_csv(LOG, parse_dates=["time"])
    rows = []
    for _, s in log.drop_duplicates("day").iterrows():
        out, r, _ = simulate(h, s["time"], int(s["side"]), float(s["price"]))
        rows.append(dict(day=s["day"], side=s["side"], out=out, R=(r - FEE) / SL))
    return pd.DataFrame(rows)


def main(risk_usd=1000.0):
    h, src, base = recent()
    try:
        full = pd.concat([archive(), h]).sort_index()
        full = full[~full.index.duplicated(keep="last")]
    except Exception:
        full = h
    now = h.index[-1] + pd.Timedelta(hours=1)
    side, px, sma, v = side_at(full, now, dollar())
    lr = np.log(full["close"]).diff()
    sig_d = float(lr.iloc[-168:].std() * np.sqrt(24))
    sig_w = sig_d * np.sqrt(7)
    stop = px * (1 - side * SL)
    target = px * (1 + side * TP)
    notional = risk_usd / (SL + FEE)
    qty = notional / px
    liq = px * (1 - side * (1 / LEV - 0.005))
    hs = history_stats(full)
    cal = vol_calibration(full)
    fr = funding(base)
    d20 = full["close"].resample("1D").last().dropna()

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    day = str(now.date())
    new = pd.DataFrame([dict(day=day, time=now, side=side, price=round(px, 2), sma20=round(sma, 2),
                             stop=round(stop, 2), target=round(target, 2), sigma_day=round(sig_d, 5), source=src)])
    # one trade at a time, exactly like the backtest: while the last signal is still open, hold it
    active = None
    if os.path.exists(LOG):
        last = pd.read_csv(LOG, parse_dates=["time"]).iloc[-1]
        out, r, _ = simulate(full, last["time"], int(last["side"]), float(last["price"]))
        if out == "open":
            active = dict(last, result=r)
    if active is None and (not os.path.exists(LOG) or day not in set(pd.read_csv(LOG)["day"].astype(str))):
        new.to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    live = score_log(full)

    word = "ЛОНГ" if side > 0 else "ШОРТ"
    lines = [
        f"XAUUSDT — {now:%Y-%m-%d %H:%M} UTC (источник цен: {src})",
        "",
        f"РЕШЕНИЕ: {word}",
        f"  Вход:    {px:,.2f}",
        f"  Стоп:    {stop:,.2f}  ({'-' if side > 0 else '+'}{SL:.0%})",
        f"  Тейк:    {target:,.2f}  ({'+' if side > 0 else '-'}{TP:.0%})",
        "  Срок:    максимум 7 дней, потом закрыть по рынку",
        f"  Размер:  {notional:,.0f} $ позиции = {qty:.3f} XAU, маржа {notional / LEV:,.0f} $ при {LEV}x",
        f"  Итог:    стоп ≈ −{risk_usd:,.0f} $ · тейк ≈ +{notional * (TP - FEE):,.0f} $",
        f"  Ликвидация {LEV}x ≈ {liq:,.2f} (стоп сработает раньше)",
        "",
        f"ВХОД ЛИМИТКОЙ (лучше по истории): {px * (1 - side * LIMIT_BETTER):,.2f} — на {LIMIT_BETTER:.1%} "
        f"{'выше' if side < 0 else 'ниже'} текущей цены, ордер живёт 24 ч; не исполнился → сделки нет.",
        f"  Тогда стоп {px * (1 - side * LIMIT_BETTER) * (1 - side * SL):,.2f}, тейк {px * (1 - side * LIMIT_BETTER) * (1 + side * TP):,.2f}. "
        f"Исполняется ~7 раз из 10.",
        "",
        f"ПОЧЕМУ {word} (2 из 3 голосов решают):",
        f"  Тренд 20 дней:  {'ЛОНГ' if v['trend20'] > 0 else 'ШОРТ'} (цена {'выше' if v['trend20'] > 0 else 'ниже'} средней {sma:,.0f})",
        (f"  Тренд 50 дней:  {'ЛОНГ' if v['trend50'] > 0 else 'ШОРТ'} (средняя {v['sma50']:,.0f})" if "trend50" in v else "  Тренд 50 дней:  нет данных"),
        (f"  Доллар 20 дней: {'ЛОНГ' if v['dollar'] > 0 else 'ШОРТ'} (индекс доллара {v['usd_chg']:+.1%}: "
         f"{'дешевеет → золото вверх' if v['dollar'] > 0 else 'дорожает → золото вниз'})" if "dollar" in v else "  Доллар: нет данных"),
        f"  Последние 5 закрытий дня: {', '.join(f'{x:,.0f}' for x in d20.iloc[-6:-1])}",
        "",
        "ВОЛАТИЛЬНОСТЬ (из движения последних 7 дней):",
        f"  За 1 день обычно ±{sig_d:.1%} (±{px * sig_d:,.0f} $ цены): от {px * (1 - sig_d):,.0f} до {px * (1 + sig_d):,.0f}",
        f"  За 7 дней обычно ±{sig_w:.1%}: от {px * (1 - sig_w):,.0f} до {px * (1 + sig_w):,.0f}; "
        f"в 1 из 20 недель больше ±{2 * sig_w:.1%}",
        f"  Проверка прогноза на истории: дневное движение укладывалось в ±1σ в {cal:.0%} дней (норма 68%)",
        f"  Фандинг сейчас: {fr * 100:+.4f}% за 8ч ({'лонги платят шортам' if fr > 0 else 'шорты платят лонгам'})"
        if fr == fr and src == "live" else "",
        "",
        f"ИСТОРИЯ ПРАВИЛА (с {full.index[0]:%Y-%m}, без перекрытий): {hs['n']} сделок, "
        f"прибыльных {hs['win']:.0%}, в среднем {hs['avg_R']:+.2f} от суммы риска на сделку",
    ]
    if live is not None and len(live):
        closed = live[live["out"] != "open"]
        lines.append(f"ЖИВОЙ ЖУРНАЛ: {len(live)} сигналов, закрыто {len(closed)}, "
                     + (f"прибыльных {np.mean(closed['R'] > 0):.0%}, сумма {closed['R'].sum():+.1f} от риска"
                        if len(closed) else "результатов ещё нет"))
    if active is not None:
        lines += ["", f"БУМАЖНЫЙ ЖУРНАЛ: сделка правила от {pd.Timestamp(active['time']):%d.%m %H:%M} UTC "
                  f"({'ЛОНГ' if int(active['side']) > 0 else 'ШОРТ'} {float(active['price']):,.2f}, стоп {float(active['stop']):,.2f}, "
                  f"тейк {float(active['target']):,.2f}) ещё открыта: {active['result']:+.2%}. "
                  "Если вы в ней, держите её уровни и не входите второй раз."]
    lines += ["", "Правило механическое и не доказано статистически: ~6 из 10 сделок закрываются стопом, "
              "плюс даёт то, что тейк вдвое больше стопа. Сверьте цену с Binance перед входом."]
    text = "\n".join(x for x in lines if x is not None)
    print(text)
    return text


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    report = main(float(args[0]) if args else float(os.environ.get("XAU_RISK_USD", 1000)))
    if "--notify" in sys.argv:
        from tk import notify
        notify.send(report.split("\n")[2].replace("РЕШЕНИЕ: ", "🥇 XAUUSDT: "), report)
