"""Hourly XAUUSDT learner: forecast the next hour, check it an hour later, learn, email.

Runs on the server once an hour (a minute after the hour closes):
  1. update the hourly raw data from the live Binance API (archive history is the starting point)
  2. score the forecast made an hour ago against what happened
  3. LEARN: one online-logistic update on the new labelled hour plus a replay of the last 7 days
  4. forecast the next hour and email both parts

Proof of learning, not storytelling: a FROZEN copy of the model from the start forecasts the same
hours. Only if the learning model beats the frozen one (and a coin) on the same hours did it learn.
The email shows which inputs changed weight and by how much.

Features (identical code for history and live, nothing from after the hour's close):
  taker buy share 1/4/24h, volume vs 7-day norm, open-interest change 1/4/24h, top-trader and global
  long/short ratios (z vs 7 days), taker long/short volume ratio, perpetual premium (1h, 24h),
  returns 1/4/24h, 24h volatility.
Order-book depth is logged live (data/hourly/book.csv) and joins the model once enough hours exist.

python -m tk.hourly            # one hourly cycle (server)
python -m tk.hourly --replay 500   # replay the cycle on the last 500 archive hours (testing)
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DIR = os.path.join(ROOT, "data", "hourly")
RAW = os.path.join(ROOT, "cache", "xau_hourly_raw.parquet")
MODEL = os.path.join(DIR, "model.json")
LOG = os.path.join(DIR, "log.csv")
SYM = "XAUUSDT"
LIVE = "https://fapi.binance.com"
L2, LR, EPOCHS, REPLAY_H = float(os.environ.get("HOURLY_L2", 1.0)), 0.05, 3, 168

NAMES = {
    "tb1": "доля агрессивных покупок за 1 ч", "tb4": "доля агрессивных покупок за 4 ч",
    "tb24": "доля агрессивных покупок за 24 ч", "vol_z": "объём относительно нормы",
    "oi1": "изменение открытого интереса за 1 ч", "oi4": "изменение открытого интереса за 4 ч",
    "oi24": "изменение открытого интереса за 24 ч", "top_ls_z": "лонги/шорты топ-трейдеров",
    "glob_ls_z": "лонги/шорты всех счетов", "taker_ls4": "покупки/продажи агрессоров за 4 ч",
    "prem": "премия фьючерса к индексу", "prem24": "премия фьючерса за 24 ч",
    "ret1": "движение за прошлый час", "ret4": "движение за 4 ч", "ret24": "движение за 24 ч",
    "vol24": "волатильность за 24 ч",
}
FEATS = list(NAMES)
# Live features built from the minute collector (tk/collect.py). No archive history exists for them:
# they stay neutral (weight 0, input 0) until enough live hours exist, then the model learns their weights.
LIVE_NAMES = {
    "lf_flow": "живой поток: чистые агрессивные покупки за час", "lf_flow15": "живой поток за последние 15 мин",
    "lf_big": "крупные сделки (≥50k$): чистые покупки", "lf_book5": "стакан, 5 ближних уровней: перевес покупок",
    "lf_book5_chg": "стакан, 5 уровней: изменение перевеса за час", "lf_liq": "ликвидации: шортов минус лонгов",
    "lf_spread": "спред", "lf_flow_corr": "насколько поток двигал цену в этот час",
}
LIVE_FEATS = list(LIVE_NAMES)
NAMES.update(LIVE_NAMES)
ALL = FEATS + LIVE_FEATS
MIN_DIR = os.path.join(ROOT, "cache", "minutes")
LIVE_READY = 48     # live hours needed before live inputs are scaled and start to count


# ------------------------------------------------------------------ data

def zs(s, n=168):
    return (s - s.rolling(n, min_periods=n // 2).mean()) / s.rolling(n, min_periods=n // 2).std()


def features(R):
    """R: hourly raw frame indexed by bar open. Row t uses only data known at t + 1h."""
    X = pd.DataFrame(index=R.index)
    qv, tb = R["quote_volume"], R["taker_buy_quote"]
    for n in (1, 4, 24):
        X[f"tb{n}"] = tb.rolling(n).sum() / qv.rolling(n).sum() - 0.5
    X["vol_z"] = zs(np.log(qv))
    oi = R["oi_value"].where(R["oi_value"] > 0)
    for n in (1, 4, 24):
        X[f"oi{n}"] = oi / oi.shift(n) - 1
    X["top_ls_z"] = zs(R["top_ls_pos"])
    X["glob_ls_z"] = zs(R["global_ls_acc"])
    X["taker_ls4"] = np.log(R["taker_ls_vol"]).rolling(4).mean()
    X["prem"] = R["premium"]
    X["prem24"] = R["premium"].rolling(24).mean()
    c = R["close"]
    X["ret1"], X["ret4"], X["ret24"] = c.pct_change(1), c.pct_change(4), c.pct_change(24)
    X["vol24"] = np.log(c).diff().rolling(24).std()
    return X.replace([np.inf, -np.inf], np.nan)


def load_minutes(start, end):
    """Minute rows from the live collector between start and end (UTC, naive)."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    parts = [pd.read_csv(os.path.join(MIN_DIR, f"{d.date()}.csv")) for d in days
             if os.path.exists(os.path.join(MIN_DIR, f"{d.date()}.csv"))]
    if not parts:
        return pd.DataFrame()
    m = pd.concat(parts)
    m["minute"] = pd.to_datetime(m["minute"])
    m = m.drop_duplicates("minute").set_index("minute").sort_index()
    return m[(m.index >= start) & (m.index < end)]


def _hour_live(mm):
    """Live features of one hour from its minute rows (needs 45+ minutes)."""
    if len(mm) < 45:
        return None
    tot = mm["buy_usd"] + mm["sell_usd"]
    net = mm["buy_usd"] - mm["sell_usd"]
    last15 = mm.tail(15)
    ret = mm["close"].pct_change()
    flow = net / tot.replace(0, np.nan)
    corr = ret.corr(flow) if ret.std() > 0 and flow.std() > 0 else 0.0
    return dict(lf_flow=net.sum() / tot.sum() if tot.sum() else 0.0,
                lf_flow15=(last15["buy_usd"] - last15["sell_usd"]).sum() / max((last15["buy_usd"] + last15["sell_usd"]).sum(), 1),
                lf_big=(mm["big_buy_usd"] - mm["big_sell_usd"]).sum() / max(tot.sum(), 1),
                lf_book5=mm["book5_imb"].mean(), lf_book5_chg=mm["book5_imb_end"].iloc[-1] - mm["book5_imb_end"].iloc[0],
                lf_liq=(mm["liq_short_usd"] - mm["liq_long_usd"]).sum() / max(tot.sum(), 1),
                lf_spread=(mm["spread"] / mm["close"]).mean(), lf_flow_corr=corr)


def live_features(index):
    """Hourly live features for the given bar-open times (NaN where the collector has no data)."""
    out = pd.DataFrame(np.nan, index=index, columns=LIVE_FEATS)
    if not os.path.isdir(MIN_DIR) or not len(index):
        return out
    mins = load_minutes(index[0], index[-1] + pd.Timedelta(hours=1))
    if mins.empty:
        return out
    for t, mm in mins.groupby(mins.index.floor("h")):
        if t in out.index:
            f = _hour_live(mm)
            if f:
                out.loc[t] = [f[c] for c in LIVE_FEATS]
    return out


def hour_drivers(t):
    """What actually moved the price inside hour bar t, minute by minute (live collector data)."""
    mm = load_minutes(t, t + pd.Timedelta(hours=1))
    if len(mm) < 30:
        return None
    ret = mm["close"].pct_change()
    tot = mm["buy_usd"] + mm["sell_usd"]
    series = {
        "агрессивные покупки минус продажи": (mm["buy_usd"] - mm["sell_usd"]) / tot.replace(0, np.nan),
        "крупные сделки (чистые покупки)": mm["big_buy_usd"] - mm["big_sell_usd"],
        "перевес стакана (5 уровней)": mm["book5_imb"],
        "изменение перевеса стакана": mm["book5_imb_end"].diff(),
        "изменение открытого интереса": mm["oi"].pct_change(),
        "ликвидации шортов минус лонгов": mm["liq_short_usd"] - mm["liq_long_usd"],
        "премия фьючерса": mm["premium"],
    }
    rows = []
    for name, x in series.items():
        same = ret.corr(x) if x.std() > 0 and ret.std() > 0 else np.nan
        lead = ret.corr(x.shift(1)) if x.std() > 0 and ret.std() > 0 else np.nan
        rows.append((name, same, lead))
    oi = mm["oi"].dropna()
    return dict(minutes=len(mm), rows=sorted(rows, key=lambda r: -abs(r[1]) if r[1] == r[1] else 0),
                net_usd=float((mm["buy_usd"] - mm["sell_usd"]).sum()), vol_usd=float(tot.sum()),
                big_net=float((mm["big_buy_usd"] - mm["big_sell_usd"]).sum()),
                liq_long=float(mm["liq_long_usd"].sum()), liq_short=float(mm["liq_short_usd"].sum()),
                book_start=float(mm["book5_imb_end"].iloc[0]), book_end=float(mm["book5_imb_end"].iloc[-1]),
                oi_chg=float(oi.iloc[-1] / oi.iloc[0] - 1) if len(oi) > 1 and oi.iloc[0] else np.nan,
                hi=float(mm["high"].max()), lo=float(mm["low"].min()))


def features_all(R):
    return features(R).join(live_features(R.index))


def build_raw_from_archive():
    """Starting history from data.binance.vision (see research/xau_flow_data.py)."""
    sys.path.insert(0, os.path.join(ROOT, "research"))
    import xau_flow as XF
    k = XF.archive_klines()
    m = pd.read_parquet(os.path.join(XF.F, "metrics.parquet")).resample("1h").last()
    p = pd.read_parquet(os.path.join(XF.F, "premium_1h.parquet"))["premium"]
    R = k[["close", "high", "low", "quote_volume", "taker_buy_quote"]].copy()
    R["oi_value"] = m["sum_open_interest_value"].reindex(R.index)
    R["top_ls_pos"] = m["sum_toptrader_long_short_ratio"].reindex(R.index)
    R["global_ls_acc"] = m["count_long_short_ratio"].reindex(R.index)
    R["taker_ls_vol"] = m["sum_taker_long_short_vol_ratio"].reindex(R.index)
    R["premium"] = p.reindex(R.index)
    return R


def _get(path, **params):
    r = requests.get(LIVE + path, params=dict(symbol=SYM, **params), timeout=20)
    r.raise_for_status()
    return r.json()


def _hourly_last(rows, key, field):
    s = pd.Series({pd.to_datetime(int(x[key]), unit="ms"): float(x[field]) for x in rows}).sort_index()
    return s.resample("1h").last()          # the last 5-minute value inside bar [hh, hh+1)


def update_live(R):
    """Append the latest closed hours from the live API (same definitions as the archive)."""
    k = _get("/fapi/v1/klines", interval="1h", limit=48)
    kl = pd.DataFrame([dict(t=pd.to_datetime(x[0], unit="ms"), close=float(x[4]), high=float(x[2]), low=float(x[3]),
                            quote_volume=float(x[7]), taker_buy_quote=float(x[10])) for x in k]).set_index("t")
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    kl = kl[kl.index + pd.Timedelta(hours=1) <= now]             # closed bars only
    oi = _hourly_last(_get("/futures/data/openInterestHist", period="5m", limit=500), "timestamp", "sumOpenInterestValue")
    top = _hourly_last(_get("/futures/data/topLongShortPositionRatio", period="5m", limit=500), "timestamp", "longShortRatio")
    glob = _hourly_last(_get("/futures/data/globalLongShortAccountRatio", period="5m", limit=500), "timestamp", "longShortRatio")
    tak = _hourly_last(_get("/futures/data/takerlongshortRatio", period="5m", limit=500), "timestamp", "buySellRatio")
    pk = _get("/fapi/v1/premiumIndexKlines", interval="1h", limit=48)
    prem = pd.Series({pd.to_datetime(x[0], unit="ms"): float(x[4]) for x in pk})
    new = kl.assign(oi_value=oi.reindex(kl.index), top_ls_pos=top.reindex(kl.index),
                    global_ls_acc=glob.reindex(kl.index), taker_ls_vol=tak.reindex(kl.index),
                    premium=prem.reindex(kl.index))
    out = pd.concat([R, new])
    return out[~out.index.duplicated(keep="last")].sort_index()


def log_book():
    """Live depth snapshot for future book features (the archive's ±% bands are not reproducible by REST)."""
    try:
        d = _get("/fapi/v1/depth", limit=1000)
        bids = np.array(d["bids"], float)
        asks = np.array(d["asks"], float)
        mid = (bids[0, 0] + asks[0, 0]) / 2
        row = dict(time=str(pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None).floor("min")), mid=mid,
                   spread=asks[0, 0] - bids[0, 0], bid_range=1 - bids[-1, 0] / mid, ask_range=asks[-1, 0] / mid - 1)
        for p in (0.001, 0.002, 0.005, 0.01):
            b = (bids[:, 0] * bids[:, 1])[bids[:, 0] >= mid * (1 - p)].sum()
            a = (asks[:, 0] * asks[:, 1])[asks[:, 0] <= mid * (1 + p)].sum()
            row[f"imb_{p:.3f}"] = (b - a) / (b + a) if b + a else np.nan
        path = os.path.join(DIR, "book.csv")
        pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    except Exception as e:
        print("book snapshot failed:", e)


# ------------------------------------------------------------------ model

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def fit(Z, y, w=None, b=0.0, steps=400, lr=0.1):
    w = np.zeros(Z.shape[1]) if w is None else w.copy()
    for _ in range(steps):
        p = sigmoid(Z @ w + b)
        g = p - y
        w -= lr * (Z.T @ g / len(y) + L2 * w)
        b -= lr * g.mean()
    return w, b


def usable(X):
    return X[FEATS].notna().mean(axis=1) >= 0.8


def init_model(X, y):
    X = X.reindex(columns=ALL)
    ok = y.notna() & usable(X)
    Xo, yo = X[ok], y[ok]
    mu, sd = Xo.mean().fillna(0.0), Xo.std().replace(0, 1).fillna(1.0)
    Z = ((Xo - mu) / sd).fillna(0).values
    w, b = fit(Z, yo.values)
    return dict(features=ALL, mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), b=b, w0=w.tolist(), b0=b,
                trained_until=str(Xo.index[-1]), updates=0, w_history=[], live_hours=0)


def scale_live(m, X):
    """Once enough live hours exist, scale live inputs by their own live history."""
    live = X[LIVE_FEATS].dropna(how="all")
    m["live_hours"] = int(len(live))
    if len(live) >= LIVE_READY:
        mu, sd = np.array(m["mu"]), np.array(m["sd"])
        k = len(FEATS)
        mu[k:] = live.mean().fillna(0).values
        sd[k:] = live.std().replace(0, 1).fillna(1).values
        m["mu"], m["sd"] = mu.tolist(), sd.tolist()


def zrow(m, x):
    """Standardised inputs; a missing input counts as neutral (0 = its historical average)."""
    z = np.nan_to_num((np.asarray(x, float) - np.array(m["mu"])) / np.array(m["sd"]), nan=0.0)
    if m.get("live_hours", 0) < LIVE_READY:
        z[len(FEATS):] = 0.0          # live inputs stay neutral until they have been scaled
    return z


def _window_stats(Z, yy, w, b):
    p = np.clip(sigmoid(Z @ w + b), 1e-6, 1 - 1e-6)
    return dict(logloss=float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p))),
                acc=float(np.mean((p >= 0.5) == (yy == 1))))


def learn(m, X, y, t):
    """Online update on the newly labelled hour t plus a replay of the previous 7 days.
    Returns what training did: window size, error and accuracy on the window before/after, weight changes."""
    w_before, b_before = np.array(m["w"]), m["b"]
    win = X.reindex(columns=ALL).loc[:t].tail(REPLAY_H)
    ok = y.reindex(win.index).notna() & usable(win)
    Z = np.array([zrow(m, r) for r in win[ok].values])
    yy = y.reindex(win.index)[ok].values
    if len(yy) < 24:
        return dict(n=len(yy), dw=np.zeros_like(w_before), skipped=True)
    before = _window_stats(Z, yy, w_before, b_before)
    w, b = fit(Z, yy, w_before, b_before, steps=EPOCHS * 10, lr=LR)
    after = _window_stats(Z, yy, w, b)
    m["w"], m["b"], m["updates"] = w.tolist(), float(b), m["updates"] + 1
    m["w_history"] = (m["w_history"] + [dict(t=str(t), w=[round(v, 4) for v in w])])[-24 * 30:]
    return dict(n=len(yy), up_share=float(yy.mean()), before=before, after=after, dw=w - w_before,
                w_before=w_before, b_before=b_before, skipped=False)


def predict(m, x, frozen=False):
    w = np.array(m["w0"] if frozen else m["w"])
    b = m["b0"] if frozen else m["b"]
    z = zrow(m, x)
    return float(sigmoid(z @ w + b)), z * w


# ------------------------------------------------------------------ report

def ci(hits):
    n = len(hits)
    if n == 0:
        return "нет данных"
    p = np.mean(hits)
    half = 1.96 * np.sqrt(p * (1 - p) / n) if n > 1 else 0.5
    return f"{p:.0%} ({int(np.sum(hits))}/{n}, 95%: {max(p - half, 0):.0%}–{min(p + half, 1):.0%})"


def describe(name, v, pct):
    """Human reading of an input value and where it sits vs the last 30 days."""
    if v != v:
        return "нет данных"
    fmt = {"tb1": lambda v: f"покупки {50 + v * 100:.0f}% объёма", "tb4": lambda v: f"покупки {50 + v * 100:.0f}%",
           "tb24": lambda v: f"покупки {50 + v * 100:.1f}%", "vol_z": lambda v: f"{v:+.1f}σ к норме",
           "oi1": lambda v: f"{v:+.2%}", "oi4": lambda v: f"{v:+.2%}", "oi24": lambda v: f"{v:+.2%}",
           "top_ls_z": lambda v: f"{v:+.1f}σ", "glob_ls_z": lambda v: f"{v:+.1f}σ",
           "taker_ls4": lambda v: f"покупки/продажи {np.exp(v):.2f}", "prem": lambda v: f"{v * 1e4:+.1f} б.п.",
           "prem24": lambda v: f"{v * 1e4:+.1f} б.п.", "ret1": lambda v: f"{v:+.2%}", "ret4": lambda v: f"{v:+.2%}",
           "ret24": lambda v: f"{v:+.2%}", "vol24": lambda v: f"{v:.2%} в час",
           "lf_flow": lambda v: f"{v:+.1%} объёма", "lf_flow15": lambda v: f"{v:+.1%} объёма",
           "lf_big": lambda v: f"{v:+.1%} объёма", "lf_book5": lambda v: f"{v:+.2f}", "lf_book5_chg": lambda v: f"{v:+.2f}",
           "lf_liq": lambda v: f"{v:+.2%} объёма", "lf_spread": lambda v: f"{v * 1e4:.2f} б.п.",
           "lf_flow_corr": lambda v: f"корреляция {v:+.2f}"}[name](v)
    if pct != pct:
        return fmt
    tag = " — необычно высоко" if pct >= 0.85 else " — необычно низко" if pct <= 0.15 else ""
    return f"{fmt} (выше {pct:.0%} часов за 30 дн){tag}"


def usd(x):
    return f"{x / 1e6:+.2f} млн $" if abs(x) >= 1e6 else f"{x / 1e3:+.0f} тыс $"


def report(log, m, L_info, x_next, pct_next, t_next, close_now, vol1h, prev_info, drivers):
    """Email: forecast for the next hour first, then what was found in the last hour and what the model learned."""
    p, contrib = predict(m, x_next)
    pf, _ = predict(m, x_next, frozen=True)
    up = p >= 0.5
    trained = L_info is not None and not L_info.get("skipped")
    p_draft = float(sigmoid(zrow(m, x_next) @ L_info["w_before"] + L_info["b_before"])) if trained else None
    done = log.dropna(subset=["hit"])
    word = lambda q: "ВВЕРХ" if q >= 0.5 else "ВНИЗ"
    conf = lambda q: f"{max(q, 1 - q):.1%}"

    L = [f"ИТОГ: на {t_next:%H:00}–{t_next + pd.Timedelta(hours=1):%H:00} UTC {word(p)} {conf(p)}"
         + (f" (до обучения на прошлом часе было {word(p_draft)} {conf(p_draft)})" if p_draft is not None else ""), ""]

    # 1. forecast
    L += [f"1. АНАЛИЗ НА СЛЕДУЮЩИЙ ЧАС {t_next:%H:00}–{t_next + pd.Timedelta(hours=1):%H:00} UTC",
          f"   Прогноз: {word(p)} {conf(p)}"]
    if p_draft is not None:
        L.append(f"   Старой моделью (до обучения): {word(p_draft)} {conf(p_draft)} → новое знание сдвинуло вероятность роста на "
                 f"{(p - p_draft) * 100:+.2f} п.п." + (" и ПЕРЕВЕРНУЛО направление" if (p_draft >= 0.5) != up else ""))
    L.append(f"   Без всякого обучения (копия со дня запуска): {word(pf)} {conf(pf)}")
    L.append(f"   Цена {close_now:,.2f}; обычный ход за час ±{vol1h:.2%} → {close_now * (1 - vol1h):,.2f}–{close_now * (1 + vol1h):,.2f}")
    L.append("   Из чего сложился прогноз (вклад = значение × вес; плюс — за рост, минус — за падение):")
    live_on = m.get("live_hours", 0) >= LIVE_READY
    for i in np.argsort(-np.abs(contrib)):
        f = ALL[i]
        if f in LIVE_FEATS and not live_on:
            continue
        L.append(f"     {contrib[i]:+.3f}  {NAMES[f]}: {describe(f, x_next[i], pct_next[i])}")
    L.append(f"     ─────\n     {contrib.sum() + m['b']:+.3f}  итог → вероятность роста {p:.1%}")
    if not live_on:
        L.append(f"   Живые минутные данные копятся: {m.get('live_hours', 0)} из {LIVE_READY} часов, после этого они войдут в прогноз.")
    L.append("")

    # 2. what was found in the last hour
    if prev_info is not None:
        r = done.iloc[-1]
        h0 = pd.Timestamp(r["hour"])
        L += [f"2. ЧТО ОБНАРУЖЕНО ЗА ПРОШЛЫЙ ЧАС {h0:%H:00}–{h0 + pd.Timedelta(hours=1):%H:00} UTC",
              f"   Прогноз был {word(r['p_up'])} {conf(r['p_up'])}, факт {r['move']:+.2%} "
              f"({r['close_from']:,.2f} → {r['close_to']:,.2f}) → {'✅ верно' if r['hit'] else '❌ неверно'}"]
        if drivers:
            d = drivers
            L += [f"   Что реально двигало цену (живые данные, {d['minutes']} минут; корреляция минутных движений цены с каждым потоком):"]
            for name, same, lead in d["rows"]:
                if same == same:
                    L.append(f"     {name}: {same:+.2f} в ту же минуту, {lead:+.2f} с опережением на минуту")
            L += [f"   Итоги часа: агрессивные покупки минус продажи {usd(d['net_usd'])} из {d['vol_usd'] / 1e6:.1f} млн $ оборота; "
                  f"крупные сделки {usd(d['big_net'])}; ликвидации лонгов {d['liq_long'] / 1e3:.0f} тыс $, шортов {d['liq_short'] / 1e3:.0f} тыс $; "
                  f"перевес стакана {d['book_start']:+.2f} → {d['book_end']:+.2f}"
                  + (f"; открытый интерес {d['oi_chg']:+.2%}" if d['oi_chg'] == d['oi_chg'] else "")
                  + f"; диапазон {d['lo']:,.2f}–{d['hi']:,.2f}"]
        else:
            L.append("   Живых минутных данных за этот час нет (сборщик не работал), разбор только по часовым данным.")
        c_prev = prev_info["contrib"]
        actual = 1 if r["move"] > 0 else -1
        order = np.argsort(-np.abs(c_prev))
        good = [i for i in order if np.sign(c_prev[i]) == actual and abs(c_prev[i]) > 1e-6][:4]
        bad = [i for i in order if np.sign(c_prev[i]) == -actual and abs(c_prev[i]) > 1e-6][:4]
        L.append("   Метрики модели, указавшие верно: " + ("; ".join(f"{NAMES[ALL[i]]} ({c_prev[i]:+.3f})" for i in good) or "—"))
        L.append("   Метрики модели, ошибшиеся: " + ("; ".join(f"{NAMES[ALL[i]]} ({c_prev[i]:+.3f})" for i in bad) or "—"))
        L.append("")

    # 3. what the model learned from it
    if trained:
        bf, af, dw = L_info["before"], L_info["after"], L_info["dw"]
        w_now = np.array(m["w"])
        L += ["3. ЧЕМУ МОДЕЛЬ НАУЧИЛАСЬ (обучение прошло до отправки письма)",
              f"   Обучена на прошлом часе + повторе последних {L_info['n']} часов",
              f"   Ошибка модели на этих часах (log-loss): {bf['logloss']:.4f} → {af['logloss']:.4f}; "
              f"точность на них: {bf['acc']:.1%} → {af['acc']:.1%}",
              "   Сильнее всего изменились веса (влияние на «вверх»):"]
        for i in np.argsort(-np.abs(dw))[:5]:
            L.append(f"     {NAMES[ALL[i]]}: {w_now[i] - dw[i]:+.4f} → {w_now[i]:+.4f}")
        tot = w_now - np.array(m["w0"])
        L.append("   За всё время обучения: " + "; ".join(f"{NAMES[ALL[i]]} {m['w0'][i]:+.3f}→{w_now[i]:+.3f}"
                                                     for i in np.argsort(-np.abs(tot))[:4]))
        L.append(f"   Обновлений модели: {m['updates']}")
        L.append("")

    # 4. score
    if len(done):
        L.append(f"4. СЧЁТ: 24 ч {ci(done.tail(24)['hit'].astype(bool))} · всё время {ci(done['hit'].astype(bool))} · "
                 f"без обучения {ci(done['hit_frozen'].astype(bool))} · «против прошлого часа» {ci(done['hit_base'].astype(bool))}")
        diff = done["hit"].astype(float) - done["hit_frozen"].astype(float)
        if len(done) >= 24 and diff.std() > 0:
            se = diff.std() / np.sqrt(len(diff))
            L.append(f"   Обучение против копии без обучения: {diff.mean() * 100:+.1f} п.п. "
                     f"(95%: {(diff.mean() - 1.96 * se) * 100:+.1f}…{(diff.mean() + 1.96 * se) * 100:+.1f})")
    subj = f"XAU {t_next:%H:00} UTC: {word(p)} {conf(p)}"
    if len(done):
        subj += f" | прошлый час {'✅' if done.iloc[-1]['hit'] else '❌'} | 24ч {done.tail(24)['hit'].astype(bool).mean():.0%}"
    return subj, "\n".join(L), p, pf


# ------------------------------------------------------------------ cycle

def cycle(R, m, log, t_last, email=True):
    """t_last: the hour bar that just closed. Score the forecast for it, find what moved the price,
    learn, then forecast t_last + 1h with the updated model and send everything in one email."""
    X = features_all(R)
    scale_live(m, X)
    up = (R["close"].shift(-1) > R["close"]).astype(float).where(R["close"].shift(-1).notna())
    L_info, prev_info, drivers = None, None, None
    pend = log["hit"].isna() & (pd.to_datetime(log["hour"]) == t_last) if len(log) else pd.Series(dtype=bool)
    if len(log) and pend.any():
        i = log.index[pend][0]
        prev = t_last - pd.Timedelta(hours=1)
        c0, c1 = R.loc[prev, "close"], R.loc[t_last, "close"]
        move = c1 / c0 - 1
        log.loc[i, ["close_from", "close_to", "move"]] = [c0, c1, move]
        log.loc[i, "hit"] = bool((log.loc[i, "p_up"] >= 0.5) == (move > 0))
        log.loc[i, "hit_frozen"] = bool((log.loc[i, "p_frozen"] >= 0.5) == (move > 0))
        log.loc[i, "hit_base"] = bool((log.loc[i, "base_up"] == 1) == (move > 0))
        _, c_prev = predict(m, X.loc[prev].values)          # the inputs behind last hour's forecast, old weights
        prev_info = dict(contrib=c_prev)
        drivers = hour_drivers(t_last)                        # minute-by-minute: what moved the price in that hour
        L_info = learn(m, X, up, prev)                         # the outcome of `prev`'s forecast is now known
    x = X.loc[t_last]
    missing = [NAMES[c] for c in FEATS if x[c] != x[c]]
    if len(missing) > 4:
        raise RuntimeError(f"too many missing inputs at {t_last}: {missing}")
    pct = X.loc[:t_last].tail(720).rank(pct=True).iloc[-1].values
    t_next = t_last + pd.Timedelta(hours=1)
    vol1h = float(np.log(R["close"]).diff().tail(168).std())
    subj, body, p, pf = report(log, m, L_info, x.values, pct, t_next, R.loc[t_last, "close"], vol1h, prev_info, drivers)
    if missing:
        body += "\nНет данных в этот час (учтены как нейтральные): " + ", ".join(missing)
    base_up = 1 if R["close"].iloc[-1] < R["close"].iloc[-2] else 0     # opposite of the last hour
    log = pd.concat([log, pd.DataFrame([dict(hour=str(t_next), p_up=round(p, 4), p_frozen=round(pf, 4), base_up=base_up,
                                             close_from=np.nan, close_to=np.nan, move=np.nan, hit=np.nan,
                                             hit_frozen=np.nan, hit_base=np.nan)])], ignore_index=True)
    if email:
        from tk import notify
        notify.email(subj, body)
    return log, subj, body


def load():
    os.makedirs(DIR, exist_ok=True)
    R = pd.read_parquet(RAW) if os.path.exists(RAW) else build_raw_from_archive()
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(
        columns=["hour", "p_up", "p_frozen", "base_up", "close_from", "close_to", "move", "hit", "hit_frozen", "hit_base"])
    m = json.load(open(MODEL)) if os.path.exists(MODEL) else None
    if m is None or m.get("features") != ALL:
        X = features_all(R)
        y = (R["close"].shift(-1) > R["close"]).astype(float).where(R["close"].shift(-1).notna())
        m = init_model(X, y)
    return R, m, log


def save(R, m, log):
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    R.to_parquet(RAW)
    json.dump(m, open(MODEL, "w"))
    log.to_csv(LOG, index=False)


def main():
    if "--replay" in sys.argv:
        n = int(sys.argv[sys.argv.index("--replay") + 1])
        R = build_raw_from_archive()
        cut = R.index[-n - 1]
        X = features_all(R.loc[:cut])
        y = (R["close"].shift(-1) > R["close"]).astype(float).loc[:cut].iloc[:-1]
        m = init_model(X.iloc[:-1], y)
        log = pd.DataFrame(columns=["hour", "p_up", "p_frozen", "base_up", "close_from", "close_to", "move", "hit",
                                    "hit_frozen", "hit_base"])
        for t in R.index[-n - 1:-1]:
            log, subj, body = cycle(R.loc[:t], m, log, t, email=False)
        print(subj)
        print(body)
        d = log.dropna(subset=["hit"])
        print(f"\nREPLAY {len(d)} hours: learning {d['hit'].astype(bool).mean():.1%}, frozen {d['hit_frozen'].astype(bool).mean():.1%}, "
              f"opposite-last-hour {d['hit_base'].astype(bool).mean():.1%}")
        return
    R, m, log = load()
    R = update_live(R)
    t_last = R.index[-1]
    if len(log) and pd.to_datetime(log["hour"]).max() > t_last:
        print("already forecast", t_last + pd.Timedelta(hours=1))
        return
    log_book()
    log, subj, body = cycle(R, m, log, t_last)
    save(R, m, log)
    print(subj)
    print(body)


if __name__ == "__main__":
    main()
