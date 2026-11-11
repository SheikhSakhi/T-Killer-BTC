import numpy as np
import pandas as pd

from tk import hourly as H


def raw(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01", periods=n, freq="h")
    c = 4000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    qv = rng.uniform(1e6, 2e6, n)
    return pd.DataFrame(dict(close=c, high=c * 1.001, low=c * 0.999, quote_volume=qv,
                             taker_buy_quote=qv * rng.uniform(0.4, 0.6, n),
                             oi_value=1e8 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                             top_ls_pos=rng.uniform(0.8, 1.2, n), global_ls_acc=rng.uniform(0.8, 1.2, n),
                             taker_ls_vol=rng.uniform(0.8, 1.2, n), premium=rng.normal(0, 1e-4, n)), index=t)


def test_features_do_not_see_the_future():
    R = raw()
    T = R.index[250]
    X0 = H.features(R)
    R2 = R.copy()
    R2.loc[R2.index > T] *= 1.7
    X1 = H.features(R2)
    assert np.allclose(X0.loc[:T].values, X1.loc[:T].values, equal_nan=True)


def test_learning_moves_live_weights_not_frozen():
    R = raw()
    X = H.features(R)
    y = (R["close"].shift(-1) > R["close"]).astype(float).where(R["close"].shift(-1).notna())
    m = H.init_model(X.iloc[:300], y.iloc[:300])
    w0 = list(m["w0"])
    info = H.learn(m, X, y, X.index[350])
    assert np.abs(info["dw"]).sum() > 0 and info["after"]["logloss"] <= info["before"]["logloss"] + 1e-9
    assert m["w0"] == w0 and m["updates"] == 1


def test_cycle_scores_previous_forecast():
    R = raw()
    X = H.features(R)
    y = (R["close"].shift(-1) > R["close"]).astype(float).where(R["close"].shift(-1).notna())
    m = H.init_model(X.iloc[:300], y.iloc[:300])
    log = pd.DataFrame(columns=["hour", "p_up", "p_frozen", "base_up", "close_from", "close_to", "move", "hit",
                                "hit_frozen", "hit_base"])
    log, _, _ = H.cycle(R.loc[:R.index[320]], m, log, R.index[320], email=False)
    log, subj, body = H.cycle(R.loc[:R.index[321]], m, log, R.index[321], email=False)
    first = log.iloc[0]
    move = R["close"].iloc[321] / R["close"].iloc[320] - 1
    assert bool(first["hit"]) == ((first["p_up"] >= 0.5) == (move > 0))
    assert body.startswith("ИТОГ:")
    assert (body.index("АНАЛИЗ НА СЛЕДУЮЩИЙ ЧАС") < body.index("ЧТО ОБНАРУЖЕНО ЗА ПРОШЛЫЙ ЧАС")
            < body.index("ЧЕМУ МОДЕЛЬ НАУЧИЛАСЬ") < body.index("СЧЁТ"))
    assert "Старой моделью (до обучения)" in body and "Из чего сложился прогноз" in body


def minutes_for(R, hours, path, rng):
    """Synthetic collector output where aggressive buying drives the minute price change."""
    import os
    rows = []
    for t in hours:
        px = R.loc[t, "close"]
        for k in range(60):
            flow = rng.normal(0, 1)
            px *= 1 + 0.0002 * flow + rng.normal(0, 0.00005)
            buy, sell = 1e6 * (1 + 0.3 * flow), 1e6 * (1 - 0.3 * flow)
            rows.append(dict(minute=(t + pd.Timedelta(minutes=k)).strftime("%Y-%m-%d %H:%M"), open=px, high=px * 1.0002,
                             low=px * 0.9998, close=px, buy_usd=buy, sell_usd=sell, trades=100, big_buy_usd=0.1 * buy,
                             big_sell_usd=0.1 * sell, book5_imb=0.1 * flow, book20_imb=0.05, book5_imb_end=0.1 * flow,
                             book20_imb_end=0.05, spread=0.1, liq_long_usd=0.0, liq_short_usd=0.0, premium=1e-4,
                             funding=1e-4, oi=1e8))
    df = pd.DataFrame(rows)
    df["day"] = df["minute"].str[:10]
    os.makedirs(path, exist_ok=True)
    for day, g in df.groupby("day"):
        g.drop(columns="day").to_csv(os.path.join(path, f"{day}.csv"), index=False)


def test_live_minutes_feed_the_model_and_the_hour_analysis(tmp_path, monkeypatch):
    R = raw(400)
    monkeypatch.setattr(H, "MIN_DIR", str(tmp_path))
    monkeypatch.setattr(H, "LIVE_READY", 10)
    minutes_for(R, R.index[300:330], str(tmp_path), np.random.default_rng(1))
    X = H.features_all(R)
    assert X.loc[R.index[310], H.LIVE_FEATS].notna().all() and X.loc[R.index[200], H.LIVE_FEATS].isna().all()
    d = H.hour_drivers(R.index[320])
    top = d["rows"][0]
    assert top[0] == "агрессивные покупки минус продажи" and top[1] > 0.8
    y = (R["close"].shift(-1) > R["close"]).astype(float).where(R["close"].shift(-1).notna())
    m = H.init_model(X.iloc[:300], y.iloc[:300])
    log = pd.DataFrame(columns=["hour", "p_up", "p_frozen", "base_up", "close_from", "close_to", "move", "hit",
                                "hit_frozen", "hit_base"])
    log, _, _ = H.cycle(R.loc[:R.index[320]], m, log, R.index[320], email=False)
    log, _, body = H.cycle(R.loc[:R.index[321]], m, log, R.index[321], email=False)
    assert m["live_hours"] >= 10
    assert "Что реально двигало цену" in body and "живой поток" in body


def test_update_live_survives_failed_endpoints_and_keeps_archive(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "ERRORS", str(tmp_path / "errors.log"))
    idx = pd.date_range("2026-01-01", periods=30, freq="1h")
    R = pd.DataFrame({"close": 1.0, "high": 1.0, "low": 1.0, "quote_volume": 1.0, "taker_buy_quote": 0.5,
                      "oi_value": 7.0, "top_ls_pos": 1.0, "global_ls_acc": 1.0, "taker_ls_vol": 1.0,
                      "premium": 0.0}, index=idx)
    ms = lambda t: int(t.value // 10**6)
    bars = [[ms(t), "1", "2", "0.5", "1.5", "0", 0, "10", 0, 0, "5"] for t in idx[-5:].append(
        pd.DatetimeIndex([idx[-1] + pd.Timedelta(hours=1)]))]

    def fake_get(path, **kw):
        if path == "/fapi/v1/klines":
            return bars
        raise RuntimeError("451")
    monkeypatch.setattr(H, "_get", fake_get)
    out = H.update_live(R)
    assert out.loc[idx[-1], "close"] == 1.5            # live price replaces the archive bar
    assert out.loc[idx[-1], "oi_value"] == 7.0         # archive OI is kept where the live call failed
    assert len(out) == 31 and out["oi_value"].isna().sum() == 1
    assert "openInterestHist" in open(H.ERRORS).read()


def test_email_failure_does_not_lose_the_hour(monkeypatch, tmp_path):
    from tk import notify
    monkeypatch.setattr(H, "ERRORS", str(tmp_path / "errors.log"))

    def boom(s, b):
        raise OSError("smtp down")
    monkeypatch.setattr(notify, "email", boom)
    R = raw()
    X = H.features_all(R)
    y = (R["close"].shift(-1) > R["close"]).astype(float).where(R["close"].shift(-1).notna())
    m = H.init_model(X.iloc[:-2], y.iloc[:-2])
    log, subj, body = H.cycle(R, m, pd.DataFrame(columns=["hour", "p_up", "p_frozen", "base_up", "close_from",
                                                         "close_to", "move", "hit", "hit_frozen", "hit_base"]),
                              R.index[-1], email=True)
    assert len(log) == 1 and "smtp down" in open(H.ERRORS).read()


def test_log_survives_a_round_trip_through_csv(tmp_path):
    """The server reloads log.csv every hour: an all-empty 'hit' column comes back as float."""
    R = raw()
    X = H.features(R)
    y = (R["close"].shift(-1) > R["close"]).astype(float).where(R["close"].shift(-1).notna())
    m = H.init_model(X.iloc[:300], y.iloc[:300])
    log = pd.DataFrame(columns=["hour", "p_up", "p_frozen", "base_up", "close_from", "close_to", "move", "hit",
                                "hit_frozen", "hit_base"])
    path = tmp_path / "log.csv"
    for t in R.index[320:324]:
        log, subj, body = H.cycle(R.loc[:t], m, log, t, email=False)
        log.to_csv(path, index=False)
        log = pd.read_csv(path)
    assert log["hit"].notna().sum() == 3 and "СЧЁТ" in body
