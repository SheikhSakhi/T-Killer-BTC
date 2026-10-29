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
