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
    assert body.index("ПРОШЛЫЙ ЧАС") < body.index("ОБУЧЕНИЕ") < body.index("АНАЛИЗ НОВОЙ МОДЕЛЬЮ") < body.index("ПРОГНОЗ НА") < body.index("СЧЁТ")
    assert "До обучения было" in body and "Раскладка по всем входам" in body
