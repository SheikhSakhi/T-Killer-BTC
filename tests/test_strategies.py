import numpy as np
import pandas as pd
import pytest

from tk import strategies as st


def path(highs, lows, opens=None, closes=None, start="2026-01-01 07:00"):
    n = len(highs)
    t = pd.date_range(start, periods=n, freq="5min").values
    opens = opens or [100.0] * n
    closes = closes or [100.0] * n
    return dict(open=np.array(opens, float), high=np.array(highs, float), low=np.array(lows, float),
                close=np.array(closes, float), time=t)


def test_long_hits_target():
    p = path([100, 101, 103], [99.5, 99.5, 100])
    R, k = st._run(p, 0, 1, 100.0, 99.0, 102.0, 0.0)
    assert k == 2
    assert R == pytest.approx((0.02 - st.COST) / 0.01)


def test_same_bar_stop_and_target_counts_as_stop():
    p = path([100, 103], [100, 98])
    R, _ = st._run(p, 0, 1, 100.0, 99.0, 102.0, 0.0)
    assert R == pytest.approx((-0.01 - st.COST) / 0.01)


def test_short_stop():
    p = path([100, 101.5], [99.5, 99.8])
    R, _ = st._run(p, 0, -1, 100.0, 101.0, 98.0, 0.0)
    assert R == pytest.approx((-0.01 - st.COST) / 0.01)


def test_time_exit_at_last_close():
    p = path([100.5] * 4, [99.6] * 4, closes=[100, 100, 100, 100.5])
    R, k = st._run(p, 0, 1, 100.0, 99.0, 102.0, 0.0)
    assert k == 3
    assert R == pytest.approx((0.005 - st.COST) / 0.01)


def test_funding_charged_when_crossing_settlement():
    p = path([100.5] * 3, [99.6] * 3, start="2026-01-01 07:55")  # crosses 08:00
    R0, _ = st._run(p, 0, 1, 100.0, 99.0, 102.0, 0.0)
    R1, _ = st._run(p, 0, 1, 100.0, 99.0, 102.0, 0.001)
    assert R0 - R1 == pytest.approx(0.001 / 0.01)


def test_stop_too_tight_is_rejected():
    p = path([100] * 3, [100] * 3)
    assert st._run(p, 0, 1, 100.0, 99.9, 100.2, 0.0) is None


def test_breakout_skips_when_both_sides_break_in_one_bar():
    p = path([100, 102], [100, 98])
    assert st._breakout(p, 0, 101, 99, None, dict(tp_r=1.0), "range") is None


def test_grid_ids_unique():
    ids = [g["id"] for g in st.grid()]
    assert len(ids) == len(set(ids))


def test_plan_entry_matches_backtest_entry():
    row = dict(session="europe", squeeze=0.5, funding=0.0)
    prm = dict(family="orb", session="europe", or_min=30, tp_r=2.0, squeeze_max=None)
    highs = [100.5] * 6 + [100.6, 101.2, 102.0, 103.5]
    lows = [99.5] * 6 + [100.0, 100.4, 100.9, 101.5]
    p = path(highs, lows, opens=[100] * 10)
    e = st.plan_entry(prm, row, p)
    assert e["side"] == 1 and e["i"] == 6 and e["entry"] == 100.5 and e["sl"] == 99.5
    # a live path that ends on the trigger bar plans the same entry
    live = {k: v[:7] for k, v in p.items()}
    assert st.plan_entry(prm, row, live) == e
