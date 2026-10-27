import numpy as np
import pandas as pd

from tk import xau


def frame(closes, start="2026-01-01"):
    t = pd.date_range(start, periods=len(closes), freq="h")
    c = np.array(closes, float)
    return pd.DataFrame(dict(open=c, high=c * 1.001, low=c * 0.999, close=c, volume=1.0), index=t)


def test_side_follows_20_day_average():
    up = frame(np.linspace(100, 130, 24 * 25))
    assert xau.side_at(up, up.index[-1])[0] == 1
    down = frame(np.linspace(130, 100, 24 * 25))
    assert xau.side_at(down, down.index[-1])[0] == -1


def test_long_hits_target_before_time_limit():
    h = frame([100.0] * 5 + [105.0] * 5)
    out, r, _ = xau.simulate(h, h.index[4], 1, 100.0)
    assert out == "target" and r == xau.TP


def test_short_stopped():
    h = frame([100.0] * 5 + [103.0] * 5)
    out, r, _ = xau.simulate(h, h.index[4], -1, 100.0)
    assert out == "stop" and r == -xau.SL
