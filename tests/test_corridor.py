import importlib.util
import os

import numpy as np

spec = importlib.util.spec_from_file_location(
    "corridor", os.path.join(os.path.dirname(__file__), "..", "research", "corridor.py"))
corridor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corridor)


def brute(hi, lo, width):
    out = []
    for t in range(len(hi)):
        a = 0
        for s in range(t, -1, -1):
            if hi[s:t + 1].max() / lo[s:t + 1].min() - 1 <= width:
                a = t - s + 1
            else:
                break
        out.append(a)
    return np.array(out)


def test_age_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(20):
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
        hi = c * (1 + rng.uniform(0, 0.05, 300))
        lo = c * (1 - rng.uniform(0, 0.05, 300))
        age, top, bot = corridor.corridor_age(hi, lo, 0.09)
        assert (age == brute(hi, lo, 0.09)).all()


def test_single_wide_bar_resets():
    hi = np.array([101, 101, 120, 101.0])
    lo = np.array([99, 99, 99, 100.0])
    age, _, _ = corridor.corridor_age(hi, lo, 0.09)
    assert list(age) == [1, 2, 0, 1]
