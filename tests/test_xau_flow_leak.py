"""No-lookahead test for research/xau_flow.panel(): corrupting everything after time T must not
change any feature row whose bar closes at or before T."""
import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
FLOW = os.path.join(ROOT, "cache", "xauflow")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(FLOW, "book_1h.parquet")),
                                reason="order-flow cache not downloaded")


def load():
    spec = importlib.util.spec_from_file_location("xau_flow", os.path.join(ROOT, "research", "xau_flow.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_features_do_not_see_the_future(monkeypatch):
    XF = load()
    X0, _, _ = XF.panel()
    T = X0.index[len(X0) // 2]                       # features at row T may use data up to T + 1h

    orig_read, orig_k = pd.read_parquet, XF.archive_klines

    def corrupt(df):
        df = df.copy()
        cut = df.index > T + pd.Timedelta(hours=1)
        df.loc[cut] = df.loc[cut].values[::-1] * 1.37 if len(df.loc[cut]) else df.loc[cut]
        return df

    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **k: corrupt(orig_read(p, *a, **k)))
    monkeypatch.setattr(XF, "archive_klines", lambda: corrupt(orig_k()))
    X1, _, _ = XF.panel()
    a, b = X0.loc[:T], X1.loc[:T]
    diff = ~np.isclose(a.values, b.values, equal_nan=True)
    bad = a.columns[diff.any(axis=0)].tolist()
    assert not bad, f"features that changed when only the future changed: {bad}"
