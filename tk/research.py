"""The time machine: replay every strategy variant on every coin, then test the daily selection
procedure itself walk-forward, so the reported result is what the live system would have done.

python -m tk.research trades      # simulate all variants -> cache/trades.parquet
python -m tk.research select      # walk-forward selection + null test -> results/research.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from tk import config, data, features, strategies

ROOT = os.path.join(os.path.dirname(__file__), "..")
TRADES = os.path.join(data.CACHE, "trades.parquet")


def _paths(sym):
    b = data.load("5m", sym)
    t = b.index.values
    arr = {k: b[k].values for k in ("open", "high", "low", "close")}
    return t, arr


def simulate_coin(sym):
    btc = data.load("1h", "BTCUSDT") if os.path.exists(data.path("1h", "BTCUSDT")) else None
    rows = features.build(sym, btc)
    t, arr = _paths(sym)
    grid = strategies.grid()
    by_session = {s: [p for p in grid if p["session"] == s] for s in config.SESSIONS}
    out = []
    for r in rows.to_dict("records"):
        start = r["start"]
        flat = pd.Timestamp(f"{start.date()} {config.FLAT_AT}")
        i0, i1 = np.searchsorted(t, np.datetime64(start)), np.searchsorted(t, np.datetime64(flat))
        if i1 - i0 < 12:
            continue
        path = {k: v[i0:i1] for k, v in arr.items()}
        path["time"] = t[i0:i1]
        for p in by_session[r["session"]]:
            res = strategies.trade(p, r, path)
            if res is not None:
                out.append((r["day"], r["session"], sym, p["id"], res["side"], res["R"], res["entry_time"]))
    df = pd.DataFrame(out, columns=["day", "session", "sym", "variant", "side", "R", "entry_time"])
    print(sym, len(df), "trades", flush=True)
    return df


def run_trades():
    with ProcessPoolExecutor(min(8, os.cpu_count() or 1)) as ex:
        parts = list(ex.map(simulate_coin, config.COINS))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(TRADES)
    return df


# ---------------------------------------------------------------- selection

def select_day(M, d, train=90, gap_val=30, t_req=2.5, min_train=15, min_val=5, min_coins=3):
    """Champion candidates for day index d using only rows < d.
    M: DataFrame days x (variant, sym) of R (NaN = no trade). Returns ranked list of columns."""
    tr = M.iloc[max(0, d - gap_val - train):d - gap_val]
    va = M.iloc[d - gap_val:d]
    n_tr, mu_tr, sd_tr = tr.count(), tr.mean(), tr.std()
    t_tr = mu_tr / (sd_tr / np.sqrt(n_tr))
    n_va, mu_va = va.count(), va.mean()
    ok = (n_tr >= min_train) & (t_tr >= t_req) & (n_va >= min_val) & (mu_va > 0)
    # robustness: the same variant must be positive in training on several coins
    pos = (mu_tr > 0).groupby(level="variant").sum()
    robust = M.columns.get_level_values("variant").map(pos) >= min_coins
    ok &= pd.Series(np.asarray(robust), index=M.columns)
    cand = t_tr[ok].sort_values(ascending=False)
    return list(cand.index)


def walk_forward(M, test_days=365, T=None, top_k=10, **kw):
    """Live-like replay: each day take the top_k candidates and trade the one whose signal fires
    first (that is what the live bot does), at most one trade per day."""
    days = M.index
    res = []
    for d in range(len(days) - test_days, len(days)):
        cands = select_day(M, d, **kw)[:top_k]
        today = M.iloc[d]
        firing = [c for c in cands if today[c] == today[c]]
        if T is not None and firing:
            when = T.iloc[d]
            firing.sort(key=lambda c: when[c])
        pick = firing[0] if firing else None
        res.append((days[d], None if pick is None else "|".join(pick), np.nan if pick is None else today[pick],
                    len(cands)))
    return pd.DataFrame(res, columns=["day", "pick", "R", "n_cands"])


def summarize(wf):
    t = wf.dropna(subset=["R"])
    if t.empty:
        return dict(trades=0)
    eq = (1 + config.RISK_PER_TRADE * t["R"]).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(trades=int(len(t)), win=float((t["R"] > 0).mean()), avg_R=float(t["R"].mean()),
                t_stat=float(t["R"].mean() / (t["R"].std() / np.sqrt(len(t)))) if len(t) > 2 else None,
                total_R=float(t["R"].sum()), equity_mult=float(eq.iloc[-1]), max_dd=float(dd),
                days_with_candidates=int((wf["n_cands"] > 0).sum()))


def matrix(df, values="R"):
    M = df.pivot_table(index="day", columns=["variant", "sym"], values=values, aggfunc="first")
    full = pd.date_range(df["day"].min(), df["day"].max(), freq="D")
    return M.reindex(full)


def run_select(test_days=365, null_runs=30):
    df = pd.read_parquet(TRADES)
    M = matrix(df)
    T = matrix(df.assign(ts=df["entry_time"].astype("int64")), "ts").reindex(columns=M.columns)
    out = {"variants": int(df["variant"].nunique()), "combos": int(M.shape[1]),
           "trades_simulated": int(len(df)), "test_days": test_days, "settings": {}}
    rng = np.random.default_rng(0)
    for t_req in (2.0, 2.5, 3.0):
        wf = walk_forward(M, test_days, T, t_req=t_req)
        real = summarize(wf)
        # null: random sign per trade destroys any directional edge but keeps everything else
        nulls = []
        for _ in range(null_runs):
            Mn = M * rng.choice([-1.0, 1.0], size=M.shape)
            nulls.append(summarize(walk_forward(Mn, test_days, T, t_req=t_req)).get("total_R", 0.0))
        real["null_total_R_p95"] = float(np.percentile(nulls, 95))
        real["beats_null"] = bool(real.get("total_R", -1e9) > real["null_total_R_p95"])
        out["settings"][f"t>={t_req}"] = real
        wf.to_csv(os.path.join(ROOT, "results", f"walkforward-t{t_req}.csv"), index=False)
        print(t_req, real, flush=True)
    with open(os.path.join(ROOT, "results", "research.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "trades"
    if cmd == "trades":
        print(run_trades().groupby("variant")["R"].agg(["count", "mean"]).describe())
    elif cmd == "select":
        run_select()
