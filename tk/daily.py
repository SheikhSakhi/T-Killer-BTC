"""Nightly run on the server (after the flat time): refresh data, replay all variants, pick
tomorrow's candidates, write the plan and the dashboard, send the daily summary.

python -m tk.daily
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tk import config, data, notify, research, strategies
from tk.exchange import Futures

ROOT = research.ROOT
PLAN_DIR = os.path.join(ROOT, "data", "plan")
LIVE_TRADES = os.path.join(ROOT, "data", "live", "trades.csv")


def build_plan(M, grid, day):
    tab = research.candidate_table(M, len(M.index), **config.SELECT)
    ok = tab[tab["ok"]].head(config.TOP_K)
    cands = []
    for (variant, sym), r in ok.iterrows():
        cands.append(dict(variant=variant, sym=sym, params=grid[variant], n_train=int(r["n_train"]),
                          mean_train=round(float(r["mean_train"]), 3), t_train=round(float(r["t_train"]), 2),
                          n_val=int(r["n_val"]), mean_val=round(float(r["mean_val"]), 3)))
    return dict(day=str(day), made_at=str(datetime.now(timezone.utc)), select=config.SELECT,
                candidates=cands, tested=int(M.shape[1]))


def live_summary():
    if not os.path.exists(LIVE_TRADES):
        return None
    t = pd.read_csv(LIVE_TRADES)
    if t.empty:
        return None
    return dict(trades=len(t), win=float((t["pnl"] > 0).mean()), pnl=float(t["pnl"].sum()),
                avg_R=float(t["R"].mean()), last=t.tail(5)[["closed_at", "sym", "variant", "pnl", "R"]].to_dict("records"))


def dashboard(plan, wf90, wf365, live):
    L = ["# T-Killer: daily dashboard", "",
         f"_Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · mode {os.environ.get('MODE', 'paper')}_", "",
         f"## Plan for {plan['day']}", ""]
    if plan["candidates"]:
        L += ["| # | Coin | Strategy | Session | Train trades | Train avg R | t | Last 30d avg R |",
              "|---|---|---|---|---|---|---|---|"]
        for i, c in enumerate(plan["candidates"], 1):
            L.append(f"| {i} | {c['sym']} | {c['variant']} ({c['params']['family']}) | {c['params']['session']} | "
                     f"{c['n_train']} | {c['mean_train']:+.2f} | {c['t_train']} | {c['mean_val']:+.2f} |")
        L += ["", "The bot trades the first of these whose signal fires; one trade per day."]
    else:
        L += ["No variant passed the guards. **No trade tomorrow.** Sitting out is a position."]
    L += ["", "## How the selection procedure itself has done (out of sample, 1% risk per trade)", ""]
    for name, w in (("last 90 days", wf90), ("last 365 days", wf365)):
        if w.get("trades"):
            L.append(f"- {name}: {w['trades']} trades, win {w['win']:.0%}, avg {w['avg_R']:+.2f}R, "
                     f"total {w['total_R']:+.1f}R, max drawdown {w['max_dd']:.1%}")
        else:
            L.append(f"- {name}: no trades")
    L += ["", "## Live results", ""]
    if live:
        L.append(f"- {live['trades']} trades, win {live['win']:.0%}, P&L {live['pnl']:+.2f} USDT, avg {live['avg_R']:+.2f}R")
    else:
        L.append("- No closed trades yet.")
    L += ["", "> Not financial advice. Small-stake experiment; the guards and limits are in `tk/config.py`.", ""]
    return "\n".join(L)


def run():
    pub = Futures(testnet=False)   # research always uses the real market
    for sym in config.COINS + config.CONTEXT:
        try:
            data.update(sym, pub)
        except Exception as e:
            print("update failed", sym, e)
    df = research.run_trades()
    M = research.matrix(df)
    T = research.matrix(df.assign(ts=df["entry_time"].astype("int64")), "ts").reindex(columns=M.columns)
    grid = {p["id"]: p for p in strategies.grid()}

    tomorrow = (pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None) + pd.Timedelta(hours=2)).normalize().date()
    plan = build_plan(M, grid, tomorrow)
    os.makedirs(PLAN_DIR, exist_ok=True)
    for name in ("latest.json", f"{tomorrow}.json"):
        with open(os.path.join(PLAN_DIR, name), "w") as f:
            json.dump(plan, f, indent=1, default=str)

    sel = {k: v for k, v in config.SELECT.items()}
    wf90 = research.summarize(research.walk_forward(M, 90, T, config.TOP_K, **sel))
    wf365 = research.summarize(research.walk_forward(M, 365, T, config.TOP_K, **sel))
    live = live_summary()
    md = dashboard(plan, wf90, wf365, live)
    with open(os.path.join(ROOT, "LATEST.md"), "w") as f:
        f.write(md)
    with open(os.path.join(ROOT, "results", "daily.json"), "w") as f:
        json.dump(dict(plan=plan, wf90=wf90, wf365=wf365, live=live), f, indent=1, default=str)

    top = plan["candidates"][:3]
    lines = [f"{c['sym']} {c['params']['family']}/{c['params']['session']} ({c['mean_val']:+.2f}R last 30d)" for c in top]
    notify.send(f"📋 T-Killer plan for {tomorrow}: {len(plan['candidates'])} candidates",
                ("\n".join(lines) if lines else "No candidate passed the guards: no trade tomorrow.")
                + f"\n\nProcedure, last 90 days: {wf90.get('trades', 0)} trades, total {wf90.get('total_R', 0):+.1f}R")
    print(md)


if __name__ == "__main__":
    run()
