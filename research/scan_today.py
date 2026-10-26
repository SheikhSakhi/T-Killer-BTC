"""Scan a wider universe for today's candidates under the same guards as the live plan.

python research/scan_today.py download SYM1,SYM2,...   # 12 months of data for missing symbols
python research/scan_today.py select   SYM1,SYM2,...   # simulate + select candidates for the next day
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tk import config, data, research, strategies  # noqa: E402


def fetch(sym, months=12):
    if os.path.exists(data.path("metrics", sym)):
        return sym, "cached"
    try:
        data.klines(sym, "5m", months).to_parquet(data.path("5m", sym))
        data.klines(sym, "1h", months).to_parquet(data.path("1h", sym))
        data.funding(sym, months).to_frame().to_parquet(data.path("funding", sym))
        data.metrics(sym, months).to_parquet(data.path("metrics", sym))
        return sym, "ok"
    except Exception as e:
        return sym, f"failed {e!r}"[:120]


def safe_sim(sym):
    try:
        return research.simulate_coin(sym)
    except Exception as e:
        print(sym, "skipped:", repr(e)[:100], flush=True)
        return pd.DataFrame()


if __name__ == "__main__":
    cmd, syms = sys.argv[1], sys.argv[2].split(",")
    os.makedirs(data.CACHE, exist_ok=True)
    if cmd == "download":
        with ProcessPoolExecutor(6) as ex:
            for s, st in ex.map(fetch, syms):
                print(s, st, flush=True)
    else:
        with ProcessPoolExecutor(os.cpu_count()) as ex:
            df = pd.concat(list(ex.map(safe_sim, syms)), ignore_index=True)
        df.to_parquet(os.path.join(data.CACHE, "scan_trades.parquet"))
        M = research.matrix(df)
        tab = research.candidate_table(M, len(M.index), **config.SELECT)
        tab.to_csv(os.path.join(data.CACHE, "scan_table.csv"))
        print(tab[tab["ok"]].head(20).round(3).to_string())
        print("tested combos:", M.shape[1], "passing:", int(tab["ok"].sum()), "last day in data:", M.index[-1].date())
