"""10x leverage 'safe coin' hypothesis: every hourly entry, long and short, 7-day hold.
Liquidation when adverse move >= LIQ (isolated 10x: 10% minus maintenance margin ~ 9%)."""
import sys
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as sw
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from tk import data
LEV, LIQ, H = 10, 0.09, 168
FEE = 0.001 * LEV            # 0.05% taker in + out, on notional, as % of margin
FUND = 0.0001 * 21 * LEV     # ~0.01% per 8h for a week, longs usually pay
SYMS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","SUIUSDT","ADAUSDT"]
rows = []
per_entry = {}
for sym in SYMS:
    h = data.load("1h", sym)
    o = h["close"].values  # enter at hourly close
    hi, lo = h["high"].values, h["low"].values
    n = len(o) - H
    HI = sw(hi[1:], H)[:n]; LO = sw(lo[1:], H)[:n]; CL = sw(h["close"].values[1:], H)[:n]
    e = o[:n, None]
    for side in (1, -1):
        adverse = (LO / e - 1) if side > 0 else (1 - HI / e)        # negative numbers
        fav = (HI / e - 1) if side > 0 else (1 - LO / e)
        liq_hit = adverse <= -LIQ
        liq_t = np.where(liq_hit.any(1), liq_hit.argmax(1), H)
        mae = adverse.min(1)
        # back to entry within the week after being in the red
        went_red = mae < -0.005
        final = side * (CL[:, -1] / o[:n] - 1)
        res = {}
        for tp in (0.01, 0.02, 0.03, 0.05, None):
            if tp is None:
                tp_t = np.full(n, H)
            else:
                tp_hit = fav >= tp
                tp_t = np.where(tp_hit.any(1), tp_hit.argmax(1), H)
            # same-hour liq and tp: liquidation first (pessimistic)
            liq_first = liq_t <= tp_t
            liq = liq_first & (liq_t < H)
            win = (~liq) & (tp_t < H)
            pnl = np.where(liq, -1.0, np.where(win, (tp or 0) * LEV, final * LEV)) - FEE - (FUND if side > 0 else -FUND)
            pnl = np.where(liq, -1.0, pnl)   # liquidation loses the whole margin
            res[tp] = pnl
        per_entry[(sym, side)] = (h.index[:n], res, liq_t < H)
        rows.append(dict(sym=sym, side="long" if side > 0 else "short",
                         liq_week=(liq_t < H).mean(), worst_mae=mae.min(), p_back_to_entry=( (fav >= 0).any(1)[went_red] ).mean() if went_red.any() else np.nan,
                         **{f"ev_tp{int(tp*100) if tp else 'none'}": res[tp].mean() for tp in res},
                         win_tp2=(res[0.02] > 0).mean()))
df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(df.round(3).to_string(index=False))
pd.to_pickle(per_entry, '' + __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'cache') + '/lev10_per_entry.pkl')
