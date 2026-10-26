import sys
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as sw
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..'))
from tk import data
H = 168
SYMS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","SUIUSDT","ADAUSDT"]
pd.set_option("display.width", 220)

# 1) after a 5% drawdown, how often does price get back to entry before 7 days AND before 10x liquidation?
rows = []
lev_rows = []
for sym in SYMS:
    h = data.load("1h", sym); o = h["close"].values; n = len(o) - H
    HI = sw(h["high"].values[1:], H)[:n]; LO = sw(h["low"].values[1:], H)[:n]; e = o[:n, None]
    for side in (1, -1):
        adv = (LO / e - 1) if side > 0 else (1 - HI / e)
        fav = (HI / e - 1) if side > 0 else (1 - LO / e)
        dd5 = adv <= -0.05
        has = dd5.any(1); t5 = dd5.argmax(1)
        idx = np.arange(H)[None, :]
        after = idx > t5[:, None]
        back = ((fav >= 0) & after).any(1) & has
        tb = np.where(((fav >= 0) & after).any(1), ((fav >= 0) & after).argmax(1), H)
        liq = adv <= -0.09; tl = np.where(liq.any(1), liq.argmax(1), H)
        back_before_liq = back & (tb < tl)
        rows.append(dict(sym=sym, side="long" if side > 0 else "short", p_down5=has.mean(),
                         p_back_given_down5=back_before_liq[has].mean() if has.any() else np.nan,
                         p_liq_given_down5=(tl < H)[has].mean() if has.any() else np.nan))
        if sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT") and side > 0:
            mae = adv.min(1)
            lev_rows.append(dict(sym=sym, **{f"{L}x": (mae <= -(1 / L - 0.005)).mean() for L in (2, 3, 5, 10, 20)}))
print("== After going 5% against you (7-day window):")
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\n== Share of hourly LONG entries liquidated within 7 days, by leverage:")
print(pd.DataFrame(lev_rows).round(3).to_string(index=False))

# 2) stability: EV per 10x trade with TP 2% by half-year, for every coin/side
pe = pd.read_pickle('' + __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'cache') + '/lev10_per_entry.pkl')
st = {}
for (sym, side), (idx, res, liq) in pe.items():
    s = pd.Series(res[0.02], index=idx)
    st[f"{sym[:-4]} {'L' if side > 0 else 'S'}"] = s.groupby(pd.Grouper(freq="6MS")).mean()
st = pd.DataFrame(st).T
st.columns = [c.strftime("%Y-%m") for c in st.columns]
print("\n== EV per trade (% of margin), 10x, take profit +2%, by half-year:")
print((st * 100).round(1).to_string())

# 3) walk-forward: each month pick the coin/side with the LOWEST liquidation rate over the previous
#    90 days (the 'safest' coin), trade one entry per day at 00:00 the next month.
month_res = []
keys = list(pe.keys())
liqs = {k: pd.Series(pe[k][2].astype(float), index=pe[k][0]) for k in keys}
pnls = {k: pd.Series(pe[k][1][0.02], index=pe[k][0]) for k in keys}
months = pd.date_range("2025-02-01", "2026-10-01", freq="MS")
for m in months:
    # entries in the training window must have finished their 7 days before m
    tr = {k: liqs[k][(liqs[k].index >= m - pd.Timedelta(days=97)) & (liqs[k].index < m - pd.Timedelta(days=7))].mean() for k in keys}
    best = min(tr, key=tr.get)
    p = pnls[best]
    test = p[(p.index >= m) & (p.index < m + pd.offsets.MonthBegin(1)) & (p.index.hour == 0)]
    month_res.append(dict(month=m.strftime("%Y-%m"), pick=f"{best[0]} {'L' if best[1] > 0 else 'S'}",
                          train_liq=round(tr[best], 3), trades=len(test),
                          liquidated=int((test <= -1).sum()), month_pnl_pct_of_margin=round(test.sum() * 100, 1)))
wf = pd.DataFrame(month_res)
print("\n== Walk-forward: pick the 'safest' coin by the last 90 days, trade it next month (1 entry/day, 10x, TP 2%):")
print(wf.to_string(index=False))
print("total trades", wf.trades.sum(), "liquidations", wf.liquidated.sum(),
      "sum of P&L as % of one margin", round(wf.month_pnl_pct_of_margin.sum(), 1))
