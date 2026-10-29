# XAUUSDT order-flow study (`research/xau_flow.py`, data: `research/xau_flow_data.py`)

Data Dec 2025 – Oct 2026 from data.binance.vision: order-book depth snapshots every 30 s (±0.2–5%),
taker buy/sell volume, open interest and long/short ratios every 5 min, premium index, funding.
23 hourly features in five groups (flow, book, crowd, basis, context), all known at the decision time.

## 1. Each force alone (rank correlation with the next move, 1st half | 2nd half, non-overlapping)
Stable in both halves (|IC| ≥ 0.05, same sign): 6 of 46 tests (about 3 expected by chance).
Strongest: book imbalance within ±5% vs the next 24 h: +0.21 | +0.21. Others stable but weak:
top-trader long/short z (−), last-hour taker buying (−), 24 h change of 1% book imbalance (+).
Nothing is stable at the 4 h horizon.

## 2. All forces combined (ridge, weekly walk-forward re-fit, purged)
24 h direction hit 49.0%, 4 h 50.3%: no better than a coin. Trades: −0.39R / −0.36R per trade.

## 3. Book imbalance as a trading rule (daily 00:00 UTC entry)
The imbalance is positive 83% of hours, so its sign is mostly "long".
| Variant | 1st half | 2nd half |
|---|---|---|
| sign, 24 h, stop 1% / target 2% | +0.21R | −0.13R |
| vs 7-day median, 24 h, 1% / 2% | +0.17R | −0.07R |
| top/bottom 20% of its 30-day range, 24 h, 1% / 2% | +0.36R | −0.11R |
| same, 7 days, 2% / 4% | +0.56R | −0.21R |
Every variant made money while gold rose and lost when it fell: the correlation is real but does
not give a trade that survives the second half. Not adopted.
