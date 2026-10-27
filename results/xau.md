# XAUUSDT analyst (`python -m tk.xau`)

Rule chosen from 4 direction rules × 24 stop/target/holding variants on XAUUSDT (Dec 2025–Oct 2026)
and PAXGUSDT (Mar 2025–Oct 2026), checked on both halves:

| Direction rule | Result |
|---|---|
| Trend: price vs 20-day average of daily closes | positive in both halves on both instruments |
| 5-day momentum | ~0, unstable |
| Mean reversion inside the 20-day range | negative |
| Always long | negative in the second half |

Chosen exits: stop 2%, target 4%, max 7 days. Non-overlapping replay: XAUUSDT 68 trades, 44% win,
+0.09R per trade; PAXGUSDT 114 trades, 43% win, +0.10R. t < 1: not statistically proven.
Volatility forecast (7-day hourly sigma): the next day stayed inside ±1σ on 73% of days.
Every call is logged to `data/xau/signals.csv` and scored on later calls.
