# T-Killer: daily dashboard

_Updated 2026-11-11 22:45 UTC · mode paper_

## Plan for 2026-11-12

| # | Coin | Strategy | Session | Train trades | Train avg R | t | Last 30d avg R |
|---|---|---|---|---|---|---|---|
| 1 | SUIUSDT | trend-204 (trend) | europe | 27 | +0.40 | 2.41 | +0.02 |
| 2 | XRPUSDT | trend-206 (trend) | europe | 20 | +0.94 | 2.11 | +0.00 |

The bot trades the first of these whose signal fires; one trade per day.

## How the selection procedure itself has done (out of sample, 1% risk per trade)

- last 90 days: 42 trades, win 50%, avg -0.08R, total -3.3R, max drawdown -9.3%
- last 365 days: 55 trades, win 55%, avg +0.02R, total +0.9R, max drawdown -9.3%

## Live results

- No closed trades yet.

> Not financial advice. Small-stake experiment; the guards and limits are in `tk/config.py`.
