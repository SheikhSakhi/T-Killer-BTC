# T-Killer: daily dashboard

_Updated 2026-11-27 22:45 UTC · mode paper_

## Plan for 2026-11-28

| # | Coin | Strategy | Session | Train trades | Train avg R | t | Last 30d avg R |
|---|---|---|---|---|---|---|---|
| 1 | SUIUSDT | trend-206 (trend) | europe | 30 | +0.78 | 2.36 | +0.13 |
| 2 | SUIUSDT | trend-208 (trend) | europe | 29 | +0.45 | 2.08 | +0.03 |

The bot trades the first of these whose signal fires; one trade per day.

## How the selection procedure itself has done (out of sample, 1% risk per trade)

- last 90 days: 37 trades, win 43%, avg -0.22R, total -8.2R, max drawdown -11.1%
- last 365 days: 57 trades, win 53%, avg -0.02R, total -1.1R, max drawdown -11.1%

## Live results

- No closed trades yet.

> Not financial advice. Small-stake experiment; the guards and limits are in `tk/config.py`.
