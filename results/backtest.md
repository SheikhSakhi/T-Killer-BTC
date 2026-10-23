# Historical check (before live logging)

Kronos-small, BTC/USDT 1h candles from Binance, 400-candle lookback, 24h horizon.
Non-overlapping 24h windows; at each window the model saw only earlier candles.
Baseline: "price in 24h equals price now". Toy strategy: long or short for 24h on the
predicted direction, 0.1% round-trip cost.

## 60 windows, the 60 days to 2026-10-20, 5 samples per window

| Metric | Kronos | Baseline |
|---|---|---|
| Direction right | 60% (36/60) | up days: 52% |
| Mean absolute error, $ | 1,961 | 991 |
| Actual inside the 10–90% band | 20% (should be 80%) | |
| Toy strategy, total | +17.9% | buy and hold +6.0% |

Read: 60% on 60 tries is within luck (about 8% chance for a coin). Price-level error is twice the
naive baseline and the band is far too narrow. Not a tradeable edge yet.

## 180 windows, 2026-04-24 to 2026-10-20, 10 samples per window

Run on GitHub Actions (`backtest` workflow), raw output in `backtest-180w.json`.

| Metric | Kronos | Baseline |
|---|---|---|
| Direction right | 49.4% (89/180) | up days: 51.7% |
| Mean absolute error, $ | 1,751 | 1,020 |
| Actual inside the 10–90% band | 24% (should be 80%) | |
| Toy strategy, total | **−40.6%** | buy and hold +6.7% |

Read: over six months raw Kronos-small has no directional edge on BTC 24h (a coin flip), its
price error is 72% worse than "no change", and its uncertainty band is far too narrow. The
60-window +17.9% was luck. Do not trade on it; the live log keeps measuring.

## Confidence filter, partial: 96 of 180 windows (2026-04-25 to 2026-07-29), 30 samples

| Trade only when paths agree | Trades | Profitable | Avg trade | Total |
|---|---|---|---|---|
| all days | 96 | 47% | −0.17% | −16.8% |
| ≥70% | 79 | 43% | −0.37% | −26.4% |
| ≥80% | 68 | 43% | −0.43% | −26.4% |
| ≥90% | 57 | 42% | −0.43% | −22.6% |
| buy and hold | | | | −16.5% |

Read: higher agreement does not help; confident calls were wrong more often. The remaining 84
windows were not run (GitHub minutes); rerun `backtest` with windows=180 samples=30 to resume.

## Volatility instead of direction, same 96 windows

Does the width of Kronos's 10–90% band rank how much BTC moves in the next 24h? Spearman
correlation, permutation p-value, against two naive baselines (trailing realised vol).

| Predictor | vs realised vol (24 hourly returns) | vs 24h high–low range |
|---|---|---|
| Kronos band width | 0.23 (p=0.02) | 0.29 (p=0.006) |
| last 7 days' vol | 0.29 (p=0.006) | 0.27 (p=0.008) |
| last 24h vol | 0.26 (p=0.01) | 0.25 (p=0.02) |

Read: Kronos does carry volatility information, but no more than last week's volatility, and
in a joint rank regression it adds nothing significant (t=0.5 and 1.4). Its band is also too
narrow: median width 2.0% vs a median 24h range of 2.7%.
