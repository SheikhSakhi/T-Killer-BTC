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
