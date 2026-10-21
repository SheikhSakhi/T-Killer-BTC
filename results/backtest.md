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

## 180 windows

Run the `backtest` workflow from the Actions tab; output lands in `results/backtest-180w.json`.
