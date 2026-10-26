# Corridor study: coins that stay inside a 9% band (for 10x without a stop)

Script: `research/corridor.py` (+ tests/test_corridor.py). 32 most liquid USDT perpetuals,
1h candles (2 years for 10 coins, 1 year for the rest). Corridor = the longest stretch ending now in
which (max high / min low − 1) ≤ 9%. Forward window 7 days; 10x liquidation at −9%.

## Corridor age vs what happens next (hourly samples; second half is out of sample)
| Corridor age | survives +1 day | survives +7 days | 10x long liquidated in 7d | 10x short liquidated in 7d |
|---|---|---|---|---|
| < 1 day | 18% | 0.2% | 60% | 57% |
| 3–7 days | 51% | 4% | 31% | 33% |
| 7–14 days | 69% | 13% | 12% | 22% |
| 14–30 days | 79% | 23% | 4% | 23% |
| any (unconditional) | | | 42% | 43% |

## Independent trades (one per 7 days): long from the bottom third of a corridor aged ≥14 days,
take profit at the corridor top
| Variant | Trades 1st / 2nd half | Stopped or liquidated | Avg P&L per trade, % of margin | t-stat |
|---|---|---|---|---|
| 10x, liquidation as stop | 16 / 37 | 12.5% / 8.1% | +11% / +9% | 0.9 / 1.1 |
| 10x, stop 1% below corridor | 16 / 37 | 56% / 32% | −2% / +13% | −0.2 / 2.0 |
| 5x, liquidation as stop | 16 / 37 | 0% / 5.4% | +9% / +2% | 2.3 / 0.4 |
| Shorts from the top third | 19 / 46 | 26% / 28% | −19% / −24% | negative |

Read: the age of a corridor is a real signal for *safety* (old corridors liquidate far less), and
longs from the bottom of old corridors were positive in both halves. Only 53 independent trades, so
the profit is not statistically proven (t ≈ 1). Shorts lose. Corridors rarely last another full week
(13–23%), so the trade must aim for the opposite edge within days, not wait for a week.
