# T-Killer memory

Read this first in every session. Newest entries at the bottom of each section.

## Facts established
- 2026-10-21..23: Kronos-small (price-only foundation model) has no 24h edge on BTC: 49.4% direction
  over 180 windows, toy strategy −40.6%; agreement filter made it worse; its band width tracks
  volatility but adds nothing over trailing 7-day vol. Retired.
- 2026-10-24: Universe = SOL, XRP, DOGE, BNB, LINK, AVAX, SUI, ADA (BTC/ETH minimum orders too big
  for 40 USDT; BTC/ETH used as context). Data: 2 years of 5m/1h klines, funding, OI, top-trader L/S.
- 2026-10-24: 228 variants × 8 coins, 274k simulated trades; average variant −0.09R per trade
  (fees and slippage dominate tight stops).
- 2026-10-24: Walk-forward of the daily selection, last 365 days, 1 trade/day max:
  t≥1.0 → 189 trades, −28R; t≥1.5 → 107 trades, −13R; t≥2.0 → 49 trades, +6.4R (t-stat 0.88,
  does not beat the random-sign null). Looser selection loses money; kept t≥2.0.

## Environment gotchas
- api/fapi.binance.com return 451 from US hosts (GitHub Actions, Claude cloud sessions).
  The server in Frankfurt can reach them. data.binance.vision works from anywhere.
- Binance testnet has no /futures/data endpoints; OI and L/S always come from the live public API.
- Conditional orders (STOP_MARKET/TAKE_PROFIT_MARKET) go through /fapi/v1/algoOrder, with a
  fallback to /fapi/v1/order. Not yet verified on the testnet with keys.

## Tried (every idea tested, for the multiple-testing count)
| Date | Idea | Result |
|---|---|---|
| 2026-10-21 | Kronos 24h BTC direction | no edge |
| 2026-10-22 | Kronos agreement ≥80% filter | worse |
| 2026-10-23 | Kronos band as volatility forecast | no better than trailing vol |
| 2026-10-24 | 4 families (orb, squeeze, funding, trend), coarse grid, t≥2.0 guards | +6.4R/yr, not significant |

## Open questions / next ideas
- Liquidation data (needs a websocket collector on the server).
- Session-specific behaviour around funding settlements (08:00/16:00 UTC).
- Better null: compare avg R per trade, not total R, since candidate counts differ.
| 2026-10-26 | 10x "safe coin returns to entry within a week" (owner's idea) | rejected: 11–49% of entries liquidated in 7 days, walk-forward −18.4 margins; see results/lev10.md |
