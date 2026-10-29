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
| 2026-10-26 | Owner's corridor idea: band ≤9% for 10x; long from bottom third of corridors aged ≥14d, TP at top | promising, not proven: +9–11% of margin per trade in both halves, 8–12% liquidations, n=53, t≈1; shorts lose. results/corridor.md |
| 2026-10-27 | Gold (XAUUSDT 10 mo, PAXGUSDT 19 mo): corridor rule fails (TP at corridor top hit 1 in 21); direction rules: trend (price vs 20-day SMA) is the only one positive in both halves; best non-overlapping: SL 2% / TP 4% / ≤7d, +0.09..0.10R, win 42%, t<1. Mean reversion and momentum lose. | weak, unproven |
| 2026-10-27 | Gold direction factors: trend 10/20/50/100, real yields, dollar, COT, weekend fade (~15 rules) | kept vote(trend20, trend50, dollar20 lag 9d): +0.27/+0.33R PAXG, +0.19/+0.19R XAU; yields, COT, weekend: no. tk/xau.py uses it. Stooq resets automated requests: don't use. FRED broad dollar is published weekly: lag it. |
| 2026-10-28 | Mirror level (owner's friend): retest of a former 12h-pivot high from above → long (or low from below → short), XAU+PAXG | long side lost: −0.16R (2% stop) / −0.28R (stop under level), 194 trades; short ≈ 0. Not added. |
| 2026-10-28 | Friend's mirror-level scalp on gold: TP 0.5–0.6%, SL 0.5/1/2%, 48h max, maker or taker fees, XAU+PAXG, 488 trades | negative in every variant: −0.06..−0.21% of position per trade; win rate 44–70%; best case (maker, short with the trend vote, SL 1%) ≈ −0.02%. |
| 2026-10-29 | XAU rule: market entry vs limit entry 0.3/0.5/0.7% better, valid 24h (unfilled = no trade), 169 signals | 0.3% limit slightly better in both halves (+0.25/+0.12R vs +0.20/+0.11R, fills 72%); 0.5–0.7% worse (missed winners). Added as the suggested entry. Winners typically go 0.75% against first (56% of winners ≥0.7%). |
| 2026-10-29 | Gold 1-day signals: 7 direction rules (vote3, trend20/50, 1-day momentum/reversal, Asia-session direction, hourly RSI reversal) × entry 00:00/07:00 UTC × 9 stop/target pairs, 24h exit, 0.08% costs | 0 of 117 combinations positive in both halves; best pooled ≈ 0. No 1-day edge found; owner told. |
| 2026-10-29 | XAU order flow (owner's request): 23 features (taker flow, book depth imbalance ±0.2–5%, OI, L/S ratios, premium, funding), 46 IC tests + walk-forward ridge + book-imbalance trading rules | book imbalance ±5% IC +0.21/+0.21 at 24h (only strong one), but as a trade +0.2..0.6R in 1st half, −0.1..−0.2R in 2nd; combined model = coin flip. Not adopted. results/xau_flow.md |
| 2026-10-29 | Hourly predict→check→learn loop (owner's idea), replayed on 6,015 hours (Feb–Oct 2026), 23 order-flow features: ridge on all past / last 30d, 50-nearest "similar situations", repeat last hour | hit rate 50.7 / 50.7 / 52.0 / 48.0% — INVALID, see the 2026-10-29 lookahead entry below. No learning curve over months. 2-day windows range 33–67%: two days cannot tell skill from luck. Fixed zero-OI rows in xau_flow.py. |
| 2026-10-29 | AUDIT of order-flow data (owner asked "is the data right?") | Found: (1) LOOKAHEAD — metrics (OI, L/S) were shifted −1h, so each hour saw the value from 55 min into the predicted hour; (2) last ~42 price hours came from testnet (synthetic volume); (3) October funding stale. Fixed all three; added tests/test_xau_flow_leak.py (corrupts the future, features before must not change; verified it fails on the old code). Clean rerun: hourly "similar situations" 52.0% → **50.3% (coin)**; ridge 49.6%; only "opposite of last hour" 51.7% (z 2.7, untradeable after costs). 24h single-feature ICs unchanged (book imbalance ±5% +0.21/+0.21). |
