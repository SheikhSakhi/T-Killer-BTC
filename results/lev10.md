# 10x "safe coin, it always comes back" hypothesis — rejected

Data: Binance USD-M 1h candles, 2024-10 to 2026-10, 10 coins (BTC, ETH, BNB, SOL, XRP, DOGE,
LINK, AVAX, SUI, ADA). Every hourly entry, long and short, held up to 7 days. Liquidation at a 9%
adverse move (10x isolated). Fees 1% of margin per round trip, funding ~2% of margin per week.
Scripts: `research/lev10_scan.py`, `research/lev10_stability.py`.

| Finding | Number |
|---|---|
| Share of entries liquidated within 7 days at 10x | BTC 11–14%, ETH 28–30%, alts 31–49% |
| Same at 3x (BTC / ETH / SOL / ADA longs) | 0% / 1% / 1.6% / 2.6% |
| After a 5% move against you: back to entry before liquidation | 25–44% |
| After a 5% move against you: liquidated within the week | 35–69% |
| Win rate with take-profit +2% | 71–83% (matches the zero-edge gambler's-ruin value 9/11 = 82%) |
| EV per trade, any coin/side | sign flips between half-years (follows the market trend, not the coin) |
| Walk-forward: each month trade the coin with the fewest liquidations in the last 90 days | 623 trades, 52 liquidations, −18.4 margins in total |

Why it cannot work: for a price with no drift, the chance to reach +a before −b is b/(a+b), so the
expected profit of any take-profit/stop pair is zero before costs (optional stopping theorem).
A high win rate is bought with rare full-margin losses; fees and funding make the sum negative.
Past calm does not protect: October 2025's best-ranked pick (BTC long, 0.1% liquidations in the
prior 90 days) was liquidated 9 times that month.
