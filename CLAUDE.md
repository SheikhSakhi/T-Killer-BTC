# T-Killer: instructions for Claude

## What this repo is
A small-stake (40 USDT) Binance USD-M futures system. The server (Oracle VM, Frankfurt) runs:
- `deploy/run.sh live` every minute: `tk/live.py` trades today's plan, one trade per UTC day, flat by 22:30 UTC.
- `deploy/run.sh daily` at 22:45 UTC: `tk/daily.py` refreshes data, replays every strategy variant on
  8 coins (`tk/research.py`, `tk/strategies.py`), picks tomorrow's candidates through the guards in
  `tk/config.py` (SELECT), writes `data/plan/latest.json`, `LATEST.md`, `results/daily.json`.
The server pushes its logs here: `data/live/trades.csv`, `data/live/events.log`, `data/live/state.json`.

## XAUUSDT request ("XAU", "золото", "XAUUSDT", "что по золоту")
Run `python -m tk.xau <risk_usd>` (default risk 1000; use the owner's number if given) after
`pip install -q -r requirements-tk.txt` if pandas is missing. Relay its output in Russian as is:
one decision (LONG or SHORT) with entry, stop, target, size, volatility ranges and the track record.
Do not add "or" alternatives and do not override the rule with your own view. If the price source
is "testnet", say the owner should check the entry against the real Binance price. Commit the
updated `data/xau/signals.csv` so the live record keeps accumulating.

## Quick question from the owner ("что по рынку", "какое движение", "что сегодня")
Read `LATEST.md` only and answer in Russian, briefly. Do not run anything. If `LATEST.md` is
older than 36 hours, say the nightly job probably failed and point to the server logs.

## Daily review (scheduled session)
1. Read `MEMORY.md` first, then `LATEST.md`, `results/daily.json`, the tail of `data/live/events.log`,
   and `data/live/trades.csv`.
2. Find errors: exceptions in events.log, entries whose fill is far from `signal_entry`
   (slippage), trades that broke the risk frame, a plan that is stale, a nightly run that did not happen.
3. Fix bugs with a test in `tests/`, run `python -m pytest -q tests`, commit, push.
4. Improvements to strategies go through the research pipeline only: add or change a family in
   `tk/strategies.py`, rerun `python -m tk.research trades` and `select`, and adopt it only if the
   walk-forward result improves without loosening the guards. Never change `tk/config.py` risk
   numbers or SELECT guards without the owner's explicit approval.
5. Append what you learned and what you changed to `MEMORY.md` (dated, short), and record every
   tested idea in its "Tried" table so the multiple-testing count stays honest.
6. Write a 5-line summary to `data/review/latest.md` (the server sends it to Telegram and email).

## Hard rules
- No live-money change (`MODE`, `BINANCE_TESTNET`, keys) from a Claude session; the owner does that on the server.
- Never print or commit secrets. `.env` lives only on the server.
- One trade per day, 1% risk, 3x max leverage, daily stop 3%, drawdown stop 25%: unchanged without approval.
