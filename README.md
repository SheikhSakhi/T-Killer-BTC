# T-Killer-BTC

A daily BTC/USDT 24-hour forecast from [Kronos](https://github.com/shiyu-coder/Kronos)
(Kronos-small, AAAI 2026), logged before the outcome is known and scored the next day against a
"price stays flat" baseline. The point is an honest forward track record, not a trading bot.

## How it works

- `.github/workflows/daily.yml` runs every day at 00:07 UTC (and on demand from the Actions tab).
- `forecast/daily.py` downloads the last 25 days of hourly candles from Binance (archive fallback),
  fills in the actual price for every forecast whose 24h target has closed, makes today's
  forecast from 30 sampled paths, and rewrites `LATEST.md`.
- `data/log.csv` is the append-only journal. Each row is committed before its outcome exists, so the
  history cannot be edited after the fact without it showing in git.
- `forecast/backtest.py` reruns the historical check. Earlier results are in `results/backtest.md`.

## Asking Claude

Open Claude with this repository connected and ask "какое движение по битку?". `CLAUDE.md` tells
Claude to read `LATEST.md` and nothing else, so the answer costs a few thousand tokens instead of
an install-and-run session.

## Run locally

```bash
git clone https://github.com/shiyu-coder/Kronos vendor/Kronos
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python forecast/daily.py
```

Not financial advice.
