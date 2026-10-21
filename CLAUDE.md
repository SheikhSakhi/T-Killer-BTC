# T-Killer-BTC: instructions for Claude

When the user asks about BTC ("какое движение по битку", "что с биткоином", "прогноз BTC"):

1. Read `LATEST.md` only. It is regenerated every day by GitHub Actions and holds today's
   forecast plus the live track record. Do not run the model, install packages, download
   candles or clone anything. This keeps the answer cheap.
2. Answer in Russian, briefly: price now, model lean and range, then the track-record verdict.
   If the verdict is "пока шум", say plainly that the forecast is not a trade signal.
3. If `LATEST.md` is older than 36 hours, say the daily job has probably failed and point to
   the Actions tab. Do not invent a fresh forecast.
4. Only open `data/log.csv` when the user asks for history or statistics.

Do not change the model, lookback, horizon, sample count or seed in `forecast/core.py`
without the user asking: it would mix incompatible results in the live log.
