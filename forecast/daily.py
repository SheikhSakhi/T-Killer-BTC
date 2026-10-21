"""Daily job: score past forecasts, make today's forecast, rewrite LATEST.md.

Run: python forecast/daily.py
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import core  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(ROOT, "data", "log.csv")
LATEST_MD = os.path.join(ROOT, "LATEST.md")
LATEST_JSON = os.path.join(ROOT, "data", "latest.json")
SAMPLES = int(os.environ.get("SAMPLES", 30))

LOG_COLS = ["made_at", "origin", "target", "price", "p10", "p50", "p90", "up_share",
            "model", "source", "actual", "err_kronos", "err_naive", "dir_ok", "in_band"]


def read_log():
    if os.path.exists(LOG):
        return pd.read_csv(LOG)
    return pd.DataFrame(columns=LOG_COLS)


def score(log, df):
    """Fill in actuals for forecasts whose 24h target candle has closed."""
    closes = dict(zip(df["timestamps"].astype(str), df["close"]))
    for i, r in log.iterrows():
        if pd.notna(r["actual"]) and r["actual"] != "":
            continue
        a = closes.get(str(pd.Timestamp(r["target"])))
        if a is None:
            continue
        log.loc[i, "actual"] = a
        log.loc[i, "err_kronos"] = round(abs(r["p50"] - a), 2)
        log.loc[i, "err_naive"] = round(abs(r["price"] - a), 2)
        log.loc[i, "dir_ok"] = bool(np.sign(r["p50"] - r["price"]) == np.sign(a - r["price"]))
        log.loc[i, "in_band"] = bool(r["p10"] <= a <= r["p90"])
    return log


def stats(log):
    s = log.dropna(subset=["actual"])
    s = s[s["actual"].astype(str) != ""]
    if s.empty:
        return None
    dir_ok = s["dir_ok"].astype(str).str.lower() == "true"
    in_band = s["in_band"].astype(str).str.lower() == "true"
    return dict(n=len(s), dir_acc=dir_ok.mean(), band_cov=in_band.mean(),
                mae_k=s["err_kronos"].astype(float).mean(), mae_n=s["err_naive"].astype(float).mean())


def write_latest(row, st, df):
    hist = df["close"].iloc[-24:]
    chg24 = (hist.iloc[-1] / hist.iloc[0] - 1) * 100
    move = (row["p50"] / row["price"] - 1) * 100
    lean = "вниз" if move < -0.3 else "вверх" if move > 0.3 else "боковик"
    lines = [
        "# BTC: прогноз Kronos на 24 часа",
        "",
        f"_Обновлено: {row['made_at']} UTC. Последняя закрытая свеча: {row['origin']} UTC._",
        "",
        "| | |",
        "|---|---|",
        f"| Цена сейчас | **{row['price']:,.0f}** |",
        f"| За прошлые 24ч | {chg24:+.2f}% (мин {hist.min():,.0f} / макс {hist.max():,.0f}) |",
        f"| Медиана прогноза на {row['target']} | **{row['p50']:,.0f}** ({move:+.2f}%) |",
        f"| Диапазон 10–90% сценариев | {row['p10']:,.0f} – {row['p90']:,.0f} |",
        f"| Сценариев с ростом | {row['up_share']*100:.0f}% |",
        f"| Наклон модели | **{lean}** |",
        "",
        "## Насколько этому верить (живая статистика)",
        "",
    ]
    if st:
        verdict = ("пока шум" if st["n"] < 30 or st["dir_acc"] < 0.58 or st["mae_k"] >= st["mae_n"]
                   else "есть признаки преимущества")
        lines += [
            f"- Прогнозов сверено: **{st['n']}**",
            f"- Угадано направление: **{st['dir_acc']*100:.0f}%**",
            f"- Средняя ошибка: Kronos {st['mae_k']:,.0f} $ vs «цена не изменится» {st['mae_n']:,.0f} $",
            f"- Цена попала в диапазон 10–90%: {st['band_cov']*100:.0f}% (норма 80%)",
            f"- Вердикт: **{verdict}**",
        ]
    else:
        lines += ["- Сверенных прогнозов пока нет."]
    lines += [
        "",
        "Историческая проверка (до запуска): см. `results/backtest.md`.",
        "",
        "> Не торговый совет. Одна точка прогноза — не сигнал. Диапазон модели исторически "
        "слишком узкий: реальный разброс шире.",
        "",
    ]
    with open(LATEST_MD, "w") as f:
        f.write("\n".join(lines))
    with open(LATEST_JSON, "w") as f:
        json.dump(dict(forecast=row, stats=st), f, default=float, indent=1)


def main():
    df, source = core.load_candles(days=25)
    log = score(read_log(), df)

    origin = df["timestamps"].iloc[-1]
    target = origin + pd.Timedelta(hours=core.HORIZON)
    if (log["origin"].astype(str) == str(origin)).any():
        print(f"already forecast from {origin}; scoring only")
        row = log[log["origin"].astype(str) == str(origin)].iloc[-1].to_dict()
    else:
        paths = core.forecast(df, len(df), SAMPLES)
        fin = paths[:, -1]
        price = float(df["close"].iloc[-1])
        row = dict(
            made_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            origin=str(origin), target=str(target), price=price,
            p10=round(float(np.percentile(fin, 10)), 2), p50=round(float(np.median(fin)), 2),
            p90=round(float(np.percentile(fin, 90)), 2), up_share=round(float((fin > price).mean()), 3),
            model=core.MODEL, source=source,
            actual="", err_kronos="", err_naive="", dir_ok="", in_band="")
        log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log[LOG_COLS].to_csv(LOG, index=False)
    write_latest(row, stats(log), df)
    print(open(LATEST_MD).read())


if __name__ == "__main__":
    main()
