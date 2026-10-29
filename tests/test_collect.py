from tk.collect import MinuteAgg


def trade(t_ms, p, q, seller_aggressor):
    return "xauusdt@aggTrade", dict(T=t_ms, p=str(p), q=str(q), m=seller_aggressor)


def test_minute_rows_from_stream():
    rows = []
    a = MinuteAgg(rows.append)
    t0 = 1_800_000_000_000 - (1_800_000_000_000 % 60000)
    a.on(*trade(t0 + 1000, 4000, 10, False))            # 40k aggressive buy
    a.on(*trade(t0 + 2000, 4002, 20, True))             # 80k aggressive sell, large
    a.on("xauusdt@depth20@500ms", dict(E=t0 + 3000, b=[["3999", "30"]] * 20, a=[["4001", "10"]] * 20))
    a.on("xauusdt@forceOrder", dict(E=t0 + 4000, o=dict(S="SELL", ap="3998", p="3998", z="5", q="5")))
    a.on("xauusdt@markPrice@1s", dict(E=t0 + 5000, p="4001", i="4000", r="0.0001"))
    a.on(*trade(t0 + 61000, 4003, 1, False))            # next minute closes the first
    assert len(rows) == 1
    r = rows[0]
    assert r["open"] == 4000 and r["high"] == 4002 and r["close"] == 4002 and r["trades"] == 2
    assert r["buy_usd"] == 40000 and r["sell_usd"] == 80040 and r["big_sell_usd"] == 80040 and r["big_buy_usd"] == 0
    assert abs(r["book5_imb"] - (3999 * 30 - 4001 * 10) / (3999 * 30 + 4001 * 10)) < 1e-9
    assert r["liq_long_usd"] == 3998 * 5 and r["liq_short_usd"] == 0
    assert abs(r["premium"] - 0.00025) < 1e-12 and r["spread"] == 2
