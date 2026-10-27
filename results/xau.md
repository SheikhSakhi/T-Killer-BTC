# XAUUSDT analyst (`python -m tk.xau [risk_usd]`)

Direction = majority vote of three signals (never a tie):
price vs 20-day average, price vs 50-day average, 20-day change of the US dollar
(FRED DTWEXBGS, value at least 9 days old so nothing unpublished is used).
Exits: stop 2%, target 4%, max 7 days. Size so the stop loses the chosen dollar risk.

Tested (research/xau_factors.py) on PAXGUSDT (Mar 2025–) and XAUUSDT (Dec 2025–), both halves,
non-overlapping trades, R = result / risk:

| Rule | PAXG R 1st / 2nd half | XAU R 1st / 2nd half |
|---|---|---|
| trend 20 days | +0.09 / +0.24 | +0.14 / −0.09 |
| trend 50 days | +0.29 / +0.24 | +0.19 / +0.11 |
| dollar 20 days (9-day lag) | +0.36 / +0.24 | +0.03 / +0.25 |
| **vote of the three** | **+0.27 / +0.33 (t 2.3)** | **+0.19 / +0.19 (t 1.0)** |
| real 10y yield 5/20 days | mixed / negative | mixed |
| COT managed money, contrarian | negative | mixed |
| fade the weekend drift | no effect (hit 48–52%) | no effect |

About 15 rules were tried, so a t of 2.3 is suggestive, not proof. Dollar and trend both have a long
economic record for gold, which is why they were kept.
