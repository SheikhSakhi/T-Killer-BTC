"""Fixed settings shared by research and live trading."""
COINS = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "BNBUSDT", "LINKUSDT", "AVAXUSDT", "SUIUSDT", "ADAUSDT"]
CONTEXT = ["BTCUSDT", "ETHUSDT"]

# Costs (Binance USD-M futures, regular tier): taker 0.05% per side, plus slippage allowance
TAKER_FEE = 0.0005
SLIPPAGE = 0.0002

# Daily cycle in UTC: positions must be flat by FLAT_AT; research runs after
FLAT_AT = "22:30"
SESSIONS = {"asia": "00:00", "europe": "07:00", "us": "13:30"}

# Risk frame for the 40 USDT account (changed only with the owner's consent)
CAPITAL = 40.0
RISK_PER_TRADE = 0.01      # fraction of equity lost if the stop is hit
MAX_LEVERAGE = 3
MAX_OPEN = 1
DAILY_LOSS_STOP = 0.03
TOTAL_DRAWDOWN_STOP = 0.25

# Selection guards (see tk/research.py: candidate_table)
SELECT = dict(train=90, gap_val=30, t_req=2.0, min_train=15, min_val=5, min_coins=3)
TOP_K = 10
