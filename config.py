# Flattrade API Credentials
API_KEY = "XXXXXXXXXXXX"    # Enter your details
CLIENT_ID = "FXXXXX" # Enter your Credentials
PASSWORD = "NokiXXXX" # Enter your Credentials
TWO_FA = "M75673N45K723MDC5GNR4J4Q2O6377NP" # Enter your Credentials
API_SECRET = "2023.172372dbffd440bcbea4feed80c54d0381362b59d39f5700" # Enter your Credentials

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36",
    "Referer": "https://auth.flattrade.in/"
}

# Trading Parameters
SYMBOL = "NIFTY"
QTY = 1
VWAP_WINDOW = 40
NIFTY_EXPIRY = '16DEC25'
BANKNIFTY_EXPIRY = '30DEC25'

# Time Interval
TIME_INTERVAL = 3
TIME_INTERVAL_STRING = '3T'

# Strategy Settings - Stop Loss, Target Profit, and Trailing Stop Loss
# MODE: Set to 'PERCENT' for percentage-based, 'POINTS' for absolute points, or 0 to disable
# SL Configuration
SL_MODE = 'POINTS'  # 'PERCENT', 'POINTS', or 0 to disable
SL_PERCENT = 0       # Stop Loss percentage (if SL_MODE = 'PERCENT')
SL_POINTS = 100       # Stop Loss points (if SL_MODE = 'POINTS')

# Target Profit Configuration
TP_MODE = 'POINTS'  # 'PERCENT', 'POINTS', or 0 to disable
TP_PERCENT = 0      # Target Profit percentage (if TP_MODE = 'PERCENT')
TP_POINTS = 100      # Target Profit points (if TP_MODE = 'POINTS')

# Trailing Stop Loss Configuration
TSL_MODE = 'POINTS'  # 'PERCENT', 'POINTS', or 0 to disable
TRAILING_SL_PERCENT = 0  # Trailing SL percentage (if TSL_MODE = 'PERCENT')
TRAILING_SL_POINTS = 60  # Trailing SL points (if TSL_MODE = 'POINTS')

# Exit Rules
EXIT_ON_VWAP_CROSS = True
EXIT_ON_EMA_VWAP_REVERSE = False

# Risk Management
MAX_DAILY_LOSS = 2000
MAX_LOSS_PER_TRADE = 1000
TAKE_PROFIT_PERCENT = 70

# WebSocket
SOCKET_URL = "wss://ws.flattrade.in/NorenWS/"

# Option Settings
NIFTY_STRIKE_STEP = 50
BANKNIFTY_STRIKE_STEP = 100
NIFTY_OPTION_DISTANCE_STRIKES = 1
NIFTY_OPTION_DISTANCE_POINTS = 50
BANKNIFTY_OPTION_DISTANCE_STRIKES = 1
BANKNIFTY_OPTION_DISTANCE_POINTS = 100
STRIKES_COUNT = 1

def get_api():
    from api_adapter import get_adapter
    adapter = get_adapter(API_BACKEND)
    return adapter.initialize()

API_BACKEND = 'noren'

# Market Settings
MARKET_START_TIME = "09:15"
MARKET_END_TIME = "15:30"
TRADING_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Storage
DATA_FOLDER = "data"
LOG_FOLDER = "logs"

# Lot Sizes
LOT_SIZE_NIFTY = 75
LOT_SIZE_BANKNIFTY = 25# In config.py, simply add:
SL_PERCENT_BELOW_VWAP = 1  # Change to 0.5, 1, 1.5, 2, 3, or 4


