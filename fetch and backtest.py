import time, datetime, pandas as pd, numpy as np, os, warnings, glob, re
from market_data import initialize_api
from data_fetch_helper import fetch_all_option_data

warnings.filterwarnings("ignore")

# ==========================
# CONFIGURATION - Integration with config.py
# ==========================
try:
    from config import *
    # Create Config class with imported parameters
    class Config:
        # Imported from config.py
        SYMBOL = SYMBOL
        QTY = QTY
        VWAP_WINDOW = VWAP_WINDOW
        NIFTY_EXPIRY = NIFTY_EXPIRY
        BANKNIFTY_EXPIRY = BANKNIFTY_EXPIRY
        TIME_INTERVAL = TIME_INTERVAL
        TIME_INTERVAL_STRING = TIME_INTERVAL_STRING
        
        # Stop Loss Configuration
        SL_MODE = getattr(globals(), 'SL_MODE', 'PERCENT')
        SL_PERCENT = getattr(globals(), 'SL_PERCENT', 8)
        SL_POINTS = getattr(globals(), 'SL_POINTS', 50)
        
        # Target Profit Configuration
        TP_MODE = getattr(globals(), 'TP_MODE', 'PERCENT')
        TP_PERCENT = getattr(globals(), 'TP_PERCENT', 60)
        TP_POINTS = getattr(globals(), 'TP_POINTS', 300)
        
        # Trailing Stop Loss Configuration
        TSL_MODE = getattr(globals(), 'TSL_MODE', 'PERCENT')
        TRAILING_SL_PERCENT = getattr(globals(), 'TRAILING_SL_PERCENT', 20)
        TRAILING_SL_POINTS = getattr(globals(), 'TRAILING_SL_POINTS', 100)
        
        EXIT_ON_VWAP_CROSS = EXIT_ON_VWAP_CROSS
        EXIT_ON_EMA_VWAP_REVERSE = EXIT_ON_EMA_VWAP_REVERSE
        MAX_DAILY_LOSS = MAX_DAILY_LOSS
        MAX_LOSS_PER_TRADE = MAX_LOSS_PER_TRADE
        TAKE_PROFIT_PERCENT = TAKE_PROFIT_PERCENT
        NIFTY_STRIKE_STEP = NIFTY_STRIKE_STEP
        BANKNIFTY_STRIKE_STEP = BANKNIFTY_STRIKE_STEP
        NIFTY_OPTION_DISTANCE_STRIKES = NIFTY_OPTION_DISTANCE_STRIKES
        NIFTY_OPTION_DISTANCE_POINTS = NIFTY_OPTION_DISTANCE_POINTS
        BANKNIFTY_OPTION_DISTANCE_STRIKES = BANKNIFTY_OPTION_DISTANCE_STRIKES
        BANKNIFTY_OPTION_DISTANCE_POINTS = BANKNIFTY_OPTION_DISTANCE_POINTS
        LOT_SIZE_NIFTY = LOT_SIZE_NIFTY
        LOT_SIZE_BANKNIFTY = LOT_SIZE_BANKNIFTY
        DATA_FOLDER = DATA_FOLDER
        LOG_FOLDER = LOG_FOLDER
        
        # Additional parameters
        FETCH_DATE = '09-12-2025'
        EMA_PERIOD = 5
        MIN_DATA_POINTS = 50
        COMMISSION = 0.0003
        INITIAL_CAPITAL = 100000
        TIMEFRAME = TIME_INTERVAL  # Use TIME_INTERVAL from config
        DATA_DIR = "data/market_data"
        # Use strike distance from config
        NIFTY_STRIKES_COUNT = NIFTY_OPTION_DISTANCE_STRIKES
        BANKNIFTY_STRIKES_COUNT = BANKNIFTY_OPTION_DISTANCE_STRIKES
except ImportError:
    # Fallback defaults if config.py not found
    class Config:
        SYMBOL = "NIFTY"
        QTY = 1
        VWAP_WINDOW = 40
        NIFTY_EXPIRY = '09DEC25'
        BANKNIFTY_EXPIRY = '30DEC25'
        TIME_INTERVAL = 3
        TIME_INTERVAL_STRING = '3T'
        
        # Stop Loss Configuration
        SL_MODE = 'POINTS'
        SL_PERCENT = 0
        SL_POINTS = 100
        
        # Target Profit Configuration
        TP_MODE = 'POINTS'
        TP_PERCENT = 0
        TP_POINTS = 100
        
        # Trailing Stop Loss Configuration
        TSL_MODE = 'POINTS'
        TRAILING_SL_PERCENT = 0
        TRAILING_SL_POINTS = 50
        
        EXIT_ON_VWAP_CROSS = True
        EXIT_ON_EMA_VWAP_REVERSE = True
        MAX_DAILY_LOSS = 2000
        MAX_LOSS_PER_TRADE = 1000
        TAKE_PROFIT_PERCENT = 0
        NIFTY_STRIKE_STEP = 50
        BANKNIFTY_STRIKE_STEP = 100
        NIFTY_OPTION_DISTANCE_STRIKES = 2
        NIFTY_OPTION_DISTANCE_POINTS = 50
        BANKNIFTY_OPTION_DISTANCE_STRIKES = 2
        BANKNIFTY_OPTION_DISTANCE_POINTS = 100
        LOT_SIZE_NIFTY = 75
        LOT_SIZE_BANKNIFTY = 25
        DATA_FOLDER = "data"
        LOG_FOLDER = "logs"
        FETCH_DATE = '03-12-2025'
        EMA_PERIOD = 5
        MIN_DATA_POINTS = 80
        COMMISSION = 0.0003
        INITIAL_CAPITAL = 100000
        TIMEFRAME = 3
        DATA_DIR = "data/market_data"
        NIFTY_STRIKES_COUNT = 2
        BANKNIFTY_STRIKES_COUNT = 2

# Initialize directories
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.DATA_FOLDER, exist_ok=True)
os.makedirs(Config.LOG_FOLDER, exist_ok=True)

# ===================================================================
# BACKTESTER CLASS
# ===================================================================
class EMACrossVWAPBacktester:
    def __init__(self, data_dir=Config.DATA_DIR, timeframe=Config.TIMEFRAME):
        self.data_dir = data_dir; self.results = []; self.all_trades = []; self.timeframe = timeframe
    
    def detect_trend(self, df):
        df["trend"] = np.where(df["ema"] > df["vwap"], "BULLISH", "BEARISH"); return df
    
    def is_bullish_crossover(self, prev, curr):
        return (prev["ema"] <= prev["vwap"] and curr["ema"] > curr["vwap"] and curr["close"] > curr["vwap"] and curr["close"] > curr["ema"])
    
    def is_bearish_crossover(self, prev, curr):
        return (prev["ema"] >= prev["vwap"] and curr["ema"] < curr["vwap"] and curr["close"] < curr["vwap"] and curr["close"] < curr["ema"])
    
    def calculate_vwap(self, df):
        df = df.copy()
        if "volume" not in df.columns: df["volume"] = 1
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["date"] = pd.to_datetime(df["datetime"]).dt.date
        df.loc[df["volume"] == 0, "volume"] = 1
        df["tpv"] = df["typical_price"] * df["volume"]
        df["cum_vol"] = df.groupby("date")["volume"].cumsum()
        df["cum_tpv"] = df.groupby("date")["tpv"].cumsum()
        df["vwap"] = df["cum_tpv"] / df["cum_vol"]
        df["vwap"] = df["vwap"].ffill().bfill()
        df.drop(["tpv", "cum_vol", "cum_tpv", "date", "typical_price"], axis=1, inplace=True, errors="ignore")
        return df
    
    def calculate_ema(self, df):
        df["ema"] = df["close"].ewm(span=Config.EMA_PERIOD, adjust=False).mean()
        df["ema"] = df["ema"].bfill()
        return df
    
    def calculate_indicators(self, df):
        return self.detect_trend(self.calculate_ema(self.calculate_vwap(df)))
    
    def resample_data(self, df):
        if self.timeframe == 1: return df.copy()
        df = df.set_index("datetime")
        df_resampled = df.resample(f"{self.timeframe}min").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        df_resampled = df_resampled.dropna().reset_index()
        return df_resampled
    
    def parse_option_symbol(self, filename):
        basename = os.path.basename(filename).replace(".csv", "")
        patterns = [r"^(NIFTY|BANKNIFTY)(\d{2}[A-Z]{3}\d{2})([CP])(\d+)$", r"^(NIFTY|BANKNIFTY)[_-]?(\d{2}[A-Z]{3}\d{2})[_-]?([CP])[_-]?(\d+)$"]
        for p in patterns:
            m = re.match(p, basename.upper())
            if m: return m.group(1), m.group(2), m.group(3), int(m.group(4))
        return None, None, None, None
    
    def load_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
            required = ["open", "high", "low", "close"]
            for c in required:
                if c not in df.columns:
                    if c.upper() in df.columns: df[c] = df[c.upper()]
                    else: return None
            dt_cols = ["datetime", "time", "timestamp", "date", "DateTime", "Time"]
            dt_col = None
            for c in dt_cols:
                if c in df.columns: dt_col = c; break
            if dt_col: df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
            else: df["datetime"] = pd.date_range("2024-01-01 09:15", periods=len(df), freq="1min")
            df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            if "volume" not in df.columns: df["volume"] = 1
            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)
            return self.resample_data(df)
        except: return None
    
    def simulate_trades(self, df, base, expiry, option_type, strike, lot_size):
        trades = []; position = None
        for i in range(1, len(df)):
            prev = df.iloc[i-1]; curr = df.iloc[i]
            if pd.isna(prev["ema"]) or pd.isna(prev["vwap"]) or pd.isna(curr["ema"]) or pd.isna(curr["vwap"]): continue
            if position is None:
                if option_type == "C":
                    if self.is_bullish_crossover(prev, curr):
                        position = self.create_position(curr, lot_size, base, expiry, strike, "CALL")
                elif option_type == "P":
                    if self.is_bearish_crossover(prev, curr):
                        position = self.create_position(curr, lot_size, base, expiry, strike, "PUT")
            if position is not None:
                exit_flag, exit_price, reason = self.check_exit(position, curr, i, df)
                if exit_flag:
                    trades.append(self.create_trade_record(position, curr, exit_price, reason, base, strike))
                    position = None
        if position is not None:
            last = df.iloc[-1]
            trades.append(self.create_trade_record(position, last, last["close"], "FORCE_CLOSE", base, strike))
        return trades
    
    def create_position(self, candle, lot_size, base, expiry, strike, opt_type):
        entry_price = candle["close"]
        
        # Calculate SL based on mode
        if Config.SL_MODE == 0 or Config.SL_MODE == '0':
            sl_price = entry_price * 0.5  # Default fallback
        elif Config.SL_MODE == 'POINTS':
            sl_price = entry_price - Config.SL_POINTS
        else:  # 'PERCENT'
            sl_price = entry_price * (1 - Config.SL_PERCENT/100)
        
        # Calculate TP based on mode
        if Config.TP_MODE == 0 or Config.TP_MODE == '0':
            tp_price = entry_price * 2  # Default fallback
        elif Config.TP_MODE == 'POINTS':
            tp_price = entry_price + Config.TP_POINTS
        else:  # 'PERCENT'
            tp_price = entry_price * (1 + Config.TP_PERCENT/100)
        
        # Multiply lot size by QTY from config
        total_qty = lot_size * Config.QTY
        return {
            "entry_time": candle["datetime"], "entry_price": entry_price, "entry_high": candle["high"], "entry_low": candle["low"],
            "qty": total_qty, "symbol": f"{base}{expiry}{opt_type[0]}{strike}", "option_type": opt_type, "initial_sl": sl_price,
            "tp_price": tp_price, "highest_price": entry_price, "lowest_price": entry_price, "current_sl": sl_price,
            "max_mtm": 0, "min_mtm": 0, "has_sl": (Config.SL_MODE != 0 and Config.SL_MODE != '0'),
            "has_trailing_sl": (Config.TSL_MODE != 0 and Config.TSL_MODE != '0')
        }
    
    def check_exit(self, position, candle, idx, df):
        high = candle["high"]; low = candle["low"]; close = candle["close"]
        position["highest_price"] = max(position["highest_price"], high)
        position["lowest_price"] = min(position["lowest_price"], low)
        position["max_mtm"] = max(position["max_mtm"], (position["highest_price"] - position["entry_price"]) * position["qty"])
        position["min_mtm"] = min(position["min_mtm"], (position["lowest_price"] - position["entry_price"]) * position["qty"])
        
        # Update Trailing SL based on mode
        if position["has_trailing_sl"]:
            if Config.TSL_MODE == 'POINTS':
                trailing_sl = position["highest_price"] - Config.TRAILING_SL_POINTS
            else:  # 'PERCENT'
                trailing_sl = position["highest_price"] * (1 - Config.TRAILING_SL_PERCENT/100)
            position["current_sl"] = max(position["current_sl"], trailing_sl)
        
        # Check SL hit
        if position["has_sl"] and low <= position["current_sl"]: 
            return True, position["current_sl"], "SL"
        
        # Check TP hit
        if high >= position["tp_price"]: 
            return True, position["tp_price"], "TP"
        
        # VWAP exit rules
        if Config.EXIT_ON_VWAP_CROSS:
            if position["option_type"] == "CALL" and close < candle["vwap"]: 
                return True, close, "VWAP_CROSS"
            if position["option_type"] == "PUT" and close > candle["vwap"]: 
                return True, close, "VWAP_CROSS"
        
        # EMA VWAP reverse exit
        if Config.EXIT_ON_EMA_VWAP_REVERSE:
            if position["option_type"] == "CALL" and candle["ema"] < candle["vwap"]: 
                return True, close, "EMA_VWAP_REVERSE"
            if position["option_type"] == "PUT" and candle["ema"] > candle["vwap"]: 
                return True, close, "EMA_VWAP_REVERSE"
        
        # End of day exit
        if idx == len(df)-1 or candle["datetime"].date() != df.iloc[idx+1]["datetime"].date(): 
            return True, close, "EOD"
        
        return False, None, None
    
    def create_trade_record(self, position, exit_candle, exit_price, reason, base, strike):
        pnl = (exit_price - position["entry_price"]) * position["qty"]
        pnl -= (position["entry_price"] + exit_price) * position["qty"] * Config.COMMISSION
        pnl_pct = ((exit_price / position["entry_price"]) - 1) * 100
        drawdown = position["min_mtm"]
        return {
            "symbol": position["symbol"], "option_type": position["option_type"], "entry_time": position["entry_time"],
            "exit_time": exit_candle["datetime"], "entry_price": round(position["entry_price"], 2), "exit_price": round(exit_price, 2),
            "qty": position["qty"], "pnl": round(pnl, 2), "pnl_percent": round(pnl_pct, 2), "exit_reason": reason,
            "holding_minutes": (exit_candle["datetime"] - position["entry_time"]).total_seconds() / 60, "base": base, "strike": strike,
            "max_mtm": round(position["max_mtm"], 2), "min_mtm": round(position["min_mtm"], 2), "drawdown": round(drawdown, 2),
            "max_gain": round(position["max_mtm"], 2), "max_loss": round(position["min_mtm"], 2)
        }
    
    def calculate_statistics(self, trades):
        if not trades: return {"total_trades":0,"total_pnl":0,"winning_trades":0,"losing_trades":0,"win_rate":0,"avg_win":0,"avg_loss":0,"profit_factor":0,"max_win":0,"max_loss":0,"avg_mtm":0,"avg_drawdown":0}
        wins = [t for t in trades if t["pnl"] > 0]; losses = [t for t in trades if t["pnl"] < 0]
        total_pnl = sum(t["pnl"] for t in trades); win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0; avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        total_win = sum(t["pnl"] for t in wins); total_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = total_win / total_loss if total_loss > 0 else 999.99
        avg_mtm = np.mean([t["max_mtm"] for t in trades]) if trades else 0
        avg_drawdown = np.mean([t["drawdown"] for t in trades]) if trades else 0
        return {
            "total_trades": len(trades), "total_pnl": round(total_pnl, 2), "winning_trades": len(wins), "losing_trades": len(losses),
            "win_rate": round(win_rate, 2), "avg_win": round(avg_win, 2), "avg_loss": round(avg_loss, 2), "profit_factor": round(profit_factor, 2),
            "max_win": round(max([t["pnl"] for t in wins]) if wins else 0, 2), "max_loss": round(min([t["pnl"] for t in losses]) if losses else 0, 2),
            "avg_mtm": round(avg_mtm, 2), "avg_drawdown": round(avg_drawdown, 2)
        }
    
    def display_file_processing_dashboard(self):
        print("\n" + "="*100); print("CSV FILES PROCESSING DASHBOARD"); print("="*50)
        print(f"Date: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
        print(f"Timeframe: {Config.TIMEFRAME} minutes")
        print(f"SL: {Config.SL_MODE} ({Config.SL_PERCENT}% or {Config.SL_POINTS}pts) | TP: {Config.TP_MODE} ({Config.TP_PERCENT}% or {Config.TP_POINTS}pts) | TSL: {Config.TSL_MODE} ({Config.TRAILING_SL_PERCENT}% or {Config.TRAILING_SL_POINTS}pts)")
        print("="*100)
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        csv_files = [f for f in csv_files if "_trades" not in f.lower() and "_combined" not in f.lower()]
        csv_files = sorted(csv_files)
        print(f"\n{'Sr.No':<5} {'Instrument/CSV':<40} {'Pstatus':<10} {'Trade':<6} {'Entry':<6} {'Exit':<6} {'PNL':<8} {'WinRate':<8} {'Time Frame':<10}")
        print("-"*100)
        for idx, filepath in enumerate(csv_files, 1):
            filename = os.path.basename(filepath)
            try:
                df = self.load_data(filepath)
                if df is not None and len(df) >= Config.MIN_DATA_POINTS:
                    df = self.calculate_indicators(df)
                    base, expiry, option_type, strike = self.parse_option_symbol(filepath)
                    if base:
                        lot_size = Config.LOT_SIZE_BANKNIFTY if base == "BANKNIFTY" else Config.LOT_SIZE_NIFTY
                        trades = self.simulate_trades(df, base, expiry, option_type, strike, lot_size)
                        stats = self.calculate_statistics(trades)
                        print(f"{idx:<5} {filename:<40} {'OK':<10} {stats['total_trades']:<6} {stats['total_trades']:<6} {stats['total_trades']:<6} {stats['total_pnl']:<8.2f} {stats['win_rate']:<8.2f} {Config.TIMEFRAME:<10}")
                    else: print(f"{idx:<5} {filename:<40} {'Not OK':<10} {'Na':<6} {'Na':<6} {'Na':<6} {0:<8.2f} {0:<8.2f} {Config.TIMEFRAME:<10}")
                else: print(f"{idx:<5} {filename:<40} {'Not OK':<10} {'Na':<6} {'Na':<6} {'Na':<6} {0:<8.2f} {0:<8.2f} {Config.TIMEFRAME:<10}")
            except: print(f"{idx:<5} {filename:<40} {'Not OK':<10} {'Na':<6} {'Na':<6} {'Na':<6} {0:<8.2f} {0:<8.2f} {Config.TIMEFRAME:<10}")
        print("-"*100)
    
    def generate_detailed_backtest_report(self):
        if not self.all_trades: print("\nNo trades to report!"); return
        print("\n\n" + "="*100); print("CONSOLIDATED BACKTEST REPORT - ALL INSTRUMENTS"); print("="*50)
        print(f"Date: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"); print(f"Timeframe: {self.timeframe} minutes")
        print(f"SL: {Config.SL_MODE} | TP: {Config.TP_MODE} | TSL: {Config.TSL_MODE} | QTY: {Config.QTY}\n")
        header = f"{'Instrument':<25} {'Trend':<7} {'Trade No':<8} {'Entry':<8} {'Time':<8} {'Exit':<8} {'Time':<8} {'PnL':<8} {'Max MTM':<8} {'Drawdown':<10}"
        print(header); print("-"*100)
        for trade in self.all_trades:
            trend = "BULLISH" if trade["option_type"] == "CALL" else "BEARISH"
            entry_time = trade["entry_time"].strftime('%H:%M:%S') if hasattr(trade["entry_time"], 'strftime') else str(trade["entry_time"])
            exit_time = trade["exit_time"].strftime('%H:%M:%S') if hasattr(trade["exit_time"], 'strftime') else str(trade["exit_time"])
            row = f"{str(trade['symbol']):<25} {trend:<7} {str(trade.get('trade_no', '')):<8} {str(trade['entry_price']):<8} {entry_time:<8} {str(trade['exit_price']):<8} {exit_time:<8} {str(trade['pnl']):<8} {str(trade['max_mtm']):<8} {str(trade['drawdown']):<10}"
            print(row)
        print("-"*100); print("\n\n" + "="*80); print("SUMMARY STATISTICS - ALL INSTRUMENTS"); print("="*80)
        total_trades = len(self.all_trades); total_pnl = sum(t["pnl"] for t in self.all_trades)
        wins = [t for t in self.all_trades if t["pnl"] > 0]; losses = [t for t in self.all_trades if t["pnl"] < 0]
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0; avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        total_win = sum(t["pnl"] for t in wins); total_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = total_win / total_loss if total_loss > 0 else 999.99
        avg_mtm = np.mean([t["max_mtm"] for t in self.all_trades]) if self.all_trades else 0
        avg_drawdown = np.mean([t["drawdown"] for t in self.all_trades]) if self.all_trades else 0
        print(f"\nTotal Trades              : {total_trades}"); print(f"Total P&L                : {round(total_pnl, 2)}")
        print(f"Winning Trades           : {len(wins)}"); print(f"Losing Trades            : {len(losses)}")
        print(f"Win Rate (%)             : {round(win_rate, 2)}"); print(f"Average Win              : {round(avg_win, 2)}")
        print(f"Average Loss             : {round(avg_loss, 2)}"); print(f"Profit Factor            : {round(profit_factor, 2)}")
        print(f"Max Win                  : {round(max([t['pnl'] for t in wins]) if wins else 0, 2)}")
        print(f"Max Loss                 : {round(min([t['pnl'] for t in losses]) if losses else 0, 2)}")
        print(f"Average Max MTM          : {round(avg_mtm, 2)}"); print(f"Average Drawdown         : {round(avg_drawdown, 2)}")
        print(f"Total Quantity (QTY)     : {Config.QTY} lots per trade")
        print("\n" + "="*100)
    
    def run_all(self):
        print("\n" + "="*100); print("EMA + VWAP CROSSOVER BACKTEST"); print(f"TIMEFRAME: {self.timeframe} MIN | QTY: {Config.QTY}"); 
        print(f"SL: {Config.SL_MODE} | TP: {Config.TP_MODE} | TSL: {Config.TSL_MODE}"); print("="*100)
        self.display_file_processing_dashboard()
        print("\n" + "="*100); print("RUNNING DETAILED BACKTEST..."); print("="*100)
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        csv_files = [f for f in csv_files if "_trades" not in f.lower() and "_combined" not in f.lower()]
        csv_files = sorted(csv_files); trade_counter = 1
        for filepath in csv_files:
            base, expiry, option_type, strike = self.parse_option_symbol(filepath)
            if not base: continue
            df = self.load_data(filepath)
            if df is None or len(df) < Config.MIN_DATA_POINTS: continue
            df = self.calculate_indicators(df)
            lot_size = Config.LOT_SIZE_BANKNIFTY if base == "BANKNIFTY" else Config.LOT_SIZE_NIFTY
            trades = self.simulate_trades(df, base, expiry, option_type, strike, lot_size)
            for idx, trade in enumerate(trades, 1):
                trade["trade_no"] = trade_counter; self.all_trades.append(trade); trade_counter += 1
            self.results.append({"file": filepath, **self.calculate_statistics(trades)})
        self.generate_detailed_backtest_report()
        print("\nBacktest Complete!")

# ===================================================================
# DATA FETCHING FUNCTIONS - Using Config parameters correctly
# ===================================================================
def search_specific_contract(api, symbol):
    resp = api.searchscrip(exchange='NFO', searchtext=symbol)
    if not resp or resp.get('stat') != 'Ok': return None, None
    for v in resp.get('values', []):
        tsym = v.get('tsym', '')
        if tsym == symbol: return v.get('token'), tsym
    return None, None

def get_time(time_string):
    data = time.strptime(time_string, '%d-%m-%Y %H:%M:%S')
    return time.mktime(data)

def fetch_option_data():
    api = initialize_api()
    if not api: raise SystemExit("API initialization failed")
    
    # Get today's date
    today = datetime.date.today()
    fetch_date = today.strftime('%d-%m-%Y')
    
    start_datetime = f"{fetch_date} 09:15:00"
    end_datetime = f"{fetch_date} 15:30:00"
    
    # Use the unified fetch helper
    result = fetch_all_option_data(
        api,
        Config.NIFTY_EXPIRY,
        Config.BANKNIFTY_EXPIRY,
        Config.NIFTY_STRIKES_COUNT,
        Config.BANKNIFTY_STRIKES_COUNT,
        Config.NIFTY_STRIKE_STEP,
        Config.BANKNIFTY_STRIKE_STEP,
        start_datetime,
        end_datetime,
        Config.DATA_DIR,
        verbose=True
    )

# ===================================================================
# MAIN FUNCTION
# ===================================================================
def main():
    print("\n" + "="*100); print("COMBINED DATA FETCH + EMA+VWAP BACKTESTING"); print("="*100)
    print(f"SYMBOL: {Config.SYMBOL} | QTY: {Config.QTY} | TIME INTERVAL: {Config.TIME_INTERVAL} min")
    print(f"SL: {Config.SL_MODE} ({'Disabled' if Config.SL_MODE == 0 or Config.SL_MODE == '0' else f'{Config.SL_PERCENT}%' if Config.SL_MODE == 'PERCENT' else f'{Config.SL_POINTS} pts'}) | ", end='')
    print(f"TP: {Config.TP_MODE} ({'Disabled' if Config.TP_MODE == 0 or Config.TP_MODE == '0' else f'{Config.TP_PERCENT}%' if Config.TP_MODE == 'PERCENT' else f'{Config.TP_POINTS} pts'}) | ", end='')
    print(f"TSL: {Config.TSL_MODE} ({'Disabled' if Config.TSL_MODE == 0 or Config.TSL_MODE == '0' else f'{Config.TRAILING_SL_PERCENT}%' if Config.TSL_MODE == 'PERCENT' else f'{Config.TRAILING_SL_POINTS} pts'})")
    print(f"NIFTY Strikes: ±{Config.NIFTY_STRIKES_COUNT} | BANKNIFTY Strikes: ±{Config.BANKNIFTY_STRIKES_COUNT}")
    print("="*100)
    
    fetch_choice = input("\nFetch new option data? (y/n): ").lower()
    if fetch_choice == 'y': fetch_option_data()
    
    backtest_choice = input("\nRun EMA+VWAP backtest? (y/n): ").lower()
    if backtest_choice == 'y':
        bt = EMACrossVWAPBacktester(data_dir=Config.DATA_DIR, timeframe=Config.TIMEFRAME)
        bt.run_all()
    
    print("\n" + "="*100); print("PROGRAM COMPLETED"); print("="*100)

if __name__ == "__main__": main()