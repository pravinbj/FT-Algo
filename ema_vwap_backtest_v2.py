"""
EMA9 + VWAP Crossover Backtest - Enhanced Version
Features:
- Trend-based trading (Bullish/Bearish only)
- Custom time intervals (1, 2, 3, 4, 5 min)
- Optional SL/Trailing SL (0 = disabled)
- Max/Min MTM per trade
- Drawdown tracking
- Consolidated report with proper formatting
"""

import pandas as pd
import numpy as np
import os
import glob
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ==========================
# CONFIG
# ==========================

SL_PERCENT = 0.08
TP_PERCENT = 1.0
TRAILING_SL_PERCENT = 0.6

EMA_PERIOD = 5
MIN_DATA_POINTS = 40

LOT_SIZE_NIFTY = 75
LOT_SIZE_BANKNIFTY = 25

COMMISSION = 0.0003
INITIAL_CAPITAL = 100000

TIMEFRAME = 3

# ===================================================================
# BACKTESTER CLASS
# ===================================================================
class EMACrossVWAPBacktester:

    def __init__(self, data_dir="data/market_data", timeframe=1):
        self.data_dir = data_dir
        self.results = []
        self.all_trades = []
        self.timeframe = timeframe

    # ========================
    # TREND DETECTION
    # ========================

    def detect_trend(self, df):
        df = df.copy()
        df["trend"] = np.where(df["ema"] > df["vwap"], "BULLISH", "BEARISH")
        return df

    def is_bullish_crossover(self, prev, curr):
        return (
            prev["ema"] <= prev["vwap"] and
            curr["ema"] > curr["vwap"] and
            curr["close"] > curr["vwap"] and
            curr["close"] > curr["ema"]
        )

    def is_bearish_crossover(self, prev, curr):
        return (
            prev["ema"] >= prev["vwap"] and
            curr["ema"] < curr["vwap"] and
            curr["close"] < curr["vwap"] and
            curr["close"] < curr["ema"]
        )

    # ========================
    # Data & Indicators
    # ========================

    def calculate_vwap(self, df):
        df = df.copy()
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

        if "volume" not in df.columns:
            df["volume"] = 1

        df.loc[df["volume"] == 0, "volume"] = 1
        df["date"] = pd.to_datetime(df["datetime"]).dt.date

        df["tpv"] = df["typical_price"] * df["volume"]
        df["cum_vol"] = df.groupby("date")["volume"].cumsum()
        df["cum_tpv"] = df.groupby("date")["tpv"].cumsum()

        df["vwap"] = df["cum_tpv"] / df["cum_vol"]
        df["vwap"] = df["vwap"].ffill().bfill()

        df.drop(["tpv", "cum_vol", "cum_tpv", "date", "typical_price"], axis=1, inplace=True)
        return df

    def calculate_ema(self, df):
        df = df.copy()
        df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
        df["ema"] = df["ema"].bfill()
        return df

    def calculate_indicators(self, df):
        df = self.calculate_vwap(df)
        df = self.calculate_ema(df)
        df = self.detect_trend(df)
        return df

    # ========================
    # Resample Data
    # ========================

    def resample_data(self, df):
        if self.timeframe == 1:
            return df.copy()

        df = df.set_index("datetime")
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        df_resampled = df.resample(f"{self.timeframe}min").agg(agg_dict)
        df_resampled = df_resampled.dropna()
        df_resampled = df_resampled.reset_index()
        return df_resampled

    # ========================
    # File Parsing
    # ========================

    def parse_option_symbol(self, filename):
        basename = os.path.basename(filename).replace(".csv", "")

        patterns = [
            r"^(NIFTY|BANKNIFTY)(\d{2}[A-Z]{3}\d{2})([CP])(\d+)$",
            r"^(NIFTY|BANKNIFTY)[_-]?(\d{2}[A-Z]{3}\d{2})[_-]?([CP])[_-]?(\d+)$"
        ]

        for p in patterns:
            m = re.match(p, basename.upper())
            if m:
                return m.group(1), m.group(2), m.group(3), int(m.group(4))

        return None, None, None, None

    # ========================
    # Data Loader
    # ========================

    def load_data(self, filepath):
        try:
            df = pd.read_csv(filepath)

            required = ["open", "high", "low", "close"]
            for c in required:
                if c not in df.columns:
                    if c.upper() in df.columns:
                        df[c] = df[c.upper()]
                    else:
                        return None

            dt_cols = ["datetime", "time", "timestamp", "date", "DateTime", "Time"]
            dt_col = None
            for c in dt_cols:
                if c in df.columns:
                    dt_col = c
                    break

            if dt_col:
                df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
            else:
                df["datetime"] = pd.date_range("2024-01-01 09:15", periods=len(df), freq="1min")

            df = df.dropna(subset=["datetime"])
            df = df.sort_values("datetime").reset_index(drop=True)

            if "volume" not in df.columns:
                df["volume"] = 1

            df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].apply(pd.to_numeric)

            df = self.resample_data(df)

            return df

        except Exception as e:
            print(f"     Error loading: {e}")
            return None

    # ===================================================================
    # TRADING SIMULATION
    # ===================================================================

    def simulate_trades(self, df, base, expiry, option_type, strike, lot_size):
        trades = []
        position = None

        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]

            if pd.isna(prev["ema"]) or pd.isna(prev["vwap"]) or pd.isna(curr["ema"]) or pd.isna(curr["vwap"]):
                continue

            # ENTRY LOGIC
            if position is None:

                if option_type == "C":
                    if self.is_bullish_crossover(prev, curr):
                        position = self.create_position(curr, lot_size, base, expiry, strike, "CALL")

                elif option_type == "P":
                    if self.is_bearish_crossover(prev, curr):
                        position = self.create_position(curr, lot_size, base, expiry, strike, "PUT")

            # EXIT LOGIC
            if position is not None:
                exit_flag, exit_price, reason = self.check_exit(position, curr, i, df)

                if exit_flag:
                    trade = self.create_trade_record(position, curr, exit_price, reason, base, strike)
                    trades.append(trade)
                    position = None

        # Force close last position
        if position is not None:
            last = df.iloc[-1]
            trade = self.create_trade_record(position, last, last["close"], "FORCE_CLOSE", base, strike)
            trades.append(trade)

        return trades

    # ===================================================================
    # Position & Exit Logic
    # ===================================================================

    def create_position(self, candle, qty, base, expiry, strike, opt_type):
        entry_price = candle["close"]

        sl_price = None
        if SL_PERCENT > 0:
            sl_price = entry_price * (1 - SL_PERCENT)
        else:
            sl_price = entry_price * 0.5

        tp_price = entry_price * (1 + TP_PERCENT)

        return {
            "entry_time": candle["datetime"],
            "entry_price": entry_price,
            "entry_high": candle["high"],
            "entry_low": candle["low"],
            "qty": qty,
            "symbol": f"{base}{expiry}{opt_type[0]}{strike}",
            "option_type": opt_type,
            "initial_sl": sl_price,
            "tp_price": tp_price,
            "highest_price": entry_price,
            "lowest_price": entry_price,
            "current_sl": sl_price,
            "max_mtm": 0,
            "min_mtm": 0,
            "has_sl": SL_PERCENT > 0,
            "has_trailing_sl": TRAILING_SL_PERCENT > 0 and SL_PERCENT > 0
        }

    def check_exit(self, position, candle, idx, df):

        high = candle["high"]
        low = candle["low"]
        close = candle["close"]
        position["highest_price"] = max(position["highest_price"], high)
        position["lowest_price"] = min(position["lowest_price"], low)
        mtm = (close - position["entry_price"]) * position["qty"]
        position["max_mtm"] = max(position["max_mtm"], (position["highest_price"] - position["entry_price"]) * position["qty"])
        position["min_mtm"] = min(position["min_mtm"], (position["lowest_price"] - position["entry_price"]) * position["qty"])
        if position["has_trailing_sl"]:
            trailing_sl = position["highest_price"] * (1 - TRAILING_SL_PERCENT)
            position["current_sl"] = max(position["current_sl"], trailing_sl)
        if position["has_sl"] and low <= position["current_sl"]:
            return True, position["current_sl"], "SL"
        if high >= position["tp_price"]:
            return True, position["tp_price"], "TP"
        if idx == len(df)-1 or candle["datetime"].date() != df.iloc[idx+1]["datetime"].date():
            return True, close, "EOD"
        return False, None, None
    def create_trade_record(self, position, exit_candle, exit_price, reason, base, strike):
        pnl = (exit_price - position["entry_price"]) * position["qty"]
        pnl -= (position["entry_price"] + exit_price) * position["qty"] * COMMISSION
        pnl_pct = ((exit_price / position["entry_price"]) - 1) * 100

        drawdown = position["min_mtm"]

        return {
            "symbol": position["symbol"],
            "option_type": position["option_type"],
            "entry_time": position["entry_time"],
            "exit_time": exit_candle["datetime"],
            "entry_price": round(position["entry_price"], 2),
            "exit_price": round(exit_price, 2),
            "qty": position["qty"],
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl_pct, 2),
            "exit_reason": reason,
            "holding_minutes": (exit_candle["datetime"] - position["entry_time"]).total_seconds() / 60,
            "base": base,
            "strike": strike,
            "max_mtm": round(position["max_mtm"], 2),
            "min_mtm": round(position["min_mtm"], 2),
            "drawdown": round(drawdown, 2),
            "max_gain": round(position["max_mtm"], 2),
            "max_loss": round(position["min_mtm"], 2)
        }

    # ===================================================================
    # STATISTICS
    # ===================================================================
    def calculate_statistics(self, trades):
        if not trades:
            return {
                "total_trades": 0,
                "total_pnl": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "max_win": 0,
                "max_loss": 0,
                "avg_mtm": 0,
                "avg_drawdown": 0
            }

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] < 0]
        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        total_win = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = total_win / total_loss if total_loss > 0 else 999.99
        avg_mtm = np.mean([t["max_mtm"] for t in trades]) if trades else 0
        avg_drawdown = np.mean([t["drawdown"] for t in trades]) if trades else 0

        return {
            "total_trades": len(trades),
            "total_pnl": round(total_pnl, 2),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_win": round(max([t["pnl"] for t in wins]) if wins else 0, 2),
            "max_loss": round(min([t["pnl"] for t in losses]) if losses else 0, 2),
            "avg_mtm": round(avg_mtm, 2),
            "avg_drawdown": round(avg_drawdown, 2)
        }

    # ===================================================================
    # CONSOLIDATED REPORT
    # ===================================================================

    def generate_consolidated_report(self):
        if not self.all_trades:
            print("\nNo trades to report!")
            return

        print("\n\n" + "="*100)
        print("CONSOLIDATED BACKTEST REPORT - ALL INSTRUMENTS")
        print("="*50)
        print(f"Date: {datetime.now().strftime('%d-%m-%Y')}")  
        print(f"Timeframe: {self.timeframe} minutes")
        print(f"SL: {SL_PERCENT if SL_PERCENT > 0 else 'DISABLED'} | TSL: {TRAILING_SL_PERCENT if TRAILING_SL_PERCENT > 0 else 'DISABLED'}\n")

        # Header with proper spacing
        header = f"{'Instrument':<25} {'Trend':<5} {'Trade No':<8} {'Entry':<8} {'Time':<8} {'Exit':<8} {'Time':<8} {'PnL':<8} {'Max MTM':<8} {'Drawdown':<10}"
        print(header)
        print("-"*80)

        for trade in self.all_trades:
            trend = "BULLISH" if trade["option_type"] == "CALL" else "BEARISH"
            entry_time = trade["entry_time"].strftime('%H:%M:%S') if hasattr(trade["entry_time"], 'strftime') else str(trade["entry_time"])
            exit_time = trade["exit_time"].strftime('%H:%M:%S') if hasattr(trade["exit_time"], 'strftime') else str(trade["exit_time"])

            row = f"{str(trade['symbol']):<25} {trend:<5} {str(trade.get('trade_no', '')):<8} {str(trade['entry_price']):<8} {entry_time:<8} {str(trade['exit_price']):<8} {exit_time:<8} {str(trade['pnl']):<8} {str(trade['max_mtm']):<8} {str(trade['drawdown']):<10}"
            print(row)

        print("-"*80)

        # Summary Statistics
        print("\n\n" + "="*80)
        print("SUMMARY STATISTICS - ALL INSTRUMENTS")
        print("="*80)

        total_trades = len(self.all_trades)
        total_pnl = sum(t["pnl"] for t in self.all_trades)
        wins = [t for t in self.all_trades if t["pnl"] > 0]
        losses = [t for t in self.all_trades if t["pnl"] < 0]

        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

        total_win = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = total_win / total_loss if total_loss > 0 else 999.99

        avg_mtm = np.mean([t["max_mtm"] for t in self.all_trades]) if self.all_trades else 0
        avg_drawdown = np.mean([t["drawdown"] for t in self.all_trades]) if self.all_trades else 0

        print(f"\nTotal Trades              : {total_trades}")
        print(f"Total P&L                : {round(total_pnl, 2)}")
        print(f"Winning Trades           : {len(wins)}")
        print(f"Losing Trades            : {len(losses)}")
        print(f"Win Rate (%)             : {round(win_rate, 2)}")
        print(f"Average Win              : {round(avg_win, 2)}")
        print(f"Average Loss             : {round(avg_loss, 2)}")
        print(f"Profit Factor            : {round(profit_factor, 2)}")
        print(f"Max Win                  : {round(max([t['pnl'] for t in wins]) if wins else 0, 2)}")
        print(f"Max Loss                 : {round(min([t['pnl'] for t in losses]) if losses else 0, 2)}")
        print(f"Average Max MTM          : {round(avg_mtm, 2)}")
        print(f"Average Drawdown         : {round(avg_drawdown, 2)}")

        print("\n" + "="*100)

    # ===================================================================
    # MAIN RUN
    # ===================================================================

    def run_all(self, export_csv=False):

        print("\n" + "="*100)
        print("EMA9 + VWAP CROSSOVER BACKTEST - TREND-BASED OPTION BUYING")
        print(f"TIMEFRAME: {self.timeframe} MIN")
        print("="*100)

        search_paths = [
            self.data_dir,
            "data",
            "data/market_data",
            ".",
        ]

        csv_files = []

        print("\nScanning for CSV files...\n")

        for path in search_paths:
            if os.path.exists(path):
                found = glob.glob(os.path.join(path, "*.csv"))
                if found:
                    print(f"Found {len(found)} file(s) in: {path}")
                    csv_files.extend(found)
            else:
                os.makedirs(path, exist_ok=True)

        csv_files = [f for f in csv_files if "_trades" not in f.lower()]
        csv_files = sorted(list(set(csv_files)))

        if not csv_files:
            print("\nNo CSV files found!")
            return

        print(f"\nTotal CSV files to process: {len(csv_files)}\n")

        trade_counter = 1

        for filepath in csv_files:
            print("-" * 100)
            print(f"Processing: {os.path.basename(filepath)}")

            base, expiry, option_type, strike = self.parse_option_symbol(filepath)
            if not base:
                print("Skipping non-option file")
                continue

            df = self.load_data(filepath)
            if df is None or len(df) < MIN_DATA_POINTS:
                print(f"Invalid or insufficient data")
                continue

            df = self.calculate_indicators(df)

            lot_size = LOT_SIZE_BANKNIFTY if base == "BANKNIFTY" else LOT_SIZE_NIFTY
            trades = self.simulate_trades(df, base, expiry, option_type, strike, lot_size)

            stats = self.calculate_statistics(trades)

            # Add trade numbers
            for idx, trade in enumerate(trades, 1):
                trade["trade_no"] = trade_counter
                self.all_trades.append(trade)
                trade_counter += 1

            print(f"Trades found: {len(trades)} | Total P&L: {stats['total_pnl']} | Win Rate: {stats['win_rate']}%")

            self.results.append({
                "file": filepath,
                **stats
            })

        # Generate consolidated report
        self.generate_consolidated_report()

        print("\nBacktest Complete!")
# ===================================================================
# MAIN
# ===================================================================
def main():
    timeframe = TIMEFRAME
    export_trades = False

    bt = EMACrossVWAPBacktester(timeframe=timeframe)
    bt.run_all(export_csv=export_trades)

if __name__ == "__main__":
    main()