"""
Live Trading Dashboard - All Instruments
"""

import time, datetime, pandas as pd, numpy as np, os, threading, warnings, logging, sys
from config import *
from market_data import initialize_api
from data_fetch_helper import fetch_all_option_data

# Conditional imports for platform-specific modules
if os.name == 'nt':
    import msvcrt
else:
    import select
    import tty
    import termios

from typing import Optional, Dict, Any, List

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    SL_PERCENT = 0; TP_PERCENT = 0.60; TRAILING_SL_PERCENT = 0.30; EMA_PERIOD = 5
    ENTRY_DELAY = 30; BACKFILL_DAYS = 3; DATA_DIR = "data/market_data"
    LOT_NIFTY = 75; LOT_BANKNIFTY = 25; COMMISSION = 0.0003
    MAX_DAILY_LOSS = 2000; DASH_REFRESH = 2
    STRIKES_COUNT = 1

class TradingStrategy:
    def __init__(self) -> None:
        self.api: Optional[Any] = None
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.pnl: float = 0
        self.trades: int = 0
        self.last_signals: Dict[str, Optional[str]] = {}  # Track signals per symbol
        self.running: bool = True
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.instruments: List[Dict[str, Any]] = []
        self.instrument_data: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        self.position_counter: int = 1
        self.trade_counter: int = 1

    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 2: return df
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
        
        df.drop(["tpv", "cum_vol", "cum_tpv", "date", "typical_price"], 
                axis=1, inplace=True, errors="ignore")
        return df

    def calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema"] = df["close"].ewm(span=Config.EMA_PERIOD, adjust=False).mean()
        df["ema"] = df["ema"].bfill()
        return df

    def detect_signal(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[float]]:
        if len(df) < 2: return None, None
        prev, curr = df.iloc[-2], df.iloc[-1]
        
        if (prev["ema"] <= prev["vwap"] and
            curr["ema"] > curr["vwap"] and
            curr["close"] > curr["vwap"] and
            curr["close"] > curr["ema"]):
            return "BUY_CALL", curr["close"]
        
        elif (prev["ema"] >= prev["vwap"] and
              curr["ema"] < curr["vwap"] and
              curr["close"] < curr["vwap"] and
              curr["close"] < curr["ema"]):
            return "BUY_PUT", curr["close"]
        
        return None, None

    def get_atm_strikes(self, ltp, underlying, expiry):
        if 'BANK' in underlying.upper():
            strike_step = BANKNIFTY_STRIKE_STEP
            base = 'BANKNIFTY'
            lot = Config.LOT_BANKNIFTY
        else:
            strike_step = NIFTY_STRIKE_STEP
            base = 'NIFTY'
            lot = Config.LOT_NIFTY
        
        atm_strike = int(round(ltp / strike_step) * strike_step)
        
        strikes = []
        for i in range(-Config.STRIKES_COUNT, Config.STRIKES_COUNT + 1):
            strike = atm_strike + (i * strike_step)
            strikes.append(strike)
        
        return strikes, base, lot

    def search_contract(self, symbol: str) -> Optional[str]:
        try:
            resp = self.api.searchscrip(exchange='NFO', searchtext=symbol) if self.api else None
            if resp and resp.get('stat') == 'Ok':
                for v in resp.get('values', []):
                    if v.get('tsym', '').upper() == symbol.upper():
                        return v.get('token')
        except: pass
        return None

    def fetch_open_positions(self):
        """Fetch open positions from the broker"""
        try:
            if not self.api:
                logging.error("API not initialized for fetching positions")
                return []
            
            # Fetch positions from broker
            positions_data = self.api.get_positions()
            if not positions_data or positions_data.get('stat') != 'Ok':
                logging.warning(f"Could not fetch positions: {positions_data}")
                return []
            
            broker_positions = positions_data.get('values', [])
            logging.info(f"Fetched {len(broker_positions)} open positions from broker")
            
            # Print positions for reference
            if broker_positions:
                print("\n" + "="*120)
                print("OPEN POSITIONS FROM BROKER:")
                print("-"*120)
                print(f"{'Symbol':<25} {'Type':<6} {'Qty':<8} {'Avg Price':<12} {'Product':<10}")
                print("-"*120)
                for pos in broker_positions:
                    symbol = pos.get('tsym', 'N/A')
                    qty = int(pos.get('netqty', 0))
                    avg_price = float(pos.get('netavgprc', 0))
                    product = pos.get('prd', 'N/A')
                    
                    # Determine if it's a call or put
                    pos_type = 'CALL' if 'C' in symbol else 'PUT' if 'P' in symbol else 'N/A'
                    
                    print(f"{symbol:<25} {pos_type:<6} {qty:<8} {avg_price:<12.2f} {product:<10}")
                print("="*120 + "\n")
            
            return broker_positions
        except Exception as e:
            logging.error(f"Error fetching open positions: {e}")
            return []


    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        token = self.search_contract(symbol)
        if token and self.api:
            return self.api.get_quotes(exchange='NFO', token=token)
        return None

    def place_order(self, symbol: str, qty: int, action: str = 'B') -> Optional[str]:
        try:
            if not self.api:
                return None
            
            token = self.search_contract(symbol)
            if not token: return None
            
            order = self.api.place_order(
                buy_or_sell=action, product_type='M', exchange='NFO',
                tradingsymbol=symbol, quantity=qty, price_type='MKT', price=0
            )
            
            if order and order.get('stat') == 'Ok':
                order_id = order.get('norenordno')
                logging.info(f"Order {action} placed: {symbol} | Qty: {qty} | ID: {order_id}")
                return order_id
        except: pass
        return None

    def enter_trade(self, signal, symbol, price, lot):
        print(f"\n[ENTRY] Signal: {signal} | Symbol: {symbol} | Price: {price}")
        print(f"Waiting {Config.ENTRY_DELAY}s for confirmation...")
        
        start = datetime.datetime.now()
        while (datetime.datetime.now() - start).seconds < Config.ENTRY_DELAY:
            quote = self.get_quote(symbol)
            if quote:
                curr_price = float(quote.get('lp', price))
                if symbol in self.instrument_data:
                    self.instrument_data[symbol]['close'] = curr_price
                    self.instrument_data[symbol]['ltp'] = curr_price
                remaining = Config.ENTRY_DELAY - (datetime.datetime.now() - start).seconds
                print(f"  Current: {curr_price:.2f} | Time remaining: {remaining}s", end='\r')
            time.sleep(1)
        
        order_id = self.place_order(symbol, lot, 'B')
        if order_id:
            # Calculate SL and TP based on configuration
            if Config.SL_MODE == 'PERCENT' and Config.SL_PERCENT > 0:
                sl = price * (1 - Config.SL_PERCENT / 100)
            elif Config.SL_MODE == 'POINTS' and Config.SL_POINTS > 0:
                sl = price - Config.SL_POINTS
            else:
                sl = price * 0.95  # Default 5% SL
            
            if Config.TP_MODE == 'PERCENT' and Config.TP_PERCENT > 0:
                tp = price * (1 + Config.TP_PERCENT / 100)
            elif Config.TP_MODE == 'POINTS' and Config.TP_POINTS > 0:
                tp = price + Config.TP_POINTS
            else:
                tp = price * 1.05  # Default 5% TP
            
            # Calculate trailing SL reference point
            if Config.TSL_MODE == 'PERCENT' and Config.TRAILING_SL_PERCENT > 0:
                current_sl = sl
            elif Config.TSL_MODE == 'POINTS' and Config.TRAILING_SL_POINTS > 0:
                current_sl = price - Config.TRAILING_SL_POINTS
            else:
                current_sl = sl
            
            pos_id = f"{symbol}_{order_id}"
            self.positions[pos_id] = {
                'trade_no': self.trade_counter,
                'symbol': symbol,
                'type': 'CALL' if signal == 'BUY_CALL' else 'PUT',
                'entry_time': datetime.datetime.now(),
                'entry': price,
                'qty': lot,
                'order_id': order_id,
                'sl': sl,
                'tp': tp,
                'high': price,
                'low': price,
                'current_sl': current_sl,
                'max_mtm': 0,
                'min_mtm': 0,
                'exit_reason': None,
                'exit_price': None,
                'pnl': 0,
                'status': 'OPEN'
            }
            self.trade_counter += 1
            print(f"\n[TRADE ENTERED] {symbol} @ {price} | SL: {sl:.2f} | TP: {tp:.2f}")
            return pos_id
        return None

    def manage_positions(self):
        for pos_id, pos in list(self.positions.items()):
            quote = self.get_quote(pos['symbol'])
            if not quote or quote.get('stat') != 'Ok': continue
            
            curr = float(quote.get('lp', 0))
            high = float(quote.get('h', curr))
            low = float(quote.get('l', curr))
            
            pos['high'] = max(pos['high'], high)
            pos['low'] = min(pos['low'], low)
            pos['max_mtm'] = max(pos['max_mtm'], (pos['high'] - pos['entry']) * pos['qty'])
            pos['min_mtm'] = min(pos['min_mtm'], (pos['low'] - pos['entry']) * pos['qty'])
            
            # Update trailing stop loss based on configuration
            if Config.TSL_MODE == 'PERCENT' and Config.TRAILING_SL_PERCENT > 0:
                trail_sl = pos['high'] * (1 - Config.TRAILING_SL_PERCENT / 100)
                pos['current_sl'] = max(pos['current_sl'], trail_sl)
            elif Config.TSL_MODE == 'POINTS' and Config.TRAILING_SL_POINTS > 0:
                trail_sl = pos['high'] - Config.TRAILING_SL_POINTS
                pos['current_sl'] = max(pos['current_sl'], trail_sl)
            
            exit_price = None; reason = None; status = None
            # Check stop loss
            if low <= pos['current_sl']:
                exit_price = pos['current_sl']; reason = "SL"; status = "SL_HIT"
            # Check target profit
            elif high >= pos['tp']:
                exit_price = pos['tp']; reason = "TP"; status = "TGT_HIT"
            # Check end of day
            elif not self.in_trading_hours():
                exit_price = curr; reason = "EOD"; status = "EOD_CLOSE"
            
            if reason:
                self.close_position(pos_id, reason, exit_price, status)

    def close_position(self, pos_id, reason, price=None, status=None):
        pos = self.positions.get(pos_id)
        if not pos: return
        
        try:
            if price is None:
                quote = self.get_quote(pos['symbol'])
                price = float(quote.get('lp', pos['entry'])) if quote else pos['entry']
            
            self.place_order(pos['symbol'], pos['qty'], 'S')
            
            pnl = (price - pos['entry']) * pos['qty']
            pnl -= (pos['entry'] + price) * pos['qty'] * Config.COMMISSION
            self.pnl += pnl
            
            trade_record = {
                'trade_no': pos['trade_no'],
                'symbol': pos['symbol'],
                'type': pos['type'],
                'entry': pos['entry'],
                'entry_time': pos['entry_time'],
                'exit': price,
                'exit_time': datetime.datetime.now(),
                'qty': pos['qty'],
                'pnl': pnl,
                'mtm': (price - pos['entry']) * pos['qty'],
                'max_mtm': pos['max_mtm'],
                'min_mtm': pos['min_mtm'],
                'reason': reason,
                'status': status or 'CLOSED'
            }
            
            self.closed_trades.append(trade_record)
            if len(self.closed_trades) > 20:
                self.closed_trades = self.closed_trades[-20:]
            
            print(f"\n[EXIT] {pos['symbol']} | Reason: {reason} | PnL: {pnl:.2f}")
            del self.positions[pos_id]
        except Exception as e:
            logging.error(f"Error closing position: {e}")

    def manual_exit(self):
        print("\n" + "="*60)
        print("OPTIONS:")
        print("  1. Exit from tracked positions")
        print("  2. Fetch open positions from broker")
        print("  0. Cancel")
        print("="*60)
        
        try:
            choice = input("\nSelect option (0-2): ")
            
            if choice == '0':
                return
            elif choice == '2':
                # Fetch and display positions from broker
                self.fetch_open_positions()
                return
            elif choice != '1':
                print("Invalid option")
                return
            
            # Option 1: Exit from tracked positions
            if not self.positions:
                print("\nNo open tracked positions to exit")
                return
            
            print("\nOpen Tracked Positions:")
            for pos_id, pos in self.positions.items():
                quote = self.get_quote(pos['symbol'])
                curr = float(quote.get('lp', pos['entry'])) if quote else pos['entry']
                mtm = (curr - pos['entry']) * pos['qty']
                print(f"  {pos['trade_no']}. {pos['symbol']} | Entry: {pos['entry']:.2f} | Current: {curr:.2f} | MTM: {mtm:.2f}")
            
            choice = input("\nEnter trade number to exit (0 to cancel): ")
            if choice == '0': return
            
            for pos_id, pos in self.positions.items():
                if str(pos['trade_no']) == choice:
                    self.close_position(pos_id, "MANUAL", None, "MANUAL_CLOSE")
                    return
            print("Invalid trade number")
        except Exception as e:
            logging.error(f"Error in manual_exit: {e}")

    def initialize_instruments(self):
        self.instruments = []
        
        if not self.api:
            logging.error("Cannot initialize instruments: API not available")
            return
        
        nifty_token = '26000'
        banknifty_token = '26009'
        
        nifty_quote = self.api.get_quotes(exchange='NSE', token=nifty_token)
        banknifty_quote = self.api.get_quotes(exchange='NSE', token=banknifty_token)
        
        if nifty_quote and nifty_quote.get('stat') == 'Ok':
            nifty_ltp = float(nifty_quote.get('lp', 0))
            nifty_strikes, nifty_base, _ = self.get_atm_strikes(
                nifty_ltp, 'NIFTY', NIFTY_EXPIRY
            )
            logging.info(f"NIFTY LTP: {nifty_ltp} | Selected strikes: {nifty_strikes}")
            
            for strike in nifty_strikes:
                for opt_type in ['C', 'P']:
                    symbol = f"{nifty_base}{NIFTY_EXPIRY}{opt_type}{strike}"
                    self.instruments.append({
                        'symbol': symbol,
                        'underlying': 'NIFTY',
                        'strike': strike,
                        'type': opt_type,
                        'lot_size': Config.LOT_NIFTY
                    })
        
        if banknifty_quote and banknifty_quote.get('stat') == 'Ok':
            banknifty_ltp = float(banknifty_quote.get('lp', 0))
            banknifty_strikes, banknifty_base, _ = self.get_atm_strikes(
                banknifty_ltp, 'BANKNIFTY', BANKNIFTY_EXPIRY
            )
            logging.info(f"BANKNIFTY LTP: {banknifty_ltp} | Selected strikes: {banknifty_strikes}")
            
            for strike in banknifty_strikes:
                for opt_type in ['C', 'P']:
                    symbol = f"{banknifty_base}{BANKNIFTY_EXPIRY}{opt_type}{strike}"
                    self.instruments.append({
                        'symbol': symbol,
                        'underlying': 'BANKNIFTY',
                        'strike': strike,
                        'type': opt_type,
                        'lot_size': Config.LOT_BANKNIFTY
                    })
        
        logging.info(f"Initialized {len(self.instruments)} instruments")

    def backfill_historical_data(self):
        """Backfill historical data for all instruments using unified fetch logic"""
        if not self.api:
            logging.error("API not initialized")
            return
        
        logging.info("Starting historical data backfill...")
        
        # Get end time as today 3:30 PM
        end_time = datetime.datetime.combine(datetime.date.today(), datetime.time(15, 30))
        start_time = end_time - datetime.timedelta(days=Config.BACKFILL_DAYS)
        
        # Format timestamps as strings for the helper
        start_datetime = start_time.strftime('%d-%m-%Y %H:%M:%S')
        end_datetime = end_time.strftime('%d-%m-%Y %H:%M:%S')
        
        # Use the unified fetch helper
        result = fetch_all_option_data(
            self.api,
            NIFTY_EXPIRY,
            BANKNIFTY_EXPIRY,
            Config.STRIKES_COUNT,
            Config.STRIKES_COUNT,
            NIFTY_STRIKE_STEP,
            BANKNIFTY_STRIKE_STEP,
            start_datetime,
            end_datetime,
            Config.DATA_DIR,
            verbose=True
        )
        
        # Load fetched data into cache and calculate indicators
        for symbol in result['symbols']:
            filepath = os.path.join(Config.DATA_DIR, f"{symbol}.csv")
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    
                    # Ensure required columns exist
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col not in df.columns:
                            df[col] = 0
                    
                    # Calculate indicators on backfilled data
                    if len(df) >= Config.EMA_PERIOD:
                        df = self.calculate_vwap(df)
                        df = self.calculate_ema(df)
                        
                        # Initialize instrument_data from backfilled data
                        if df.iloc[-1] is not None:
                            last_row = df.iloc[-1]
                            self.instrument_data[symbol] = {
                                'symbol': symbol,
                                'ltp': float(last_row.get('close', 0)) if 'close' in df.columns else 0,
                                'vwap': float(last_row.get('vwap', 0)) if 'vwap' in df.columns else 0,
                                'ema': float(last_row.get('ema', 0)) if 'ema' in df.columns else 0,
                                'signal': None,
                                'volume': int(last_row.get('volume', 0)) if 'volume' in df.columns else 0,
                                'open': float(last_row.get('open', 0)) if 'open' in df.columns else 0,
                                'high': float(last_row.get('high', 0)) if 'high' in df.columns else 0,
                                'low': float(last_row.get('low', 0)) if 'low' in df.columns else 0
                            }
                    
                    self.data_cache[symbol] = df
                    logging.debug(f"Cached {symbol}: {len(df)} rows with indicators")
                except Exception as e:
                    logging.warning(f"Failed to cache {symbol}: {e}")
        
        logging.info(f"Backfill complete: {result['total_success']} successful, {result['total_failed']} failed")

    def update_instrument_data(self, symbol, quote):
        """Update instrument data display with latest quote (without corrupting data_cache)"""
        if not quote or quote.get('stat') != 'Ok': 
            return
        
        try:
            curr = float(quote.get('lp', 0))
            
            # Get current data from cache (backfilled data with indicators)
            if symbol not in self.data_cache or len(self.data_cache[symbol]) < 1:
                return
            
            df = self.data_cache[symbol]
            last_row = df.iloc[-1]
            
            # Update only the display data with current LTP, keep VWAP/EMA from backfilled data
            self.instrument_data[symbol] = {
                'symbol': symbol,
                'ltp': curr,  # Current price from quote
                'vwap': float(last_row.get('vwap', 0)) if 'vwap' in df.columns else 0,  # From backfilled data
                'ema': float(last_row.get('ema', 0)) if 'ema' in df.columns else 0,  # From backfilled data
                'signal': self.instrument_data.get(symbol, {}).get('signal'),  # Keep existing signal
                'volume': int(quote.get('v', 0)),  # Current volume
                'open': float(quote.get('o', curr)),
                'high': float(quote.get('h', curr)),
                'low': float(quote.get('l', curr))
            }
                    
        except Exception as e:
            logging.error(f"Error updating {symbol}: {e}")

    def display_dashboard(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*120)
        print("LIVE TRADING DASHBOARD")
        print("="*50)
        print(f"Date: {datetime.datetime.now().strftime('%d-%m-%Y')}")
        print(f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"Daily P&L: {self.pnl:.2f} | Open Trades: {len(self.positions)} | Total Instruments: {len(self.instruments)}")
        print("="*120)
        
        # ALL INSTRUMENTS - LIVE DATA
        print("\nALL INSTRUMENTS - LIVE DATA:")
        print("-"*120)
        print(f"{'Symbol':<20} {'LTP':<8} {'VWAP':<8} {'EMA':<8} {'Signal':<10} {'Volume':<10}")
        print("-"*120)
        
        sorted_instruments = sorted(self.instrument_data.values(), key=lambda x: x.get('symbol', ''))
        
        for inst in sorted_instruments:
            signal_display = inst.get('signal', '')
            ltp = inst.get('ltp', 0.0)
            vwap = inst.get('vwap', 0.0)
            ema = inst.get('ema', 0.0)
            volume = inst.get('volume', 0)
            symbol = inst.get('symbol', 'N/A')
            
            if signal_display == 'BUY_CALL':
                signal_display = "\033[92mBUY_CALL\033[0m"
            elif signal_display == 'BUY_PUT':
                signal_display = "\033[91mBUY_PUT\033[0m"
            else:
                signal_display = signal_display or ''
            
            print(f"{symbol:<20} {ltp:<8.2f} {vwap:<8.2f} "
                  f"{ema:<8.2f} {signal_display:<10} {volume:<10}")
        
        # OPEN POSITIONS
        print("\n\nOPEN POSITIONS:")
        print("-"*120)
        if self.positions:
            print(f"{'#':<3} {'Symbol':<20} {'Type':<6} {'Entry':<8} {'Entry Time':<10} {'LTP':<8} {'Qty':<6} {'MTM':<8} {'VWAP':<8} {'EMA':<8} {'Signal':<10} {'Status':<12}")
            print("-"*120)
            for pos_id, pos in self.positions.items():
                quote = self.get_quote(pos['symbol'])
                curr = float(quote.get('lp', pos['entry'])) if quote else pos['entry']
                mtm = (curr - pos['entry']) * pos['qty']
                mtm -= (pos['entry'] + curr) * pos['qty'] * Config.COMMISSION
                entry_time = pos['entry_time'].strftime('%H:%M:%S') if hasattr(pos['entry_time'], 'strftime') else str(pos['entry_time'])
                
                # Get current instrument data for VWAP, EMA, Signal
                inst_data = self.instrument_data.get(pos['symbol'], {})
                vwap = inst_data.get('vwap', 0)
                ema = inst_data.get('ema', 0)
                signal = inst_data.get('signal', '')
                
                if signal == 'BUY_CALL':
                    signal = "\033[92mBUY_CALL\033[0m"
                elif signal == 'BUY_PUT':
                    signal = "\033[91mBUY_PUT\033[0m"
                
                print(f"{pos['trade_no']:<3} {pos['symbol']:<20} {pos['type']:<6} "
                      f"{pos['entry']:<8.2f} {entry_time:<10} {curr:<8.2f} {pos['qty']:<6} "
                      f"{mtm:<8.2f} {vwap:<8.2f} {ema:<8.2f} {signal:<10} {'OPEN':<12}")
        else:
            print("No open positions")
        
        # RECENTLY CLOSED TRADES (last 10)
        if self.closed_trades:
            print("\n\nRECENTLY CLOSED TRADES:")
            print("-"*120)
            print(f"{'#':<3} {'Symbol':<20} {'Type':<6} {'Entry':<8} {'Exit':<8} {'Entry Time':<10} {'Exit Time':<10} {'Qty':<6} {'PnL':<8} {'MTM':<8} {'Status':<12}")
            print("-"*120)
            
            for trade in self.closed_trades[-10:]:
                entry_time = trade['entry_time'].strftime('%H:%M:%S') if hasattr(trade['entry_time'], 'strftime') else str(trade['entry_time'])
                exit_time = trade['exit_time'].strftime('%H:%M:%S') if hasattr(trade['exit_time'], 'strftime') else str(trade['exit_time'])
                
                status = trade['status']
                if status == 'SL_HIT':
                    status = "\033[91mSL_HIT\033[0m"
                elif status == 'TGT_HIT':
                    status = "\033[92mTGT_HIT\033[0m"
                elif status == 'EOD_CLOSE':
                    status = "\033[93mEOD_CLOSE\033[0m"
                elif status == 'MANUAL_CLOSE':
                    status = "\033[94mMANUAL_CLOSE\033[0m"
                
                pnl_color = ""
                if trade['pnl'] > 0:
                    pnl_color = "\033[92m"
                elif trade['pnl'] < 0:
                    pnl_color = "\033[91m"
                
                print(f"{trade['trade_no']:<3} {trade['symbol']:<20} {trade['type']:<6} "
                      f"{trade['entry']:<8.2f} {trade['exit']:<8.2f} {entry_time:<10} {exit_time:<10} "
                      f"{trade['qty']:<6} {pnl_color}{trade['pnl']:<8.2f}\033[0m {trade['mtm']:<8.2f} {status:<12}")
        
        print("\n" + "="*120)
        print("Commands: (M) Manual Exit | (Q) Quit")
        print("="*120)

    def update_dashboard_thread(self):
        while self.running:
            try:
                for inst in self.instruments:
                    symbol = inst['symbol']
                    quote = self.get_quote(symbol)
                    if quote:
                        self.update_instrument_data(symbol, quote)
                
                self.display_dashboard()
                time.sleep(Config.DASH_REFRESH)
            except Exception as e:
                logging.error(f"Dashboard error: {e}")
                time.sleep(1)

    def in_trading_hours(self):
        now = datetime.datetime.now().time()
        return datetime.time(9, 15) <= now <= datetime.time(15, 30)

    def initialize(self):
        try:
            self.api = initialize_api()
            if not self.api: return False
            
            self.initialize_instruments()
            
            # Backfill historical data
            logging.info("Starting historical data backfill...")
            self.backfill_historical_data()
            
            logging.info("Trading initialized")
            return True
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            return False

    def run(self):
        if not self.initialize(): return
        
        import msvcrt
        dashboard_thread = threading.Thread(target=self.update_dashboard_thread, daemon=True)
        dashboard_thread.start()
        
        print("\nStarting live trading... Press M for manual exit, Q to quit")
        
        try:
            while self.running:
                if not self.in_trading_hours():
                    time.sleep(60)
                    continue
                
                if self.pnl <= -Config.MAX_DAILY_LOSS:
                    print("\nDaily loss limit reached!")
                    break
                
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode().lower()
                    if key == 'q':
                        print("\nExiting...")
                        break
                    elif key == 'm':
                        self.manual_exit()
                
                self.manage_positions()
                
                for inst in self.instruments:
                    symbol = inst['symbol']
                    if symbol in self.data_cache and len(self.data_cache[symbol]) >= Config.EMA_PERIOD:
                        df = self.data_cache[symbol]
                        signal, sig_price = self.detect_signal(df)
                        
                        if signal:
                            has_position = any(pos['symbol'] == symbol for pos in self.positions.values())
                            last_sig = self.last_signals.get(symbol)
                            
                            # Enter trade if no existing position and signal changed
                            if not has_position and signal != last_sig:
                                pos_id = self.enter_trade(signal, symbol, sig_price, inst['lot_size'])
                                if pos_id:
                                    self.last_signals[symbol] = signal
                                    self.trades += 1
                        else:
                            # Reset signal when no signal detected
                            self.last_signals[symbol] = None
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            logging.error(f"Main loop error: {e}")
        finally:
            self.running = False
            print("\nTrading stopped")

if __name__ == "__main__":
    print("\n" + "="*120)
    print("LIVE TRADING DASHBOARD")
    print("="*120)
    print(f"Tracking: NIFTY ±{Config.STRIKES_COUNT} strikes & BANKNIFTY ±{Config.STRIKES_COUNT} strikes")
    print(f"NIFTY Expiry: {NIFTY_EXPIRY}")
    print(f"BANKNIFTY Expiry: {BANKNIFTY_EXPIRY}")
    print("="*120)
    
    strategy = TradingStrategy()
    strategy.run()