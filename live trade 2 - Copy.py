"""
Live Trading Dashboard - All Instruments (API Compliant)
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

from typing import Optional, Dict, Any, List

class TradingStrategy:
    def __init__(self) -> None:
        self.api: Optional[Any] = None
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.pnl: float = 0
        self.trades: int = 0
        self.last_signals: Dict[str, Optional[str]] = {}
        self.running: bool = True
        self.data_cache: Dict[str, pd.DataFrame] = {}
        self.instruments: List[Dict[str, Any]] = []
        self.instrument_data: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        self.position_counter: int = 1
        self.trade_counter: int = 1
        self.api_initialized: bool = False

    def _check_api(self):
        if not self.api or not self.api_initialized:
            logging.error("API not initialized")
            return False
        return True

    def calculate_vwap(self, df):
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
        df.drop(["tpv", "cum_vol", "cum_tpv", "date", "typical_price"], axis=1, inplace=True, errors="ignore")
        return df

    def calculate_ema(self, df):
        df = df.copy()
        df["ema"] = df["close"].ewm(span=Config.EMA_PERIOD, adjust=False).mean()
        df["ema"] = df["ema"].bfill()
        return df

    def detect_signal(self, df):
        if len(df) < 2: return None, None
        prev, curr = df.iloc[-2], df.iloc[-1]
        if (prev["ema"] <= prev["vwap"] and curr["ema"] > curr["vwap"] and 
            curr["close"] > curr["vwap"] and curr["close"] > curr["ema"]):
            return "BUY_CALL", curr["close"]
        elif (prev["ema"] >= prev["vwap"] and curr["ema"] < curr["vwap"] and 
              curr["close"] < curr["vwap"] and curr["close"] < curr["ema"]):
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
        strikes = [atm_strike + (i * strike_step) for i in range(-Config.STRIKES_COUNT, Config.STRIKES_COUNT + 1)]
        return strikes, base, lot

    def search_contract(self, symbol: str) -> Optional[str]:
        try:
            if not self._check_api() or self.api is None:
                return None
            resp = self.api.searchscrip(exchange='NFO', searchtext=symbol)
            if resp and resp.get('stat') == 'Ok':
                for v in resp.get('values', []):
                    if v.get('tsym', '').upper() == symbol.upper():
                        return v.get('token')
        except Exception as e:
            logging.error(f"Error searching contract {symbol}: {e}")
        return None

    def fetch_open_positions(self):
        try:
            if not self._check_api() or self.api is None:
                logging.error("API not initialized for fetching positions")
                return []
            positions_data = self.api.get_positions()
            if not positions_data or positions_data.get('stat') != 'Ok':
                logging.warning(f"Could not fetch positions: {positions_data}")
                return []
            broker_positions = positions_data.get('values', []) if isinstance(positions_data, dict) else positions_data
            logging.info(f"Fetched {len(broker_positions)} open positions from broker")
            if broker_positions:
                print("\n" + "="*120)
                print("OPEN POSITIONS FROM BROKER (PositionBook):")
                print("-"*120)
                print(f"{'Symbol':<25} {'Type':<6} {'Net Qty':<8} {'Avg Price':<12} {'Product':<10} {'LTP':<10} {'URMTOM':<10}")
                print("-"*120)
                for pos in broker_positions:
                    symbol = pos.get('tsym', 'N/A')
                    netqty = int(pos.get('netqty', 0))
                    netavgprc = float(pos.get('netavgprc', 0))
                    product = pos.get('prd', 'N/A')
                    ltp = float(pos.get('lp', 0))
                    urmtom = float(pos.get('urmtom', 0))
                    pos_type = 'CALL' if 'C' in symbol else 'PUT' if 'P' in symbol else 'N/A'
                    print(f"{symbol:<25} {pos_type:<6} {netqty:<8} {netavgprc:<12.2f} {product:<10} {ltp:<10.2f} {urmtom:<10.2f}")
                print("="*120 + "\n")
            return broker_positions
        except Exception as e:
            logging.error(f"Error fetching open positions: {e}")
            return []

    def sync_positions(self):
        logging.info("Syncing positions with broker...")
        try:
            if not self._check_api() or self.api is None:
                return
            positions_data = self.api.get_positions()
            if not positions_data or positions_data.get('stat') != 'Ok':
                return
            broker_positions = positions_data.get('values', []) if isinstance(positions_data, dict) else positions_data
            if not broker_positions:
                logging.info("No open positions found on broker to sync.")
                return
            logging.info(f"Found {len(broker_positions)} open positions on broker. Syncing...")
            for pos_data in broker_positions:
                symbol = pos_data.get('tsym')
                if not symbol or any(p['symbol'] == symbol for p in self.positions.values()):
                    continue
                try:
                    netqty = int(pos_data.get('netqty', 0))
                    if netqty == 0:
                        continue
                    netavgprc = float(pos_data.get('netavgprc', 0))
                    if Config.SL_MODE == 'POINTS' and Config.SL_POINTS > 0:
                        sl = netavgprc - Config.SL_POINTS
                    elif Config.SL_MODE == 'PERCENT' and Config.SL_PERCENT > 0:
                        sl = netavgprc * (1 - Config.SL_PERCENT / 100)
                    else:
                        sl = netavgprc * 0.95
                    if Config.TP_MODE == 'POINTS' and Config.TP_POINTS > 0:
                        tp = netavgprc + Config.TP_POINTS
                    elif Config.TP_MODE == 'PERCENT' and Config.TP_PERCENT > 0:
                        tp = netavgprc * (1 + Config.TP_PERCENT / 100)
                    else:
                        tp = netavgprc * 1.05
                    current_sl = (netavgprc - Config.TRAILING_SL_POINTS) if (Config.TSL_MODE == 'POINTS' and Config.TRAILING_SL_POINTS > 0) else sl
                    pos_id = f"{symbol}_synced_{self.position_counter}"
                    self.positions[pos_id] = {
                        'trade_no': self.trade_counter, 'symbol': symbol, 'type': 'CALL' if 'C' in symbol else 'PUT',
                        'entry_time': datetime.datetime.now(), 'entry': netavgprc, 'qty': abs(netqty), 'order_id': 'synced',
                        'sl': sl, 'tp': tp, 'high': netavgprc, 'low': netavgprc, 'current_sl': current_sl,
                        'max_mtm': 0, 'min_mtm': 0, 'exit_reason': None, 'exit_price': None, 'pnl': 0, 'status': 'OPEN'
                    }
                    self.trade_counter += 1
                    self.position_counter += 1
                    logging.info(f"Synced position: {symbol} @ {netavgprc}, SL: {sl:.2f}, TP: {tp:.2f}")
                except Exception as e:
                    logging.error(f"Error processing synced position for {symbol}: {e}")
            logging.info(f"Position sync complete. Now tracking {len(self.positions)} positions.")
        except Exception as e:
            logging.error(f"Error in sync_positions: {e}")

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self._check_api() or self.api is None:
            return None
        token = self.search_contract(symbol)
        if token:
            try:
                quote = self.api.get_quotes(exchange='NFO', token=token)
                if quote and quote.get('stat') == 'Ok':
                    return quote
            except Exception as e:
                logging.error(f"Exception in get_quote for {symbol}: {e}")
        return None

    def place_order(self, symbol: str, qty: int, action: str = 'B') -> Optional[str]:
        try:
            if not self._check_api() or self.api is None:
                return None
            token = self.search_contract(symbol)
            if not token: 
                return None
            order = self.api.place_order(buy_or_sell=action, product_type='M', exchange='NFO', 
                                        tradingsymbol=symbol, quantity=qty, price_type='MKT', 
                                        price=0, discloseqty=0, retention='DAY')
            if order and order.get('stat') == 'Ok':
                order_id = order.get('norenordno')
                logging.info(f"Order {action} placed: {symbol} | Qty: {qty} | Order ID: {order_id}")
                return order_id
            else:
                logging.error(f"Order placement failed: {order}")
        except Exception as e:
            logging.error(f"Error placing order for {symbol}: {e}")
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
                    self.instrument_data[symbol]['ltp'] = curr_price
                remaining = Config.ENTRY_DELAY - (datetime.datetime.now() - start).seconds
                print(f"  Current: {curr_price:.2f} | Time remaining: {remaining}s", end='\r')
            time.sleep(1)
        order_id = self.place_order(symbol, lot, 'B')
        if order_id:
            sl = (price * (1 - Config.SL_PERCENT / 100) if Config.SL_MODE == 'PERCENT' and Config.SL_PERCENT > 0 
                  else price - Config.SL_POINTS if Config.SL_MODE == 'POINTS' and Config.SL_POINTS > 0 else price * 0.95)
            tp = (price * (1 + Config.TP_PERCENT / 100) if Config.TP_MODE == 'PERCENT' and Config.TP_PERCENT > 0 
                  else price + Config.TP_POINTS if Config.TP_MODE == 'POINTS' and Config.TP_POINTS > 0 else price * 1.05)
            current_sl = (sl if Config.TSL_MODE == 'PERCENT' and Config.TRAILING_SL_PERCENT > 0 
                         else price - Config.TRAILING_SL_POINTS if Config.TSL_MODE == 'POINTS' and Config.TRAILING_SL_POINTS > 0 else sl)
            pos_id = f"{symbol}_{order_id}"
            self.positions[pos_id] = {
                'trade_no': self.trade_counter, 'symbol': symbol, 'type': 'CALL' if signal == 'BUY_CALL' else 'PUT',
                'entry_time': datetime.datetime.now(), 'entry': price, 'qty': lot, 'order_id': order_id,
                'sl': sl, 'tp': tp, 'high': price, 'low': price, 'current_sl': current_sl,
                'max_mtm': 0, 'min_mtm': 0, 'exit_reason': None, 'exit_price': None, 'pnl': 0, 'status': 'OPEN'
            }
            self.trade_counter += 1
            print(f"\n[TRADE ENTERED] {symbol} @ {price} | SL: {sl:.2f} | TP: {tp:.2f}")
            return pos_id
        return None

    def manage_positions(self):
        if not self._check_api():
            return
        for pos_id, pos in list(self.positions.items()):
            quote = self.get_quote(pos['symbol'])
            if not quote or quote.get('stat') != 'Ok': 
                continue
            curr = float(quote.get('lp', 0))
            high = float(quote.get('h', curr))
            low = float(quote.get('l', curr))
            pos['high'] = max(pos['high'], high)
            pos['low'] = min(pos['low'], low)
            pos['max_mtm'] = max(pos['max_mtm'], (pos['high'] - pos['entry']) * pos['qty'])
            pos['min_mtm'] = min(pos['min_mtm'], (pos['low'] - pos['entry']) * pos['qty'])
            if Config.TSL_MODE == 'PERCENT' and Config.TRAILING_SL_PERCENT > 0:
                trail_sl = pos['high'] * (1 - Config.TRAILING_SL_PERCENT / 100)
                pos['current_sl'] = max(pos['current_sl'], trail_sl)
            elif Config.TSL_MODE == 'POINTS' and Config.TRAILING_SL_POINTS > 0:
                trail_sl = pos['high'] - Config.TRAILING_SL_POINTS
                pos['current_sl'] = max(pos['current_sl'], trail_sl)
            exit_price = reason = status = None
            if low <= pos['current_sl']:
                exit_price, reason, status = pos['current_sl'], "SL", "SL_HIT"
            elif high >= pos['tp']:
                exit_price, reason, status = pos['tp'], "TP", "TGT_HIT"
            elif not self.in_trading_hours():
                exit_price, reason, status = curr, "EOD", "EOD_CLOSE"
            if reason:
                self.close_position(pos_id, reason, exit_price, status)

    def close_position(self, pos_id, reason, price=None, status=None):
        pos = self.positions.get(pos_id)
        if not pos: 
            return
        try:
            if price is None:
                quote = self.get_quote(pos['symbol'])
                price = float(quote.get('lp', pos['entry'])) if quote else pos['entry']
            self.place_order(pos['symbol'], pos['qty'], 'S')
            pnl = (price - pos['entry']) * pos['qty'] - (pos['entry'] + price) * pos['qty'] * Config.COMMISSION
            self.pnl += pnl
            self.closed_trades.append({
                'trade_no': pos['trade_no'], 'symbol': pos['symbol'], 'type': pos['type'],
                'entry': pos['entry'], 'entry_time': pos['entry_time'], 'exit': price,
                'exit_time': datetime.datetime.now(), 'qty': pos['qty'], 'pnl': pnl,
                'mtm': (price - pos['entry']) * pos['qty'], 'max_mtm': pos['max_mtm'],
                'min_mtm': pos['min_mtm'], 'reason': reason, 'status': status or 'CLOSED'
            })
            if len(self.closed_trades) > 20:
                self.closed_trades = self.closed_trades[-20:]
            print(f"\n[EXIT] {pos['symbol']} | Reason: {reason} | PnL: {pnl:.2f}")
            del self.positions[pos_id]
        except Exception as e:
            logging.error(f"Error closing position: {e}")

    def manual_exit(self):
        print("\n" + "="*60)
        print("OPTIONS:\n  1. Exit from tracked positions\n  2. Fetch open positions from broker\n  0. Cancel")
        print("="*60)
        try:
            choice = input("\nSelect option (0-2): ")
            if choice == '0':
                return
            elif choice == '2':
                self.fetch_open_positions()
                return
            elif choice != '1':
                print("Invalid option")
                return
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
            if choice == '0': 
                return
            for pos_id, pos in self.positions.items():
                if str(pos['trade_no']) == choice:
                    self.close_position(pos_id, "MANUAL", None, "MANUAL_CLOSE")
                    return
            print("Invalid trade number")
        except Exception as e:
            logging.error(f"Error in manual_exit: {e}")

    def initialize_instruments(self):
        self.instruments = []
        if not self._check_api() or self.api is None:
            logging.error("Cannot initialize instruments: API not available")
            return
        nifty_quote = self.api.get_quotes(exchange='NSE', token='26000')
        banknifty_quote = self.api.get_quotes(exchange='NSE', token='26009')
        if nifty_quote and nifty_quote.get('stat') == 'Ok':
            nifty_ltp = float(nifty_quote.get('lp', 0))
            nifty_strikes, nifty_base, _ = self.get_atm_strikes(nifty_ltp, 'NIFTY', NIFTY_EXPIRY)
            logging.info(f"NIFTY LTP: {nifty_ltp} | Selected strikes: {nifty_strikes}")
            for strike in nifty_strikes:
                for opt_type in ['C', 'P']:
                    self.instruments.append({'symbol': f"{nifty_base}{NIFTY_EXPIRY}{opt_type}{strike}", 
                                           'underlying': 'NIFTY', 'strike': strike, 'type': opt_type, 'lot_size': Config.LOT_NIFTY})
        if banknifty_quote and banknifty_quote.get('stat') == 'Ok':
            banknifty_ltp = float(banknifty_quote.get('lp', 0))
            banknifty_strikes, banknifty_base, _ = self.get_atm_strikes(banknifty_ltp, 'BANKNIFTY', BANKNIFTY_EXPIRY)
            logging.info(f"BANKNIFTY LTP: {banknifty_ltp} | Selected strikes: {banknifty_strikes}")
            for strike in banknifty_strikes:
                for opt_type in ['C', 'P']:
                    self.instruments.append({'symbol': f"{banknifty_base}{BANKNIFTY_EXPIRY}{opt_type}{strike}", 
                                           'underlying': 'BANKNIFTY', 'strike': strike, 'type': opt_type, 'lot_size': Config.LOT_BANKNIFTY})
        logging.info(f"Initialized {len(self.instruments)} instruments")

    def backfill_historical_data(self):
        if not self._check_api() or self.api is None:
            return
        logging.info("Starting historical data backfill...")
        end_time = datetime.datetime.combine(datetime.date.today(), datetime.time(15, 30))
        start_time = end_time - datetime.timedelta(days=Config.BACKFILL_DAYS)
        result = fetch_all_option_data(self.api, NIFTY_EXPIRY, BANKNIFTY_EXPIRY, Config.STRIKES_COUNT,
                                      Config.STRIKES_COUNT, NIFTY_STRIKE_STEP, BANKNIFTY_STRIKE_STEP,
                                      start_time.strftime('%d-%m-%Y %H:%M:%S'), end_time.strftime('%d-%m-%Y %H:%M:%S'),
                                      Config.DATA_DIR, verbose=True)
        for symbol in result['symbols']:
            filepath = os.path.join(Config.DATA_DIR, f"{symbol}.csv")
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col not in df.columns:
                            df[col] = 0
                    if len(df) >= Config.EMA_PERIOD:
                        df = self.calculate_vwap(df)
                        df = self.calculate_ema(df)
                        if not df.empty:
                            last_row = df.iloc[-1]
                            self.instrument_data[symbol] = {'symbol': symbol, 'ltp': float(last_row.get('close', 0)),
                                'vwap': float(last_row.get('vwap', 0)), 'ema': float(last_row.get('ema', 0)), 'signal': None,
                                'volume': int(last_row.get('volume', 0)), 'open': float(last_row.get('open', 0)),
                                'high': float(last_row.get('high', 0)), 'low': float(last_row.get('low', 0))}
                    self.data_cache[symbol] = df
                    logging.debug(f"Cached {symbol}: {len(df)} rows")
                except Exception as e:
                    logging.warning(f"Failed to cache {symbol}: {e}")

    def update_instrument_data(self, symbol, quote):
        if not quote or quote.get('stat') != 'Ok': 
            return
        try:
            curr = float(quote.get('lp', 0))
            if symbol not in self.data_cache or len(self.data_cache[symbol]) < 1:
                return
            df = self.data_cache[symbol]
            last_row = df.iloc[-1]
            self.instrument_data[symbol] = {'symbol': symbol, 'ltp': curr,
                'vwap': float(last_row.get('vwap', 0)), 'ema': float(last_row.get('ema', 0)),
                'signal': self.instrument_data.get(symbol, {}).get('signal'), 'volume': int(quote.get('v', 0)),
                'open': float(quote.get('o', curr)), 'high': float(quote.get('h', curr)), 'low': float(quote.get('l', curr))}
        except Exception as e:
            logging.error(f"Error updating {symbol}: {e}")

    def display_dashboard(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        now = datetime.datetime.now()
        print("="*120)
        print("LIVE TRADING DASHBOARD")
        print("="*120)
        print(f"Date: {now.strftime('%d-%m-%Y')} | Time: {now.strftime('%H:%M:%S')}")
        print(f"Daily P&L: {self.pnl:.2f} | Open Trades: {len(self.positions)} | Instruments: {len(self.instruments)}")
        print("="*120 + "\nALL INSTRUMENTS - LIVE DATA:\n" + "-"*120)
        print(f"{'Symbol':<20} {'LTP':<8} {'VWAP':<8} {'EMA':<8} {'Signal':<10} {'Volume':<10}")
        print("-"*120)
        for inst in sorted(self.instrument_data.values(), key=lambda x: x.get('symbol', '')):
            signal = inst.get('signal', '')
            if signal == 'BUY_CALL':
                signal = "\033[92mBUY_CALL\033[0m"
            elif signal == 'BUY_PUT':
                signal = "\033[91mBUY_PUT\033[0m"
            print(f"{inst.get('symbol', 'N/A'):<20} {inst.get('ltp', 0):<8.2f} {inst.get('vwap', 0):<8.2f} "
                  f"{inst.get('ema', 0):<8.2f} {signal:<10} {inst.get('volume', 0):<10}")
        print("\n\nOPEN POSITIONS:\n" + "-"*120)
        if self.positions:
            print(f"{'#':<3} {'Symbol':<20} {'Type':<6} {'Entry':<8} {'Entry Time':<10} {'LTP':<8} {'Qty':<6} {'MTM':<8} {'Status':<12}")
            print("-"*120)
            for pos_id, pos in self.positions.items():
                quote = self.get_quote(pos['symbol'])
                curr = float(quote.get('lp', pos['entry'])) if quote else pos['entry']
                mtm = (curr - pos['entry']) * pos['qty'] - (pos['entry'] + curr) * pos['qty'] * Config.COMMISSION
                entry_time = pos['entry_time'].strftime('%H:%M:%S') if hasattr(pos['entry_time'], 'strftime') else str(pos['entry_time'])
                print(f"{pos['trade_no']:<3} {pos['symbol']:<20} {pos['type']:<6} {pos['entry']:<8.2f} "
                      f"{entry_time:<10} {curr:<8.2f} {pos['qty']:<6} {mtm:<8.2f} {'OPEN':<12}")
        else:
            print("No open positions")
        if self.closed_trades:
            print("\n\nRECENTLY CLOSED TRADES:\n" + "-"*120)
            print(f"{'#':<3} {'Symbol':<20} {'Type':<6} {'Entry':<8} {'Exit':<8} {'Entry Time':<10} {'Exit Time':<10} {'Qty':<6} {'PnL':<8} {'Status':<12}")
            print("-"*120)
            for trade in self.closed_trades[-10:]:
                entry_time = trade['entry_time'].strftime('%H:%M:%S') if hasattr(trade['entry_time'], 'strftime') else str(trade['entry_time'])
                exit_time = trade['exit_time'].strftime('%H:%M:%S') if hasattr(trade['exit_time'], 'strftime') else str(trade['exit_time'])
                status = trade['status']
                if status == 'SL_HIT':
                    status = "\033[91mSL_HIT\033[0m"
                elif status == 'TGT_HIT':
                    status = "\033[92mTGT_HIT\033[0m"
                pnl_color = "\033[92m" if trade['pnl'] > 0 else "\033[91m" if trade['pnl'] < 0 else ""
                print(f"{trade['trade_no']:<3} {trade['symbol']:<20} {trade['type']:<6} {trade['entry']:<8.2f} "
                      f"{trade['exit']:<8.2f} {entry_time:<10} {exit_time:<10} {trade['qty']:<6} "
                      f"{pnl_color}{trade['pnl']:<8.2f}\033[0m {status:<12}")
        print("\n" + "="*120)
        print("Commands: (M) Manual Exit | (Q) Quit")
        print("="*120)

    def update_dashboard_thread(self):
        while self.running:
            try:
                if not self._check_api():
                    time.sleep(5)
                    continue
                for inst in self.instruments:
                    quote = self.get_quote(inst['symbol'])
                    if quote:
                        self.update_instrument_data(inst['symbol'], quote)
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
            if not self.api:
                logging.error("Failed to initialize API")
                return False
            self.api_initialized = True
            self.sync_positions()
            self.initialize_instruments()
            logging.info("Starting historical data backfill...")
            self.backfill_historical_data()
            logging.info("Trading initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            self.api = None
            self.api_initialized = False
            return False

    def run(self):
        if not self.initialize():
            logging.error("Failed to initialize trading system. Exiting...")
            return
        use_msvcrt = os.name == 'nt'
        old_settings = None
        dashboard_thread = threading.Thread(target=self.update_dashboard_thread, daemon=True)
        dashboard_thread.start()
        print("\nStarting live trading... Press M for manual exit, Q to quit")
        if not use_msvcrt:
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception as e:
                logging.warning(f"Could not set up terminal settings: {e}")
        try:
            while self.running:
                if not self.in_trading_hours():
                    time.sleep(60)
                    continue
                if self.pnl <= -Config.MAX_DAILY_LOSS:
                    print("\nDaily loss limit reached!")
                    break
                if use_msvcrt:
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode().lower()
                        if key == 'q':
                            print("\nExiting...")
                            break
                        elif key == 'm':
                            self.manual_exit()
                else:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
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
                            if not has_position and signal != last_sig:
                                pos_id = self.enter_trade(signal, symbol, sig_price, inst['lot_size'])
                                if pos_id:
                                    self.last_signals[symbol] = signal
                                    self.trades += 1
                        else:
                            self.last_signals[symbol] = None
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            logging.error(f"Main loop error: {e}")
        finally:
            self.running = False
            if not use_msvcrt and old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except Exception as e:
                    logging.warning(f"Could not restore terminal settings: {e}")
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