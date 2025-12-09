"""
Unified data fetch helper for both live trading and backtesting
Provides consistent contract discovery and historical data fetching
"""

import time
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_time(time_string):
    """Convert time string to timestamp"""
    data = time.strptime(time_string, '%d-%m-%Y %H:%M:%S')
    return time.mktime(data)


def search_specific_contract(api, symbol):
    """Search for specific contract by symbol"""
    try:
        resp = api.searchscrip(exchange='NFO', searchtext=symbol)
        if not resp or resp.get('stat') != 'Ok':
            return None, None
        for v in resp.get('values', []):
            tsym = v.get('tsym', '')
            if tsym == symbol:
                return v.get('token'), tsym
        return None, None
    except Exception as e:
        logging.debug(f"Search error for {symbol}: {e}")
        return None, None


def fetch_contract_data(api, symbol, start_timestamp, end_timestamp, exchange='NFO'):
    """
    Fetch historical data for a single contract
    Returns: (success: bool, dataframe: pd.DataFrame or None, message: str)
    """
    # Search for contract
    token, found_symbol = search_specific_contract(api, symbol)
    if not token:
        return False, None, f"Contract not found"
    
    try:
        # Fetch data
        ret = api.get_time_price_series(
            exchange=exchange,
            token=token,
            starttime=start_timestamp,
            endtime=end_timestamp
        )
    except Exception as e:
        return False, None, f"API Error: {e}"
    
    if not ret or (isinstance(ret, dict) and ret.get('stat') == 'Not_Ok'):
        return False, None, f"No data returned"
    
    # Create DataFrame from API response
    try:
        df = pd.DataFrame(ret)
    except Exception as e:
        # If DataFrame creation fails, return error
        return False, None, f"DataFrame error: {str(e)[:50]}"
    
    if df.empty:
        return False, None, f"Empty dataframe"
    
    # Column mapping
    column_map = {
        'time': 'datetime',
        'ssboe': 'datetime',
        'into': 'open',
        'inth': 'high',
        'intl': 'low',
        'intc': 'close',
        'v': 'volume',
        'intv': 'volume',
        'intoi': 'open_interest',
        'oi': 'open_interest'
    }
    df.rename(columns=column_map, inplace=True)
    
    # Validate required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return False, None, f"Missing columns: {missing}"
    
    # Ensure datetime column exists
    if 'datetime' not in df.columns:
        return False, None, "No datetime column"
    
    # Convert datetime if numeric (safely check Series dtype)
    try:
        # Get the datetime column
        if 'datetime' in df.columns:
            # Try to convert it
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', unit='s')
    except Exception as e:
        # If conversion fails, log it but don't fail completely
        logging.debug(f"Datetime conversion warning: {str(e)[:50]}")
    
    # Ensure volume column exists
    if 'volume' not in df.columns:
        df['volume'] = 1
    
    # Sort by datetime if possible
    try:
        if 'datetime' in df.columns:
            df = df.drop_duplicates(subset=['datetime'], keep='first')
            df = df.sort_values('datetime').reset_index(drop=True)
    except Exception as e:
        # If sorting fails, just reset index without sorting
        df = df.reset_index(drop=True)
    
    return True, df, f"Success: {len(df)} candles"


def fetch_option_data_for_underlying(
    api,
    underlying_name,
    index_token,
    strike_step,
    expiry,
    strikes_count,
    start_timestamp,
    end_timestamp,
    data_dir="data/market_data",
    verbose=True
):
    """
    Fetch option data for an underlying (NIFTY or BANKNIFTY)
    Returns: (successful_count: int, failed_count: int, symbols_fetched: list)
    """
    success_count = 0
    failed_count = 0
    symbols_fetched = []
    
    if verbose:
        print(f"\nProcessing {underlying_name}...")
    
    # Get underlying quotes
    try:
        idx_q = api.get_quotes(exchange='NSE', token=index_token)
    except Exception as e:
        if verbose:
            print(f"Failed to get quotes for {underlying_name}: {e}")
        return 0, strikes_count * 2, []
    
    if not idx_q or idx_q.get('stat') != 'Ok':
        if verbose:
            if idx_q is None:
                print(f"Failed to get quotes for {underlying_name}: API returned None")
            else:
                print(f"Failed to get quotes for {underlying_name}: {idx_q.get('emsg', 'Unknown error')}")
        return 0, strikes_count * 2, []
    
    try:
        ltp = float(idx_q.get('lp', 0))
    except (ValueError, TypeError):
        if verbose:
            print(f"Failed to parse LTP for {underlying_name}: {idx_q.get('lp', 'N/A')}")
        return 0, strikes_count * 2, []
    
    atm = int(round(ltp / strike_step) * strike_step)
    base = 'BANKNIFTY' if 'Bank' in underlying_name or 'BANK' in underlying_name.upper() else 'NIFTY'
    
    if verbose:
        print(f"LTP: {ltp:.2f} | ATM Strike: {atm} | Strike Step: {strike_step}")
    
    # Generate strikes
    strikes = []
    for i in range(-strikes_count, strikes_count + 1):
        strike = atm + (i * strike_step)
        strikes.append(strike)
    
    if verbose:
        print(f"Selected strikes (ATM +- {strikes_count}): {strikes}")
    
    # Fetch data for each strike and type
    for strike in strikes:
        for typ in ['C', 'P']:
            symbol = f"{base}{expiry}{typ}{strike}"
            
            if verbose:
                print(f"  Fetching: {symbol}", end=" ... ")
            
            success, df, msg = fetch_contract_data(api, symbol, start_timestamp, end_timestamp)
            
            if success:
                # Save to CSV
                os.makedirs(data_dir, exist_ok=True)
                filename = f"{symbol}.csv"
                outfile = os.path.join(data_dir, filename)
                df.to_csv(outfile, index=False)
                
                if verbose:
                    print(f"[OK] {msg}")
                
                symbols_fetched.append(symbol)
                success_count += 1
                time.sleep(0.5)  # Rate limiting
            else:
                if verbose:
                    print(f"[SKIP] {msg}")
                failed_count += 1
    
    return success_count, failed_count, symbols_fetched


def fetch_all_option_data(
    api,
    nifty_expiry,
    banknifty_expiry,
    nifty_strikes_count,
    banknifty_strikes_count,
    nifty_strike_step=50,
    banknifty_strike_step=100,
    start_datetime=None,
    end_datetime=None,
    data_dir="data/market_data",
    verbose=True
):
    """
    Fetch option data for all underlyings (NIFTY and BANKNIFTY)
    
    Args:
        api: Initialized API object
        nifty_expiry: NIFTY expiry string (e.g., '09DEC25')
        banknifty_expiry: BANKNIFTY expiry string
        nifty_strikes_count: Number of strikes around ATM for NIFTY
        banknifty_strikes_count: Number of strikes around ATM for BANKNIFTY
        nifty_strike_step: Strike increment for NIFTY (default 50)
        banknifty_strike_step: Strike increment for BANKNIFTY (default 100)
        start_datetime: Start datetime string (format: '09-12-2025 09:15:00')
        end_datetime: End datetime string
        data_dir: Directory to save data
        verbose: Print detailed output
    
    Returns:
        {
            'total_success': int,
            'total_failed': int,
            'symbols': list,
            'nifty': {'success': int, 'failed': int, 'symbols': list},
            'banknifty': {'success': int, 'failed': int, 'symbols': list}
        }
    """
    
    # Default time range if not provided
    if not start_datetime:
        start_datetime = '09-12-2025 09:15:00'
    if not end_datetime:
        end_datetime = '09-12-2025 15:30:00'
    
    start_timestamp = get_time(start_datetime)
    end_timestamp = get_time(end_datetime)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FETCHING OPTION DATA")
        print(f"{'='*70}")
        print(f"Time Range: {start_datetime} to {end_datetime}")
        print(f"NIFTY Expiry: {nifty_expiry} | Strikes: +- {nifty_strikes_count}")
        print(f"BANKNIFTY Expiry: {banknifty_expiry} | Strikes: +- {banknifty_strikes_count}")
        print(f"{'='*70}")
    
    # Fetch NIFTY data
    nifty_success, nifty_failed, nifty_symbols = fetch_option_data_for_underlying(
        api,
        'NIFTY',
        '26000',
        nifty_strike_step,
        nifty_expiry,
        nifty_strikes_count,
        start_timestamp,
        end_timestamp,
        data_dir,
        verbose
    )
    
    # Fetch BANKNIFTY data
    banknifty_success, banknifty_failed, banknifty_symbols = fetch_option_data_for_underlying(
        api,
        'Nifty Bank',
        '26009',
        banknifty_strike_step,
        banknifty_expiry,
        banknifty_strikes_count,
        start_timestamp,
        end_timestamp,
        data_dir,
        verbose
    )
    
    total_success = nifty_success + banknifty_success
    total_failed = nifty_failed + banknifty_failed
    all_symbols = nifty_symbols + banknifty_symbols
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Data fetch completed!")
        print(f"Total files saved: {total_success}")
        print(f"Total failed: {total_failed}")
        print(f"Directory: {data_dir}")
        print(f"{'='*70}")
    
    return {
        'total_success': total_success,
        'total_failed': total_failed,
        'symbols': all_symbols,
        'nifty': {
            'success': nifty_success,
            'failed': nifty_failed,
            'symbols': nifty_symbols
        },
        'banknifty': {
            'success': banknifty_success,
            'failed': banknifty_failed,
            'symbols': banknifty_symbols
        }
    }
