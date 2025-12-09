from NorenRestApiPy.NorenApi import NorenApi
from auth import get_flattrade_token
from config import CLIENT_ID, PASSWORD

class FlatTradeAPI(NorenApi):
    def __init__(self):
        """Initialize FlatTrade API with correct endpoints"""
        NorenApi.__init__(self, 
            host='https://piconnect.flattrade.in/PiConnectTP/',
            websocket='wss://piconnect.flattrade.in/PiConnectWSTp/'
        )

def initialize_api():
    """Initialize and authenticate the API"""
    try:
        # Create API instance
        api = FlatTradeAPI()
        
        # Get authentication token
        token = get_flattrade_token()
        
        # Set the session
        ret = api.set_session(
            userid=CLIENT_ID,
            password=PASSWORD,
            usertoken=token
        )
        
        if ret != None:
            print("Login successful!")
            return api
        else:
            print("Login failed!")
            return None
            
    except Exception as e:
        print(f"Error initializing API: {str(e)}")
        return None

def get_nifty_data(api):
    """Get NIFTY50 market data"""
    try:
        # Get NIFTY50 quote
        nifty_quote = api.get_quotes(exchange='NSE', token='26000')  # NIFTY50 token
        print("\nNIFTY50 Quote:")
        print(f"Last Trade Price: {nifty_quote.get('lp', 'N/A')}")
        print(f"Change: {nifty_quote.get('c', 'N/A')}")
        print(f"High: {nifty_quote.get('h', 'N/A')}")
        print(f"Low: {nifty_quote.get('l', 'N/A')}")
        return nifty_quote
    except Exception as e:
        print(f"Error fetching NIFTY data: {str(e)}")
        return None

def get_banknifty_data(api):
    """Get BANKNIFTY market data"""
    try:
        # Get BANKNIFTY quote
        banknifty_quote = api.get_quotes(exchange='NSE', token='26009')  # BANKNIFTY token
        print("\nBANKNIFTY Quote:")
        print(f"Last Trade Price: {banknifty_quote.get('lp', 'N/A')}")
        print(f"Change: {banknifty_quote.get('c', 'N/A')}")
        print(f"High: {banknifty_quote.get('h', 'N/A')}")
        print(f"Low: {banknifty_quote.get('l', 'N/A')}")
        return banknifty_quote
    except Exception as e:
        print(f"Error fetching BANKNIFTY data: {str(e)}")
        return None

def get_instrument_limits(api):
    """Get trading limits"""
    try:
        limits = api.get_limits()
        print("\nTrading Limits:")
        print(f"Cash Available: {limits.get('cash', 'N/A')}")
        print(f"Total Limits: {limits.get('t_limit', 'N/A')}")
        return limits
    except Exception as e:
        print(f"Error fetching limits: {str(e)}")
        return None

if __name__ == "__main__":
    # Initialize API
    api = initialize_api()
    
    if api:
        # Get trading limits
        get_instrument_limits(api)
        
        # Get market data
        get_nifty_data(api)
        get_banknifty_data(api)