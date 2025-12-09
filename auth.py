import requests
import hashlib
import pyotp
from urllib.parse import parse_qs, urlparse
from config import API_KEY, CLIENT_ID, PASSWORD, TWO_FA, API_SECRET, HEADERS

def get_flattrade_token():
    """
    Get authentication token from Flattrade API
    Returns:
        str: Authentication token
    """
    try:
        # Create a session
        ses = requests.Session()
        
        # Step 1: Get session ID
        ses_url = 'https://authapi.flattrade.in/auth/session'
        password_encrypted = hashlib.sha256(PASSWORD.encode()).hexdigest()
        
        res_pin = ses.post(ses_url, headers=HEADERS)
        sid = res_pin.text
        print(f'Session ID: {sid}')
        
        # Step 2: Authenticate and get request code
        auth_url = 'https://authapi.flattrade.in/ftauth'
        auth_payload = {
            "UserName": CLIENT_ID,
            "Password": password_encrypted,
            "PAN_DOB": pyotp.TOTP(TWO_FA).now(),
            "App": "",
            "ClientID": "",
            "Key": "",
            "APIKey": API_KEY,
            "Sid": sid,
            "Override": "Y",
            "Source": "AUTHPAGE"
        }
        
        res2 = ses.post(auth_url, json=auth_payload)
        reqcode_res = res2.json()
        print("Authentication response:", reqcode_res)
        
        # Parse the redirect URL to get request code
        parsed = urlparse(reqcode_res['RedirectURL'])
        req_code = parse_qs(parsed.query)['code'][0]
        
        # Step 3: Generate API secret and get token
        api_secret_raw = API_KEY + req_code + API_SECRET
        api_secret_hash = hashlib.sha256(api_secret_raw.encode()).hexdigest()
        
        token_payload = {
            "api_key": API_KEY,
            "request_code": req_code,
            "api_secret": api_secret_hash
        }
        
        token_url = 'https://authapi.flattrade.in/trade/apitoken'
        res3 = ses.post(token_url, json=token_payload)
        token_response = res3.json()
        print("Token response:", token_response)
        
        return token_response['token']
        
    except Exception as e:
        print(f"Error in authentication: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the authentication
    try:
        token = get_flattrade_token()
        print(f"\nAuthentication successful!\nToken: {token}")
    except Exception as e:
        print(f"Authentication failed: {str(e)}")