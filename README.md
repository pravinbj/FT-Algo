# Flattrade Trading Bot

This is an automated trading bot built for Flattrade using their API integration and websocket connection for real-time data.

## Prerequisites

- Python 3.8 or higher
- Flattrade Trading Account
- API Credentials from Flattrade

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:

Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

Windows Command Prompt:
```cmd
.\venv\Scripts\activate.bat
```

Linux/Mac:
```bash
source venv/bin/activate
```

4. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a copy of `config.py` with your credentials:
   - Add your Flattrade API credentials
   - Configure your trading parameters

## Usage

1. Make sure your virtual environment is activated
2. Run the main bot script:
```bash
python strategy.py
```

For websocket example:
```bash
python websocket_example.py
```

## Operational checklist

### Daily actions
Follow these steps each trading day before you start automated runs.

- Activate your virtual environment (PowerShell):
```powershell
.\venv\Scripts\Activate.ps1
```
- Pull the latest code if you're keeping the repo under version control:
```powershell
git pull origin main
```
- Run a short smoke test to verify connectivity and credentials:
```powershell
python auth.py
python market_data.py
```
- Start the live data monitors (websocket/options monitor) and verify price feed:
```powershell
python websocket_stream.py
python nifty_options_monitor.py
```
- Review the bot logs and today's account balance/limits before enabling live trading.
- If running the strategy, start it in paper/simulated mode first or with conservative settings:
```powershell
python strategy.py
```
- Monitor open positions and the bot's PnL periodically. If anything unexpected happens, stop the bot and investigate logs.

### One-time actions (setup)
Perform these once when installing the bot on a new machine or when doing an initial setup.

- Clone the repository and create a virtual environment:
```powershell
git clone <repository-url>
cd <repository-name>
python -m venv venv
```
- Activate the virtual environment and install dependencies:
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
- Edit `config.py` and add your Flattrade credentials and trading parameters (API key, client id, password, TOTP key, API secret, etc.).
- Run `auth.py` once to verify you can obtain a token from Flattrade.
- Run `market_data.py` and `nifty_options_monitor.py` to verify data fetching and option token lookup.
- Configure logging and data directories (the bot writes CSVs to the `data/` folder). Make sure those folders are writable by your account.
- (Optional) Set up an OS-level scheduler or service to auto-start the bot during trading hours (Windows Task Scheduler or a process manager). Test this thoroughly.


## Required Packages

- NorenRestApiPy==0.0.20
- requests==2.32.5
- pandas==2.3.3
- numpy==2.3.4
- websocket-client==1.8.0
- python-dateutil==2.9.0
- pyotp==2.9.0

## Important Notes

- Make sure to test the bot in a paper trading environment before using it with real money
- Keep your API credentials secure and never share them
- Monitor the bot's performance regularly
- Check Flattrade's API documentation for any updates or changes

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are correctly installed
2. Verify your API credentials are correct
3. Check your internet connection
4. Look for any error messages in the console output

## Disclaimer

This bot is for educational purposes only. Trade at your own risk. Always understand the trading strategy and risk management before deploying any automated trading system.
