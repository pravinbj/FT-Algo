"""
Debug script to inspect actual tsym formats for options in searchscrip response.
This helps us understand how to properly match option symbols and expiry formats.
"""

import os
import pandas as pd
from market_data import initialize_api

# Create output folder
os.makedirs("data", exist_ok=True)

# Initialize API
api = initialize_api()
if not api:
    raise SystemExit("❌ API initialization failed")

# Run for both NIFTY and BANKNIFTY
for search_text in ["NIFTY", "Nifty Bank"]:
    print(f"\n{'='*80}")
    print(f"🔍 Search results for: {search_text}")
    print("="*80)

    resp = api.searchscrip(exchange="NFO", searchtext=search_text)
    values = resp.get("values", []) if resp else []

    # Filter for only index options
    options = [v for v in values if v.get("instname") == "OPTIDX"]

    print(f"\nTotal scrips found: {len(values)} | Options (OPTIDX): {len(options)}")

    if not options:
        print("⚠ No options found — check API response or market data connection.")
        continue

    # Convert to DataFrame
    df = pd.DataFrame(options)

    # Show selected columns
    cols = ["tsym", "token", "exd", "optt", "instname"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""

    print("\nFirst 20 option records:")
    print("-" * 80)
    print(df[cols].head(20).to_string(index=False))

    # Group by expiry for pattern inspection
    print("\nOptions by expiry (first 30):")
    print("-" * 80)
    for v in options[:30]:
        tsym = v.get("tsym", "")
        exd = v.get("exd", "")
        optt = v.get("optt", "")
        token = v.get("token", "")
        print(f"{tsym:35} | {exd:12} | {optt:3} | Token: {token}")

    # Save to CSV
    filename = f"data/{search_text.replace(' ', '_').upper()}_search_debug.csv"
    df.to_csv(filename, index=False)
    print(f"\n✅ Saved full results to: {filename}")
