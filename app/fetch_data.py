import requests
import pandas as pd
import sqlite3
from datetime import datetime
import os

# symbols path
SYMBOLS_PATH = "/data/symbols.txt"

# API details
API_URL = "https://markets.ft.com/research/webservices/securities/v1/historical-series-quotes"

symbols_file = os.path.join(os.path.dirname(__file__), SYMBOLS_PATH)
SYMBOLS = []

with open(symbols_file, "r") as f:
    SYMBOLS = [line.strip() for line in f.readlines() if line.strip()]

PARAMS_TEMPLATE = {
    "intervalType": "day",
    "dayCount": 900,
    "source": "59da6cdd1d8fd97c"
}

# Database details
DB_PATH = '/data/stocks.db'
TABLE_NAME = 'historical_data'

# Fetch and save data for a symbol
def fetch_and_save_data(symbol):
    print(f"Fetching data for symbol: {symbol}")
    
    # Update parameters with the symbol
    params = PARAMS_TEMPLATE.copy()
    params["symbols"] = symbol

    # Fetch data
    response = requests.get(API_URL, params=params)
    if response.status_code == 200:
        data = response.json()['data']['items'][0]['historicalSeries']['historicalQuoteData']
        df = pd.DataFrame(data)

        # Add metadata columns
        df['symbol'] = symbol
        df['fetched_at'] = datetime.utcnow()

        # Save to database
        conn = sqlite3.connect(DB_PATH)
        create_table_if_not_exists(conn)
        save_to_db(conn, df)
        conn.close()

        print(f"Data for {symbol} saved successfully!")
    else:
        print(f"Failed to fetch data for {symbol}: {response.status_code}")

# Create the table if it doesn't exist
def create_table_if_not_exists(conn):
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        date TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        symbol TEXT,
        ma_9 REAL,
        ma_21 REAL,
        lag_1 REAL,
        fetched_at TEXT,
        PRIMARY KEY (date, symbol)  -- Avoid duplicates
    )
    """
    conn.execute(create_table_query)
    conn.commit()

# Save data to database, avoiding duplicates
def save_to_db(conn, df):
    # Calculate moving averages
    df['ma_9'] = df['close'].rolling(window=9).mean().fillna(0)
    df['ma_21'] = df['close'].rolling(window=21).mean().fillna(0)
    df['lag_1'] = df['close'].shift(1).fillna(df['close'].mean())

    # Avoid duplicates by using INSERT OR IGNORE
    for _, row in df.iterrows():
        date_part = datetime.strptime(row['date'], '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d')
        conn.execute(f"""
        INSERT OR IGNORE INTO {TABLE_NAME} (date, open, high, low, close, volume, symbol, ma_9, ma_21, lag_1, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date_part, row['open'], row['high'], row['low'],
            row['close'], row['volume'], row['symbol'], 
            row['ma_9'], row['ma_21'], row['lag_1'], row['fetched_at'].strftime('%Y-%m-%dT%H:%M:%S')
        ))
    conn.commit()

# Fetch and save data for all symbols
if __name__ == "__main__":
    for symbol in SYMBOLS:
        fetch_and_save_data(symbol)
