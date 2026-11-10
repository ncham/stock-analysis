from flask import Flask, request, jsonify, render_template, Response
from datetime import datetime, timedelta
from train_model import train_model, predict_stock, predict_future_prices
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import os
import joblib
import json
import numpy as np

app = Flask(__name__)

DB_PATH = "/data/stocks.db"
MODEL_PATH = "/data/stock_model.pkl"
SYMBOLS_PATH = "/data/symbols.txt"

# Route: Home
@app.route("/")
def home():
    # return render_template("index.html")
    # Read symbols from the text file
    symbols_file = os.path.join(os.path.dirname(__file__), SYMBOLS_PATH)
    with open(symbols_file, "r") as f:
        symbols = [line.strip() for line in f.readlines()]
    symbols.sort()
    return render_template("index.html", symbols=symbols)

# Route: Train the model
@app.route("/train", methods=["POST"])
def train():
    try:
        symbol = request.json.get("symbol")
        model_path = train_model(symbol)
        return jsonify({"status": "success", "message": "Model trained successfully.", "model_path": model_path})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Route: Predict stock price
@app.route("/predict", methods=["POST"])
def predict():
    try:
        symbol = request.json.get("symbol")
        date = request.json.get("date")
        predicted_price = predict_stock(symbol, date)
        return jsonify({"status": "success", "predicted_price": predicted_price})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Route: Visualization
@app.route("/plot")
def plot():
    symbol = request.args.get("symbol", "COMB.N0000:CSE")
    conn = sqlite3.connect("/data/stocks.db")
    query = f"SELECT date, close, ma_9, ma_21 FROM historical_data WHERE symbol = '{symbol}' order by date"
    df = pd.read_sql(query, conn)
    conn.close()

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Replace 0 with NaN
    df['ma_9'] = df['ma_9'].replace(0, np.nan)
    df['ma_21'] = df['ma_21'].replace(0, np.nan)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['close'], label='Close Price', color='red')
    plt.plot(df['date'], df['ma_9'], label='9-Day MA', color='green')
    plt.plot(df['date'], df['ma_21'], label='21-Day MA', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Stock Prices and Moving Averages for {symbol}')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object and send it as a response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return Response(img, mimetype='image/png')

@app.route("/plot_predict", methods=["GET"])
def plot_predict():
    symbol = request.args.get("symbol")

    # Connect to database and fetch data
    conn = sqlite3.connect("/data/stocks.db")
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=90)
    
    query = f"""
    SELECT date, close, open, high, low, volume, ma_9, ma_21, lag_1 FROM historical_data
    WHERE symbol = '{symbol}' AND date BETWEEN '{start_date}' AND '{end_date}' ORDER BY date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        raise Exception("No data found for the specified symbol and date range.")
    
    # Load trained model
    model = joblib.load(MODEL_PATH)

    # Predict closing prices
    features = df[['open', 'high', 'low', 'volume', 'ma_9', 'ma_21', 'lag_1']]
    df['predicted_close'] = model.predict(features)

    # Create a comparison DataFrame
    comparison_df = df[['date', 'close', 'predicted_close']].copy()
    comparison_df.set_index('date', inplace=True)

    # Plot actual vs predicted values
    comparison_df.plot(kind='bar', figsize=(12, 6))
    plt.title(f"Actual vs Predicted Closing Prices ({symbol})")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.xticks(rotation=45)
    plt.legend(["Actual", "Predicted"])
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to a BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # Return the image as a response
    return Response(buf, mimetype="image/png")

@app.route("/plot_predict_future", methods=["GET"])
def plot_future_predictions():
    symbol = request.args.get("symbol")
    num_days = request.args.get("num_days")
    num_days = int(num_days)

    future_prices = predict_future_prices(symbol, num_days)

    # Get historical data for visualization
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT date, close FROM historical_data
    WHERE symbol = '{symbol}' ORDER BY date
    """
    historical_data = pd.read_sql(query, conn)
    conn.close()

    # Create a date range for future predictions
    last_date = pd.to_datetime(historical_data['date'].iloc[-1])
    future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(num_days)]
    historical_data_dates = pd.to_datetime(historical_data['date'])

    # Plot historical and predicted data
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data_dates, historical_data['close'], label="Historical Prices", color='blue')
    plt.plot(future_dates, future_prices, label="Future Predictions", color='orange', linestyle='--')

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Future Stock Price Predictions for {symbol}")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    # plt.show()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return Response(img, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
