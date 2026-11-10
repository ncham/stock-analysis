import pandas as pd
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

DB_PATH = "/data/stocks.db"
MODEL_PATH = "/data/stock_model.pkl"

def train_model(symbol):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT open, high, low, volume, close FROM historical_data
    WHERE symbol = '{symbol}' order by date
    """
    df = pd.read_sql(query, conn)
    conn.close()

    # Feature engineering
    df['ma_9'] = df['close'].rolling(window=9).mean().fillna(df['close'].mean())
    df['ma_21'] = df['close'].rolling(window=21).mean().fillna(df['close'].mean())
    df['lag_1'] = df['close'].shift(1).fillna(df['close'].mean())

    # Train-test split
    X = df[['open', 'high', 'low', 'volume', 'ma_9', 'ma_21', 'lag_1']]
    y = df['close']

    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model using the testing data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test data: {mse}")

    # Save the trained model to a file
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return MODEL_PATH

def predict_stock(symbol, date):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT open, high, low, volume, close, ma_9, ma_21, lag_1 FROM historical_data 
    WHERE symbol = '{symbol}' AND date = '{date}'
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        raise Exception("No data found for the specified symbol and date.")
    
    model = joblib.load(MODEL_PATH)
    features = df[['open', 'high', 'low', 'volume', 'ma_9', 'ma_21', 'lag_1']]

    # Use the model to make a prediction
    predicted_price = model.predict(features)[0]
    return predicted_price

def predict_future_prices(symbol, num_days):
    """
    Predict future stock prices for a given symbol and number of days into the future.
    """
    import numpy as np

    # MODEL_PATH = f"/data/models/{symbol}_stock_model.pkl"
    # if not os.path.exists(MODEL_PATH):
    #     raise Exception(f"Model for {symbol} not found. Train the model first.")

    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT date, open, high, low, volume, close FROM historical_data 
    WHERE symbol = '{symbol}' ORDER BY date DESC LIMIT 40
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        raise Exception("No data found for the specified symbol.")

    # Sort data to maintain sequential order
    df = df.sort_values('date')

    # Compute moving averages and other features
    df['ma_9'] = df['close'].rolling(window=9).mean().fillna(df['close'].mean())
    df['ma_21'] = df['close'].rolling(window=21).mean().fillna(df['close'].mean())
    df['lag_1'] = df['close'].shift(1).fillna(df['close'].mean())

    # Use the most recent data as the starting point for predictions
    latest_data = df.iloc[-1][['open', 'high', 'low', 'volume', 'ma_9', 'ma_21', 'lag_1']].values.reshape(1, -1)

    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Store predicted prices
    future_predictions = []

    for _ in range(num_days):
        # Predict the next price
        predicted_price = model.predict(latest_data)[0]
        future_predictions.append(predicted_price)

        # Update the input features for the next prediction
        # Here, you assume other features (open, high, low, volume) remain constant or adjust as needed
        next_features = latest_data.flatten()
        next_features[0] = predicted_price  # Use predicted close as the next open
        next_features[1] = predicted_price * 1.02  # Simulate high as 2% above predicted close
        next_features[2] = predicted_price * 0.98  # Simulate low as 2% below predicted close
        next_features[3] = df['volume'].mean()    # Use average volume
        next_features[4] = (sum(future_predictions[-9:]) / min(9, len(future_predictions)))  # Update ma_9
        next_features[5] = (sum(future_predictions[-21:]) / min(21, len(future_predictions)))  # Update ma_21
        next_features[6] = predicted_price  # Update lag_1

        # Reshape for next prediction
        latest_data = np.array(next_features).reshape(1, -1)

    return future_predictions