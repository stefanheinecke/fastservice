import yfinance as yf
import ta
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from google.cloud import bigquery
import matplotlib.pyplot as plt

# Fix randomness
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Load data
def load_data(symbol: str):
    # For EURO STOXX 50, use the ticker symbol "^STOXX50E"
    df = yf.Ticker("^" + symbol).history(period="5y")[["Open", "High", "Low", "Close", "Volume"]]
    df = ta.add_all_ta_features(
        df,
        open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True)
    return df.dropna()

def create_predictions(symbol):
    df = load_data(symbol)
    df.index = pd.to_datetime(df.index).date
    df["Date"] = df.index

    # Feature setup
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "momentum_rsi", "trend_macd", "momentum_stoch",
        "volatility_bbm", "volatility_bbh", "volatility_bbl",
        "volatility_atr", "trend_ema_fast", "volume_obv"
    ]
    df = df[feature_cols]

    # Sliding window
    window_size = 5
    features, labels = [], []

    for i in range(window_size, len(df)):
        window = df.iloc[i - window_size:i].values.flatten()
        features.append(window)
        labels.append(df["Close"].values[i])

    X = np.array(features)
    y = np.array(labels).reshape(-1, 1)

    # Scaling
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=SEED)

    # Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    # Evaluation
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = y_scaler.inverse_transform(y_test).flatten()
    real_mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    direction_acc = np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0))

    # Rolling predictions
    actual = df["Close"].values[-30:]
    preds = []
    for i in range(-30, 0):
        w = df.iloc[i - window_size:i].values.flatten().reshape(1, -1)
        ws = X_scaler.transform(w)
        p = model.predict(ws)
        preds.append(y_scaler.inverse_transform(p)[0][0])
    preds = np.array(preds)

    # Forecast future
    future_preds = []
    latest_window = df.iloc[-window_size:].values.copy()
    for _ in range(5):
        inp = X_scaler.transform(latest_window.flatten().reshape(1, -1))
        p = model.predict(inp)
        unscaled = y_scaler.inverse_transform(p)[0][0]
        future_preds.append(unscaled)
        last_day = df.iloc[-1].copy()
        last_day["Close"] = unscaled
        latest_window = np.vstack((latest_window[1:], last_day.values))

    # 🧮 Display 5-Day Forecasted Prices
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5, freq="B").strftime("%Y-%m-%d")
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": np.round(future_preds, 2)
    })
    return forecast_df

def store_predictions(forecast_df):

    print("Storing predictions in BigQuery...")

    project_id = "my-sh-project-398715"
    dataset_id = "predict_data"
    table_id = "prediction"

    # Ensure only Date and Predicted_Close columns are present
    forecast_df = forecast_df[["Date", "Predicted_Close"]]

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    job = client.load_table_from_dataframe(
        forecast_df,
        table_ref,
        job_config=bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema=[
                bigquery.SchemaField("Date", "STRING"),
                bigquery.SchemaField("Predicted_Close", "FLOAT"),
            ]
        )
    )
    job.result()
