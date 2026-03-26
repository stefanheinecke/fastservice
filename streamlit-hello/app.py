import os
import io
import base64
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import ta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from flask import Flask, jsonify, render_template, send_file

# --------------- reproducibility ---------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

app = Flask(__name__)

# --------------- data / model (computed once) ---------------
_cache = {}


def _build():
    """Load data, train model, compute metrics & forecasts."""
    if _cache:
        return _cache

    # --- data ---
    df = yf.Ticker("^STOXX50E").history(period="5y")[
        ["Open", "High", "Low", "Close", "Volume"]
    ]
    df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close",
        volume="Volume", fillna=True,
    )
    df = df.dropna()
    df.index = pd.to_datetime(df.index).date

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "momentum_rsi", "trend_macd", "momentum_stoch",
        "volatility_bbm", "volatility_bbh", "volatility_bbl",
        "volatility_atr", "trend_ema_fast", "volume_obv",
    ]
    historical_tail = df[["Open", "High", "Low", "Close", "Volume"]].tail(10).copy()
    historical_tail["Date"] = [str(d) for d in historical_tail.index]
    df = df[feature_cols]

    # --- sliding window ---
    window_size = 5
    features, labels = [], []
    for i in range(window_size, len(df)):
        features.append(df.iloc[i - window_size:i].values.flatten())
        labels.append(df["Close"].values[i])

    X = np.array(features)
    y = np.array(labels).reshape(-1, 1)

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=SEED,
    )

    # --- model ---
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation="relu", input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    # --- evaluation ---
    y_pred_scaled = model.predict(X_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = y_scaler.inverse_transform(y_test).flatten()
    real_mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    direction_acc = float(np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)))

    # --- rolling predictions (last 30 days) ---
    actual = df["Close"].values[-30:]
    preds = []
    for i in range(-30, 0):
        w = df.iloc[i - window_size:i].values.flatten().reshape(1, -1)
        ws = X_scaler.transform(w)
        p = model.predict(ws, verbose=0)
        preds.append(float(y_scaler.inverse_transform(p)[0][0]))

    # --- 5-day forecast ---
    future_preds = []
    latest_window = df.iloc[-window_size:].values.copy()
    for _ in range(5):
        inp = X_scaler.transform(latest_window.flatten().reshape(1, -1))
        p = model.predict(inp, verbose=0)
        unscaled = float(y_scaler.inverse_transform(p)[0][0])
        future_preds.append(unscaled)
        last_day = df.iloc[-1].copy()
        last_day["Close"] = unscaled
        latest_window = np.vstack((latest_window[1:], last_day.values))

    future_dates = (
        pd.date_range(
            start=pd.Timestamp(df.index[-1]) + pd.Timedelta(days=1),
            periods=5, freq="B",
        ).strftime("%Y-%m-%d").tolist()
    )

    # --- chart (base64 PNG) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual, label="Actual", color="black")
    ax.plot(preds, label="Predicted", color="orange")
    ax.plot(
        range(len(actual), len(actual) + 5), future_preds,
        linestyle="--", marker="o", label="Forecast", color="blue",
    )
    ax.set_title("EURO STOXX 50 (SX5E) – Actual vs Predicted + 5-Day Forecast")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    chart_b64 = base64.b64encode(buf.read()).decode()

    # --- forecast CSV bytes ---
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": np.round(future_preds, 2),
    })
    csv_bytes = forecast_df.to_csv(index=False).encode("utf-8")

    _cache.update({
        "metrics": {
            "mae": round(real_mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "direction_accuracy": round(direction_acc * 100, 2),
        },
        "historical": historical_tail.reset_index(drop=True).to_dict(orient="records"),
        "forecast": forecast_df.to_dict(orient="records"),
        "chart_b64": chart_b64,
        "csv_bytes": csv_bytes,
    })
    return _cache


# --------------- routes ---------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/data")
def api_data():
    data = _build()
    return jsonify({
        "metrics": data["metrics"],
        "historical": data["historical"],
        "forecast": data["forecast"],
        "chart": data["chart_b64"],
    })


@app.route("/api/download")
def download_csv():
    data = _build()
    buf = io.BytesIO(data["csv_bytes"])
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name="sx5e_prediction_report.csv")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
