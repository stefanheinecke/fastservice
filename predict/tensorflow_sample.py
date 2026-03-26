import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf
import ta
import matplotlib.pyplot as plt
import random
import os

# 🔒 Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# 📥 Download Apple stock data
aapl = yf.Ticker("AAPL")
df = aapl.history(period="5y")[["Open", "High", "Low", "Close", "Volume"]].dropna()

# ➕ Add technical indicators
df = ta.add_all_ta_features(
    df,
    open="Open",
    high="High",
    low="Low",
    close="Close",
    volume="Volume",
    fillna=True
)

# 🎯 Choose features
feature_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "momentum_rsi", "trend_macd", "momentum_stoch",
    "volatility_bbm", "volatility_bbh", "volatility_bbl",
    "volatility_atr", "trend_ema_fast", "volume_obv"
]
df = df[feature_cols].dropna()

# 🧱 Build rolling windows
window_size = 5
features, labels = [], []

for i in range(window_size, len(df)):
    window = df.iloc[i - window_size:i].values.flatten()
    features.append(window)
    labels.append(df["Close"].values[i])  # actual next Close price

X = np.array(features)
y = np.array(labels).reshape(-1, 1)

# 🔁 Scale features and target separately
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# 🧪 Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# 🧠 Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(X_train, y_train, epochs=10, batch_size=16)

# 🎯 Evaluation metrics
loss, mae = model.evaluate(X_test, y_test)
real_error = mae * (y_scaler.data_max_[0] - y_scaler.data_min_[0])
print(f"\nMean Absolute Error (MAE) ≈ ${real_error:.2f}")

# 📊 Prediction vs Actual comparison
y_pred_scaled = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
y_true = y_scaler.inverse_transform(y_test).flatten()

# Directional accuracy
actual_direction = (np.diff(y_true) > 0).astype(int)
predicted_direction = (np.diff(y_pred) > 0).astype(int)
direction_accuracy = np.mean(actual_direction == predicted_direction)
print(f"Directional Accuracy: {direction_accuracy * 100:.2f}%")

# Percentage deviation (MAPE)
epsilon = 1e-8
percentage_errors = np.abs((y_true - y_pred) / (y_true + epsilon)) * 100
mean_pct_error = np.mean(percentage_errors)
print(f"Mean Absolute Percentage Error (MAPE): {mean_pct_error:.2f}%")

# 📉 Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# 🔮 Recursive forecast for next 5 days
future_predictions = []
latest_window = df.iloc[-window_size:].values.copy()

for _ in range(5):
    input_window = latest_window.flatten().reshape(1, -1)
    input_scaled = X_scaler.transform(input_window)
    pred_scaled = model.predict(input_scaled)
    pred_price = y_scaler.inverse_transform(pred_scaled)[0][0]
    future_predictions.append(pred_price)

    # Update window
    next_day = df.iloc[-1].copy()
    next_day["Close"] = pred_price
    latest_window = np.vstack((latest_window[1:], next_day.values))

# 📅 Print 5-day forecast
print("\n📅 Predicted Prices for the Next 5 Days:")
for i, price in enumerate(future_predictions, 1):
    print(f"Day {i}: ${price:.2f}")

# 📈 Visualization
actual_closes = df["Close"].values[-30:]

# Rolling predictions (last 30 days)
rolling_predictions = []
for i in range(-30, 0):
    window = df.iloc[i - window_size:i].values.flatten().reshape(1, -1)
    window_scaled = X_scaler.transform(window)
    pred_scaled = model.predict(window_scaled)
    pred_price = y_scaler.inverse_transform(pred_scaled)[0][0]
    rolling_predictions.append(pred_price)
rolling_predictions = np.array(rolling_predictions)

# 🖼️ Plotting
plt.figure(figsize=(12, 6))
plt.plot(range(len(actual_closes)), actual_closes, label="Actual Close", color="black")
plt.plot(range(len(actual_closes)), rolling_predictions, label="Predicted Close", color="orange")
plt.plot(range(len(actual_closes), len(actual_closes) + 5), future_predictions, label="5-Day Forecast", color="blue", linestyle="--", marker="o")

# 🧪 Add text with evaluation metrics
metrics_text = (
    f"MAE: ${real_error:.2f}\n"
    f"RMSE: ${rmse:.2f}\n"
    f"Directional Accuracy: {direction_accuracy*100:.2f}%\n"
    f"MAPE: {mean_pct_error:.2f}%"
)
plt.annotate(metrics_text, xy=(0.75, 0.05), xycoords="axes fraction", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="lightyellow"))

plt.title("Apple (AAPL) – Actual vs Predicted + 5-Day Forecast")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📦 Create a DataFrame of recent prediction results
test_indices = y_test.shape[0]
dates = df.index[-test_indices:]  # assumes no shuffling

# Re-align indices if needed
if len(dates) != len(y_true):
    dates = pd.date_range(end=df.index[-1], periods=len(y_true), freq="B")  # fallback

# Ensure all arrays are same length (N-1)
report_df = pd.DataFrame({
    "Date": dates[1:],  # skip first to align with diff
    "Actual_Close": y_true[1:],
    "Predicted_Close": y_pred[1:],
    "Absolute_Error": np.abs(y_true[1:] - y_pred[1:]),
    "Percentage_Error": percentage_errors[1:],
    "Correct_Direction": (actual_direction == predicted_direction).astype(int)
})


# ✅ Save to CSV
report_df.to_csv("prediction_report.csv", index=False)
print("\n📁 Saved prediction report to prediction_report.csv")
