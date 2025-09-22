import streamlit as st
import yfinance as yf
import ta
import datetime
import pandas as pd
import numpy as np
# import tensorflow as tf
import os
import random
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
from google.cloud import bigquery
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from enums import CloudProvider
from data import Predictor

# Fix randomness
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# tf.random.set_seed(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)


st.set_page_config(page_title="Forecast", layout="wide")
st.title("Prediction")

adsense_code = """
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-9614544934238374"
     crossorigin="anonymous"></script>
"""

st.markdown(adsense_code, unsafe_allow_html=True)

cloud_provider = CloudProvider.GCP

symbol = "GC=F"

# Load data
# @st.cache_data
# def load_data():
    
#     df = yf.Ticker(symbol).history(period="5y")[["Open", "High", "Low", "Close", "Volume"]]
#     df = ta.add_all_ta_features(
#         df,
#         open="Open", high="High", low="Low", close="Close", volume="Volume",
#         fillna=True)
#     return df.dropna()

# df = load_data()
# df.index = pd.to_datetime(df.index).date
# df["Date"] = df.index
# st.subheader("Historical Data Preview")
# st.dataframe(df.tail(10), use_container_width=True)

# # Feature setup
# feature_cols = [
#     "Open", "High", "Low", "Close", "Volume",
#     "momentum_rsi", "trend_macd", "momentum_stoch",
#     "volatility_bbm", "volatility_bbh", "volatility_bbl",
#     "volatility_atr", "trend_ema_fast", "volume_obv"
# ]
# df = df[feature_cols]

# # Sliding window
# window_size = 5
# features, labels = [], []

# for i in range(window_size, len(df)):
#     window = df.iloc[i - window_size:i].values.flatten()
#     features.append(window)
#     labels.append(df["Close"].values[i])

# X = np.array(features)
# y = np.array(labels).reshape(-1, 1)

# # Scaling
# X_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
# y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
# X_scaled = X_scaler.fit_transform(X)
# y_scaled = y_scaler.fit_transform(y)

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=SEED)

# # Model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(256, activation="relu", input_shape=(X.shape[1],)),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(1)
# ])
# model.compile(optimizer="adam", loss="mae", metrics=["mse"])
# model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

# # Evaluation
# loss, mae = model.evaluate(X_test, y_test, verbose=0)
# y_pred_scaled = model.predict(X_test)
# y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
# y_true = y_scaler.inverse_transform(y_test).flatten()
# real_mae = np.mean(np.abs(y_true - y_pred))
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# direction_acc = np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0))

# # Rolling predictions
# actual = df["Close"].values[-30:]
# preds = []
# for i in range(-30, 0):
#     w = df.iloc[i - window_size:i].values.flatten().reshape(1, -1)
#     ws = X_scaler.transform(w)
#     p = model.predict(ws)
#     preds.append(y_scaler.inverse_transform(p)[0][0])
# preds = np.array(preds)

# # Forecast future
# future_preds = []
# latest_window = df.iloc[-window_size:].values.copy()
# for _ in range(5):
#     inp = X_scaler.transform(latest_window.flatten().reshape(1, -1))
#     p = model.predict(inp)
#     unscaled = y_scaler.inverse_transform(p)[0][0]
#     future_preds.append(unscaled)
#     last_day = df.iloc[-1].copy()
#     last_day["Close"] = unscaled
#     latest_window = np.vstack((latest_window[1:], last_day.values))

# st.subheader("Model Evaluation")
# st.metric("Mean Absolute Error (MAE)", f"${real_mae:.2f}")
# st.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
# st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")
# st.metric("Directional Accuracy", f"{direction_acc*100:.2f}%")

predict_obj = Predictor(cloud_provider.project_id, cloud_provider.dataset_id, cloud_provider.table_id, symbol)
df = predict_obj.fetch_prediction_history()
df.index = df["Date"]
df["Real_Close"] = pd.to_numeric(df["Real_Close"], errors="coerce")
#df["Real_Close"] = df["Real_Close"].fillna(0)
st.dataframe(df, use_container_width=True)

# Plot with dates on x-axis
actual = df["Real_Close"].head(10).values
preds = df["Predicted_Close"].head(10).values
future_preds = df["Predicted_Close"].head(1).values
dates = pd.to_datetime(df["Date"].head(10))
future_date = pd.to_datetime(df["Date"].head(1)).values[0]

st.subheader("Actual vs. Predicted & 1-Day Forecast")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates, actual, label="Actual", color="black")
ax.plot(dates, preds, label="Predicted", color="orange")
ax.plot([future_date], future_preds, linestyle="--", marker="o", label="Forecast", color="blue")
ax.set_title("Forecast")
ax.set_xlabel("Date")
ax.legend()
fig.autofmt_xdate()
st.pyplot(fig)

# Plotly
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=actual, name="Actual", line=dict(color="black")))
# fig.add_trace(go.Scatter(y=preds, name="Predicted", line=dict(color="orange")))
# fig.add_trace(go.Scatter(
#     x=list(range(len(actual), len(actual) + 1)),
#     y=future_preds,
#     name="Forecast",
#     line=dict(color="blue", dash="dash"),
#     mode="lines+markers"
# ))
# fig.update_layout(title="Forecast")
# st.plotly_chart(fig, use_container_width=True)

# Display 1-Day Forecasted Prices
st.subheader("Next 1-Day Forecast")
#future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=1, freq="B").strftime("%Y-%m-%d")
#future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=0), periods=1, freq="B").date
future_dates = df["Date"].head(1).values
forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": np.round(future_preds, 2)
})
st.dataframe(forecast_df, use_container_width=True)

# Store forecast_df in BigQuery

# The BigQuery table must have the following fields (columns):
# - Date (STRING or DATE)
# - Predicted_Close (FLOAT)

# client = bigquery.Client(project=cloud_provider.project_id)
# table_ref = f"{cloud_provider.project_id}.{cloud_provider.dataset_id}.{cloud_provider.table_id}"

# print(f"forecast_df: {forecast_df}")
# print(f"table_ref: {table_ref}")

# job = client.load_table_from_dataframe(
#     forecast_df,
#     table_ref,
#     job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
# )
# job.result()  # Wait for the job to complete
# st.success("Forecast uploaded to BigQuery!")

csv = forecast_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV Report", data=csv, file_name="sx5e_prediction_report.csv", mime="text/csv")

# Evaluation Table: Actual vs Predicted
# st.subheader("Actual vs. Predicted Table")

# # Get the last 30 dates
# dates = df.index[-30:]

# # Calculate deviation and direction correctness
# deviation_pct = ((preds - actual) / actual) * 100

# direction_correct = np.sign(np.diff(actual)) == np.sign(np.diff(preds))
# direction_correct = np.append(direction_correct, np.nan)  # Last row has no next-day comparison

# # Build DataFrame
# eval_df = pd.DataFrame({
#     "Date": dates,
#     "Actual_Close": np.round(actual, 2),
#     "Predicted_Close": np.round(preds, 2),
#     "Deviation (%)": np.round(deviation_pct, 2),
#     "Correct Direction": direction_correct
# })

# st.dataframe(eval_df, use_container_width=True)
