import uuid
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
    df = yf.Ticker(symbol).history(period="5y", interval="1d")[["Open", "High", "Low", "Close", "Volume"]]
    df = ta.add_all_ta_features(
        df,
        open="Open", high="High", low="Low", close="Close", volume="Volume",
        fillna=True)
    return df.dropna()

class Predictor:
    def __init__(self, project_id: str, dataset_id: str, table_id: str, symbol: str):
        self.client = bigquery.Client(project=project_id)
        self.symbol = symbol
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id

    def create_predictions(self):
        df = load_data(self.symbol)
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
        window_size = 100
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
        initializer = tf.keras.initializers.GlorotUniform(seed=SEED)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Reshape((window_size, -1), input_shape=(X.shape[1],)),  # reshape to 3D
            tf.keras.layers.LSTM(128, return_sequences=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation="relu"),
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

        print(f"Evaluation Metrics for {self.symbol}:")
        print(f" - Scaled Loss (MSE): {loss:.4f}")
        print(f" - Scaled MAE: {mae:.4f}")
        print(f" - Real MAE: {real_mae:.4f}")
        print(f" - RMSE: {rmse:.4f}")
        print(f" - MAPE: {mape:.2f}%")
        print(f" - Directional Accuracy: {direction_acc:.2f}")

        # Rolling predictions
        preds = []
        for i in range(-window_size, 0):
            w = df.iloc[i - window_size:i].values.flatten().reshape(1, -1)
            ws = X_scaler.transform(w)
            p = model.predict(ws)
            preds.append(y_scaler.inverse_transform(p)[0][0])
        preds = np.array(preds)

        # Forecast future
        future_preds = []
        latest_window = df.iloc[-window_size:].values.copy()
        for _ in range(1):
            inp = X_scaler.transform(latest_window.flatten().reshape(1, -1))
            p = model.predict(inp)
            unscaled = y_scaler.inverse_transform(p)[0][0]
            future_preds.append(unscaled)
            last_day = df.iloc[-1].copy()
            last_day["Close"] = unscaled
            latest_window = np.vstack((latest_window[1:], last_day.values))

        # 🧮 Display 1-Day Forecasted Price
        past_dates = pd.date_range(start=df.index[-window_size] + pd.Timedelta(days=1), periods=window_size, freq="B").strftime("%Y-%m-%d")
        past_df = pd.DataFrame({
            "id": [str(uuid.uuid4()) for _ in range(window_size)],    
            "Date": past_dates,
            "Predicted_Close": np.round(preds, 2)
        })
        past_df["Symbol"] = self.symbol
        #past_df["Created_at"] = pd.Timestamp.today().date()

        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=1, freq="B").strftime("%Y-%m-%d")
        forecast_df = pd.DataFrame({
            "id": [str(uuid.uuid4()) for _ in range(1)],    
            "Date": future_dates,
            "Predicted_Close": np.round(future_preds, 2)
        })
        forecast_df["Symbol"] = self.symbol
        return forecast_df, past_df, df

    def store_predictions(self, forecast_df):
        print("Storing predictions in BigQuery...")

        query = f"""
            SELECT id, Date, Symbol
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            """
        existing = self.client.query(query).to_dataframe()

        # Ensure only Date and Predicted_Close columns are present
        forecast_df = forecast_df[["id", "Symbol", "Date", "Predicted_Close"]]
        # Ensure Date column is of datetime.date type
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date
        #forecast_df["Created_at"] = pd.to_datetime(forecast_df["Created_at"]).dt.date
        # Merge and find new records
        merged = forecast_df.merge(
            existing,
            on=["Date", "Symbol"],
            how="left",
            indicator=True
        )
        merged.rename(columns={"id_x": "id"}, inplace=True)
        merged.drop(columns=["id_y"], inplace=True)

        new_rows = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

        new_rows.dropna(inplace=True)

        if not new_rows.empty:
            table_ref = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
            job = self.client.load_table_from_dataframe(
                new_rows,
                table_ref,
                job_config=bigquery.LoadJobConfig(
                    write_disposition="WRITE_APPEND",
                    schema=[
                        bigquery.SchemaField("id", "STRING"),
                        #bigquery.SchemaField("Created_at", "DATE"),
                        bigquery.SchemaField("Symbol", "STRING"),
                        bigquery.SchemaField("Date", "DATE"),
                        bigquery.SchemaField("Predicted_Close", "FLOAT"),
                    ]
                )
            )
            job.result()
            print(f"Job completed: {job.job_id}")
        else:
            print("No new records to insert.")

    def update_with_real_close(self, df):

        # --- Update Real_Close column for matching dates ---
        print("Updating Real_Close values in BigQuery...")

        # Prepare a DataFrame with Date and Real_Close from df
        real_close_df = pd.DataFrame({
            "Date": pd.to_datetime(df.index).date,
            "Real_Close": df["Close"].values
        })

        print(f"real_close_df:\n{real_close_df}")

        # For each date, update Real_Close in BigQuery
        # Batch update Real_Close using a temporary table and MERGE statement for efficiency

        # Prepare DataFrame for upload
        real_close_df["Date"] = pd.to_datetime(real_close_df["Date"])
        temp_table_id = f"{self.dataset_id}.temp_real_close_update"

        # Upload to temporary table
        job = self.client.load_table_from_dataframe(
            real_close_df,
            f"{self.project_id}.{temp_table_id}",
            job_config=bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",
                schema=[
                    bigquery.SchemaField("Date", "DATE"),
                    bigquery.SchemaField("Real_Close", "FLOAT"),
                ]
            )
        )
        job.result()

        # Use MERGE to update Real_Close in one query
        merge_query = f"""
            MERGE `{self.project_id}.{self.dataset_id}.{self.table_id}` T
            USING `{self.project_id}.{temp_table_id}` S
            ON T.Date = S.Date
            WHEN MATCHED THEN
            UPDATE SET T.Real_Close = S.Real_Close
        """

        merge_job = self.client.query(merge_query)
        merge_job.result()

        # Optionally, delete the temp table
        # self.client.delete_table(f"{self.project_id}.{temp_table_id}", not_found_ok=True)
        print("Real_Close column updated for matching dates.")

    def fetch_prediction_history(self):
        query = f"""
            SELECT Date, Symbol, Real_Close, Predicted_Close
            FROM `{self.project_id}.{self.dataset_id}.{self.table_id}`
            WHERE Symbol = @Symbol
            ORDER BY Date DESC
        """
        print(f"Executing query: {query}")
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("Symbol", "STRING", self.symbol)
            ]
        )
        print(f"Query Job Config: {job_config}")
        df = self.client.query(query, job_config=job_config).to_dataframe()
        #df["Created_at"] = df["Created_at"].astype(str)
        df["Real_Close"] = df["Real_Close"].apply(lambda x: "NaN" if pd.isna(x) else x)
        # Compare each value with the previous day's value
        df["Real_Close"] = pd.to_numeric(df["Real_Close"], errors="coerce")
        df["Predicted_Close"] = pd.to_numeric(df["Predicted_Close"], errors="coerce")
        df["Real_Close_diff"] = df["Real_Close"] - df["Real_Close"].shift(-1)
        df["Predicted_Close_diff"] = df["Predicted_Close"] - df["Predicted_Close"].shift(-1)

        # Condition: both increasing OR both decreasing
        df["Correct_Direction"] = ((df["Real_Close_diff"] > 0) & (df["Predicted_Close_diff"] > 0)) | \
                ((df["Real_Close_diff"] < 0) & (df["Predicted_Close_diff"] < 0))
        #df.drop(columns=["Real_Close_diff", "Predicted_Close_diff"], inplace=True)
        print(f"Fetched prediction history for {self.symbol}:\n{df}")
        correct_direction = df["Correct_Direction"].sum()
        correct_direction_perc = correct_direction / len(df) * 100 if len(df) > 0 else 0
        return df, correct_direction_perc
