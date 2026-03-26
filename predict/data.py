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
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Bidirectional, LSTM, Reshape
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.initializers import GlorotUniform

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
    TABLE_NAME = "predictions"

    def __init__(self, database_url: str, symbol: str):
        self.engine = create_engine(database_url)
        self.symbol = symbol
        self._ensure_table()

    def _ensure_table(self):
        """Create the predictions table if it doesn't exist."""
        ddl = text(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                predicted_close DOUBLE PRECISION,
                real_close DOUBLE PRECISION
            )
        """)
        with self.engine.begin() as conn:
            conn.execute(ddl)

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
        num_features = len(feature_cols)

        # Sliding window — keep as 3D for proper LSTM input
        window_size = 100
        features, labels = [], []

        for i in range(window_size, len(df)):
            window = df.iloc[i - window_size:i].values  # (window_size, num_features)
            features.append(window)
            labels.append(df["Close"].values[i])

        X = np.array(features)                    # (samples, window_size, num_features)
        y = np.array(labels).reshape(-1, 1)

        # Per-feature scaling across all windows
        n_samples = X.shape[0]
        X_flat = X.reshape(-1, num_features)
        X_scaler = MinMaxScaler()
        X_flat_scaled = X_scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, window_size, num_features)

        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y)

        # Chronological train/test split (no data leakage)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

        # Model — stacked LSTM with proper 3D input
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True,
                                 input_shape=(window_size, num_features)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mse",
            metrics=["mae"],
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        )

        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.15,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        )

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

        # Rolling predictions over all available data (after the initial training window)
        num_predictions = len(df) - window_size
        preds = []
        for i in range(-num_predictions, 0):
            w = df.iloc[i - window_size:i].values          # (window_size, num_features)
            ws = X_scaler.transform(w.reshape(-1, num_features)).reshape(1, window_size, num_features)
            p = model.predict(ws)
            preds.append(y_scaler.inverse_transform(p)[0][0])
        preds = np.array(preds)

        # Forecast future (1 day)
        future_preds = []
        latest_window = df.iloc[-window_size:].values.copy()
        for _ in range(1):
            inp = X_scaler.transform(latest_window.reshape(-1, num_features)).reshape(1, window_size, num_features)
            p = model.predict(inp)
            unscaled = y_scaler.inverse_transform(p)[0][0]
            future_preds.append(unscaled)
            last_day = df.iloc[-1].copy()
            last_day["Close"] = unscaled
            latest_window = np.vstack((latest_window[1:], last_day.values))

        # 🧮 Display 1-Day Forecasted Price
        past_dates = pd.date_range(end=df.index[-1], periods=num_predictions, freq="B").strftime("%Y-%m-%d")
        past_df = pd.DataFrame({
            "id": [str(uuid.uuid4()) for _ in range(num_predictions)],    
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
        print("Storing predictions in Postgres...")

        forecast_df = forecast_df[["id", "Symbol", "Date", "Predicted_Close"]].copy()
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date

        # Find existing dates+symbols to avoid duplicates
        with self.engine.connect() as conn:
            existing = pd.read_sql(
                text(f"SELECT id, date, symbol FROM {self.TABLE_NAME}"),
                conn,
            )

        merged = forecast_df.merge(
            existing,
            left_on=["Date", "Symbol"],
            right_on=["date", "symbol"],
            how="left",
            indicator=True,
        )
        new_rows = merged[merged["_merge"] == "left_only"][
            ["id_x", "Symbol", "Date", "Predicted_Close"]
        ].rename(columns={"id_x": "id"}).dropna()

        if not new_rows.empty:
            records = new_rows.rename(columns={
                "Symbol": "symbol",
                "Date": "date",
                "Predicted_Close": "predicted_close",
            })
            records.to_sql(
                self.TABLE_NAME, self.engine, if_exists="append", index=False,
            )
            print(f"Inserted {len(records)} new rows.")
        else:
            print("No new records to insert.")

    def update_with_real_close(self, df):
        print("Updating real_close values in Postgres...")

        real_close_df = pd.DataFrame({
            "date": pd.to_datetime(df.index).date,
            "real_close": df["Close"].values,
        })

        update_stmt = text(f"""
            UPDATE {self.TABLE_NAME}
            SET real_close = :real_close
            WHERE date = :date AND symbol = :symbol
        """)

        with self.engine.begin() as conn:
            for _, row in real_close_df.iterrows():
                conn.execute(update_stmt, {"real_close": float(row["real_close"]),
                                           "date": row["date"],
                                           "symbol": self.symbol})

        print("real_close column updated for matching dates.")

    def fetch_data_summary(self):
        """Return available date range and row count per symbol."""
        query = text(f"""
            SELECT symbol,
                   MIN(date) AS min_date,
                   MAX(date) AS max_date,
                   COUNT(*) AS total_rows,
                   COUNT(real_close) AS rows_with_real
            FROM {self.TABLE_NAME}
            GROUP BY symbol
            ORDER BY symbol
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        return df

    def fetch_prediction_history(self, limit=None, start_date=None, end_date=None):
        where_clauses = ["symbol = :symbol", "real_close IS NOT NULL"]
        params = {"symbol": self.symbol}

        if start_date:
            where_clauses.append("date >= :start_date")
            params["start_date"] = start_date
        if end_date:
            where_clauses.append("date <= :end_date")
            params["end_date"] = end_date

        where_sql = " AND ".join(where_clauses)
        limit_sql = f"LIMIT :limit" if limit else ""
        if limit:
            params["limit"] = limit

        query = text(f"""
            SELECT date AS "Date", symbol AS "Symbol",
                   real_close AS "Real_Close", predicted_close AS "Predicted_Close"
            FROM {self.TABLE_NAME}
            WHERE {where_sql}
            ORDER BY date DESC
            {limit_sql}
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)

        df["Real_Close"] = pd.to_numeric(df["Real_Close"], errors="coerce")
        df["Predicted_Close"] = pd.to_numeric(df["Predicted_Close"], errors="coerce")
        df["Real_Close_diff"] = df["Real_Close"] - df["Real_Close"].shift(-1)
        df["Predicted_Close_diff"] = df["Predicted_Close"] - df["Predicted_Close"].shift(-1)

        df["Correct_Direction"] = (
            ((df["Real_Close_diff"] > 0) & (df["Predicted_Close_diff"] > 0)) |
            ((df["Real_Close_diff"] < 0) & (df["Predicted_Close_diff"] < 0))
        )

        correct_direction = df["Correct_Direction"].sum()
        correct_direction_perc = correct_direction / len(df) * 100 if len(df) > 0 else 0

        mae = float(np.mean(np.abs(df["Real_Close"] - df["Predicted_Close"]))) if len(df) > 0 else 0

        return df, correct_direction_perc, mae
