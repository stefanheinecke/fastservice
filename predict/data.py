import uuid
import yfinance as yf
import ta
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

        # --- Return-based features ---
        df["return_1d"] = df["Close"].pct_change()
        df["return_5d"] = df["Close"].pct_change(5)
        df["return_10d"] = df["Close"].pct_change(10)
        df["return_20d"] = df["Close"].pct_change(20)
        df["rolling_vol_10"] = df["return_1d"].rolling(10).std()
        df["rolling_vol_20"] = df["return_1d"].rolling(20).std()
        df["close_to_ema"] = df["Close"] / df["trend_ema_fast"] - 1
        df.dropna(inplace=True)

        feature_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "momentum_rsi", "trend_macd", "momentum_stoch",
            "volatility_bbm", "volatility_bbh", "volatility_bbl",
            "volatility_atr", "trend_ema_fast", "volume_obv",
            "return_1d", "return_5d", "return_10d", "return_20d",
            "rolling_vol_10", "rolling_vol_20", "close_to_ema",
        ]
        close_prices = df["Close"].values.copy()
        df = df[feature_cols]
        num_features = len(feature_cols)

        # Target: raw log returns (no scaler — preserves sign for direction)
        log_returns = np.log(close_prices[1:] / close_prices[:-1])

        window_size = 60
        features, labels = [], []
        for i in range(window_size, len(df)):
            window = df.iloc[i - window_size:i].values
            features.append(window)
            labels.append(log_returns[i - 1])

        X = np.array(features)
        y = np.array(labels).reshape(-1, 1)  # raw log returns, no scaling

        # StandardScaler on features (handles outliers much better than MinMaxScaler)
        n_samples = X.shape[0]
        X_flat = X.reshape(-1, num_features)
        X_scaler = StandardScaler()
        X_flat_scaled = X_scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(n_samples, window_size, num_features)

        # Chronological split
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Custom loss: MSE + directional penalty
        def direction_aware_loss(y_true, y_pred):
            mse = tf.reduce_mean(tf.square(y_true - y_pred))
            # Penalise when predicted sign differs from true sign
            sign_penalty = tf.reduce_mean(tf.nn.relu(-y_true * y_pred))
            return mse + 2.0 * sign_penalty

        # Lighter model to reduce overfitting on ~1000 samples
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True,
                                 input_shape=(window_size, num_features)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
            loss=direction_aware_loss,
            metrics=["mae"],
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6
        )

        model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=64,
            validation_split=0.15,
            callbacks=[early_stop, reduce_lr],
            verbose=0,
        )

        # --- Evaluation ---
        y_pred_ret = model.predict(X_test, verbose=0).flatten()
        y_true_ret = y_test.flatten()

        # Convert returns → prices for display metrics
        test_base = close_prices[window_size + split_idx - 1:
                                 window_size + split_idx - 1 + len(y_true_ret)]
        y_pred_prices = test_base * np.exp(y_pred_ret)
        y_true_prices = test_base * np.exp(y_true_ret)

        real_mae = np.mean(np.abs(y_true_prices - y_pred_prices))
        rmse = np.sqrt(mean_squared_error(y_true_prices, y_pred_prices))
        mape = np.mean(np.abs((y_true_prices - y_pred_prices) / y_true_prices)) * 100
        direction_acc = np.mean((y_true_ret > 0) == (y_pred_ret > 0))

        print(f"Evaluation Metrics for {self.symbol}:")
        print(f" - Real MAE: ${real_mae:.2f}")
        print(f" - RMSE: ${rmse:.2f}")
        print(f" - MAPE: {mape:.2f}%")
        print(f" - Directional Accuracy: {direction_acc:.2%}")

        # --- Rolling predictions → convert back to prices ---
        num_predictions = len(df) - window_size
        pred_returns = []
        for i in range(-num_predictions, 0):
            w = df.iloc[i - window_size:i].values
            ws = X_scaler.transform(w.reshape(-1, num_features)).reshape(1, window_size, num_features)
            p = model.predict(ws, verbose=0)
            pred_returns.append(p[0][0])
        pred_returns = np.array(pred_returns)

        base_prices = close_prices[window_size - 1:-1]
        preds = base_prices * np.exp(pred_returns)

        # Forecast future (1 day)
        latest_window = df.iloc[-window_size:].values.copy()
        inp = X_scaler.transform(latest_window.reshape(-1, num_features)).reshape(1, window_size, num_features)
        p = model.predict(inp, verbose=0)
        future_price = close_prices[-1] * np.exp(p[0][0])

        # Build DataFrames
        past_dates = pd.date_range(end=df.index[-1], periods=num_predictions, freq="B").strftime("%Y-%m-%d")
        past_df = pd.DataFrame({
            "id": [str(uuid.uuid4()) for _ in range(num_predictions)],
            "Date": past_dates,
            "Predicted_Close": np.round(preds, 2),
        })
        past_df["Symbol"] = self.symbol

        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=1, freq="B").strftime("%Y-%m-%d")
        forecast_df = pd.DataFrame({
            "id": [str(uuid.uuid4()) for _ in range(1)],
            "Date": future_dates,
            "Predicted_Close": [round(future_price, 2)],
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
