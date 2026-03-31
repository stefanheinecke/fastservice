import uuid
import gc
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Limit TensorFlow memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

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

    def store_prediction_stats(self, stats: dict, stat_date=None, window_size=30):
        """
        Store statistics in the prediction_stats table.
        stats: dict with keys: correct_direction, close_correct, mae, rmse, mape
        stat_date: date for which the stats are calculated (default: today)
        window_size: window size used for stats (default: 30)
        """
        if stat_date is None:
            stat_date = pd.Timestamp.today().date()
        record = {
            "symbol": self.symbol,
            "stat_date": stat_date,
            "window_size": window_size,
            "correct_direction": stats.get("correct_direction"),
            "close_correct": stats.get("close_correct"),
            "mae": stats.get("mae"),
            "rmse": stats.get("rmse"),
            "mape": stats.get("mape"),
        }
        df = pd.DataFrame([record])
        df.to_sql("prediction_stats", self.engine, if_exists="append", index=False)
        # Usage:
        # df, correct_direction_perc, mae = predictor.fetch_prediction_history(...)
        # stats = {"correct_direction": correct_direction_perc, ...}
        # predictor.store_prediction_stats(stats, stat_date, window_size)

    def __init__(self, database_url: str, symbol: str):
        self.engine = create_engine(database_url)
        self.symbol = symbol
        self._ensure_table()

    def _ensure_table(self):
        """Create the predictions and prediction_stats tables if they don't exist."""
        ddl_predictions = text(f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                predicted_close DOUBLE PRECISION,
                real_close DOUBLE PRECISION,
                real_open DOUBLE PRECISION
            )
        """)
        ddl_stats = text("""
            CREATE TABLE IF NOT EXISTS prediction_stats (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                stat_date DATE NOT NULL,
                window_size INTEGER NOT NULL,
                correct_direction DOUBLE PRECISION,
                close_correct DOUBLE PRECISION,
                mae DOUBLE PRECISION,
                rmse DOUBLE PRECISION,
                mape DOUBLE PRECISION,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        with self.engine.begin() as conn:
            conn.execute(ddl_predictions)
            conn.execute(ddl_stats)
            # Add real_open column if missing (existing tables)
            try:
                conn.execute(text(f"ALTER TABLE {self.TABLE_NAME} ADD COLUMN real_open DOUBLE PRECISION"))
            except Exception:
                pass  # column already exists

    def create_predictions(self):
        """
        Create and return predictions as DataFrames (forecast_df for future, past_df for historical).
        Does NOT store anything in the database.
        """
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
        # Directional / momentum features
        df["ma5"] = df["Close"].rolling(5).mean()
        df["ma20"] = df["Close"].rolling(20).mean()
        df["ma_cross"] = df["ma5"] / df["ma20"] - 1  # >0 = bullish, <0 = bearish
        df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]
        df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-8)
        df.dropna(inplace=True)

        feature_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "momentum_rsi", "trend_macd", "momentum_stoch",
            "volatility_bbm", "volatility_bbh", "volatility_bbl",
            "volatility_atr", "trend_ema_fast", "volume_obv",
            "return_1d", "return_5d", "return_10d", "return_20d",
            "rolling_vol_10", "rolling_vol_20", "close_to_ema",
            "ma_cross", "high_low_range", "close_position",
        ]
        close_prices = df["Close"].values.copy()
        df = df[feature_cols]
        num_features = len(feature_cols)

        # Target: raw log returns (preserves sign for direction)
        log_returns = np.log(close_prices[1:] / close_prices[:-1])

        window_size = 60
        features, labels = [], []
        for i in range(window_size, len(df)):
            window = df.iloc[i - window_size:i].values
            features.append(window)
            labels.append(log_returns[i - 1])

        X = np.array(features)
        del features  # free list memory
        y = np.array(labels).reshape(-1, 1)
        del labels

        # StandardScaler on features (in-place reshape to avoid extra copies)
        n_samples = X.shape[0]
        X_scaler = StandardScaler()
        X_flat = X.reshape(-1, num_features)
        X_scaler.fit(X_flat)
        X_scaled = X_scaler.transform(X_flat).reshape(n_samples, window_size, num_features)
        del X_flat, X  # free unscaled data

        # Chronological split
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Custom loss: Huber (robust to outliers) + directional BCE
        def direction_aware_loss(y_true, y_pred):
            huber = tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred, delta=0.01))
            pred_dir = tf.sigmoid(y_pred * 100.0)
            true_dir = tf.cast(y_true > 0, tf.float32)
            dir_bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(true_dir, pred_dir))
            return huber + 3.0 * dir_bce

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

        # --- Rolling predictions → convert back to prices (batched) ---
        num_predictions = len(df) - window_size
        all_windows = np.array([df.iloc[i - window_size:i].values
                                for i in range(window_size, len(df))])
        all_scaled = X_scaler.transform(
            all_windows.reshape(-1, num_features)
        ).reshape(num_predictions, window_size, num_features)
        del all_windows
        pred_returns = model.predict(all_scaled, batch_size=128, verbose=0).flatten()
        del all_scaled

        base_prices = close_prices[window_size - 1:-1]
        preds = base_prices * np.exp(pred_returns)

        # Forecast future (1 day)
        latest_window = df.iloc[-window_size:].values.copy()
        latest_scaled = X_scaler.transform(latest_window.reshape(-1, num_features)).reshape(1, window_size, num_features)
        p = model.predict(latest_scaled, verbose=0)
        future_price = close_prices[-1] * np.exp(p[0][0])

        # Free TF/model memory
        del model, X_scaler, latest_scaled, latest_window
        tf.keras.backend.clear_session()
        gc.collect()

        # Build DataFrames — use actual trading dates, NOT generated business days
        actual_dates = [str(d) for d in list(df.index)[window_size:]]
        past_df = pd.DataFrame({
            "id": [str(uuid.uuid4()) for _ in range(num_predictions)],
            "Date": actual_dates,
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

    def store_predictions(self, predictions_df):
        """
        Store predictions DataFrame (either past_df or forecast_df) in the predictions table.
        Handles deduplication by date and symbol.
        """
        print("Storing predictions in Postgres...")
        predictions_df = predictions_df[["id", "Symbol", "Date", "Predicted_Close"]].copy()
        predictions_df["Date"] = pd.to_datetime(predictions_df["Date"]).dt.date
        with self.engine.connect() as conn:
            existing = pd.read_sql(
                text(f"SELECT id, date, symbol FROM {self.TABLE_NAME}"),
                conn,
            )
        merged = predictions_df.merge(
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
        print("Updating real_close and real_open values in Postgres...")

        real_df = pd.DataFrame({
            "date": pd.to_datetime(df.index).date,
            "real_close": df["Close"].values,
            "real_open": df["Open"].values,
        })

        update_stmt = text(f"""
            UPDATE {self.TABLE_NAME}
            SET real_close = :real_close, real_open = :real_open
            WHERE date = :date AND symbol = :symbol
        """)

        with self.engine.begin() as conn:
            for _, row in real_df.iterrows():
                conn.execute(update_stmt, {"real_close": float(row["real_close"]),
                                           "real_open": float(row["real_open"]),
                                           "date": row["date"],
                                           "symbol": self.symbol})

        print("real_close and real_open columns updated for matching dates.")

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
                   real_close AS "Real_Close", predicted_close AS "Predicted_Close",
                   real_open AS "Real_Open"
            FROM {self.TABLE_NAME}
            WHERE {where_sql}
            ORDER BY date DESC
            {limit_sql}
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params)

        df["Real_Close"] = pd.to_numeric(df["Real_Close"], errors="coerce")
        df["Predicted_Close"] = pd.to_numeric(df["Predicted_Close"], errors="coerce")

        # Direction: did predicted close move same way as real close vs previous day's real close?
        prev_real = df["Real_Close"].shift(-1)  # previous day (data is DESC)
        df["Real_Close_diff"] = df["Real_Close"] - prev_real
        df["Predicted_Close_diff"] = df["Predicted_Close"] - prev_real

        df["Correct_Direction"] = (
            ((df["Real_Close_diff"] > 0) & (df["Predicted_Close_diff"] > 0)) |
            ((df["Real_Close_diff"] < 0) & (df["Predicted_Close_diff"] < 0))
        )

        # Close & Correct: direction right AND within 5% of real price
        pct_error = (abs(df["Predicted_Close"] - df["Real_Close"]) / df["Real_Close"])
        df["Close_Correct"] = df["Correct_Direction"] & (pct_error <= 0.05)

        correct_direction = df["Correct_Direction"].sum()
        correct_direction_perc = correct_direction / len(df) * 100 if len(df) > 0 else 0

        mae = float(np.mean(np.abs(df["Real_Close"] - df["Predicted_Close"]))) if len(df) > 0 else 0

        return df, correct_direction_perc, mae

    def fetch_next_day_forecast(self):
        """Return the most recent prediction that has no real_close yet (i.e. future)."""
        query = text(f"""
            SELECT date AS "Date", symbol AS "Symbol",
                   predicted_close AS "Predicted_Close"
            FROM {self.TABLE_NAME}
            WHERE symbol = :symbol AND real_close IS NULL
            ORDER BY date DESC
            LIMIT 1
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"symbol": self.symbol})
        if df.empty:
            return None
        row = df.iloc[0]
        return {"date": str(row["Date"]), "predicted_close": float(row["Predicted_Close"])}
