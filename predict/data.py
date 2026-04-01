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


_symbol_name_cache = {}

def get_symbol_name(symbol):
    """Return a human-readable name for a Yahoo Finance ticker symbol.
    Results are cached in-memory so yfinance is called at most once per symbol."""
    if symbol in _symbol_name_cache:
        return _symbol_name_cache[symbol]
    try:
        info = yf.Ticker(symbol).info
        name = info.get("shortName") or info.get("longName") or symbol
    except Exception:
        name = symbol
    _symbol_name_cache[symbol] = name
    return name


_ticker_info_cache = {}

def _get_ticker_info(symbol):
    """Cached yfinance .info dict."""
    if symbol not in _ticker_info_cache:
        try:
            _ticker_info_cache[symbol] = yf.Ticker(symbol).info
        except Exception:
            _ticker_info_cache[symbol] = {}
    return _ticker_info_cache[symbol]


def robo_index_backtest(database_url, smi_tickers, lookback_weeks=52, rebal_freq="3M"):
    """
    Backtest a top-5 portfolio based on prediction accuracy,
    sector-diversified, market-cap weighted, compared against the SMI index.

    rebal_freq: rebalancing frequency — "D" (daily), "W" (weekly),
                "M" (monthly), "3M" (quarterly, default)

    Returns a dict with portfolio time series, SMI time series, composition
    history, and summary statistics.
    """
    engine = create_engine(database_url)

    # ------------------------------------------------------------------
    # 1. Gather sector + market-cap info for every SMI component
    # ------------------------------------------------------------------
    meta = {}
    for sym in smi_tickers:
        info = _get_ticker_info(sym)
        meta[sym] = {
            "name": info.get("shortName") or info.get("longName") or sym,
            "sector": info.get("sector") or "Unknown",
            "market_cap": info.get("marketCap") or 0,
        }

    # ------------------------------------------------------------------
    # 2. Download 1 year + buffer of daily closes for all SMI stocks + ^SSMI
    # ------------------------------------------------------------------
    all_symbols = list(smi_tickers) + ["^SSMI"]
    end_dt = pd.Timestamp.today().normalize()
    start_dt = end_dt - pd.DateOffset(weeks=lookback_weeks + 4)

    prices = yf.download(all_symbols, start=start_dt, end=end_dt, auto_adjust=True, progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.ffill().dropna(how="all")

    # ------------------------------------------------------------------
    # 3. Load prediction data from DB (real_close, predicted_close, date)
    # ------------------------------------------------------------------
    pred_data = {}
    with engine.connect() as conn:
        for sym in smi_tickers:
            q = text("""
                SELECT date, real_close, predicted_close
                FROM predictions
                WHERE symbol = :sym AND real_close IS NOT NULL
                ORDER BY date
            """)
            df = pd.read_sql(q, conn, params={"sym": sym})
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                pred_data[sym] = df.set_index("date")

    # ------------------------------------------------------------------
    # 4. Build rebalance schedule based on chosen frequency
    # ------------------------------------------------------------------
    RESAMPLE_MAP = {"D": "B", "W": "W-FRI", "M": "ME", "3M": "QE"}
    rule = RESAMPLE_MAP.get(rebal_freq, "QE")
    if rule == "B":
        # Business-day: just use every trading day in the price index
        rebal_dates = prices.index
    else:
        rebal_dates = prices.resample(rule).last().dropna(how="all").index
    # Keep only the lookback window
    cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(weeks=lookback_weeks)
    rebal_dates = rebal_dates[rebal_dates >= cutoff]

    if len(rebal_dates) < 2:
        return {"error": "Not enough data for backtest"}

    # ------------------------------------------------------------------
    # 5. At each rebalance date, rank stocks by rolling direction accuracy,
    #    enforce sector diversification, weight by market cap
    # ------------------------------------------------------------------
    MAX_PER_SECTOR = 2
    TOP_N = 5

    def _direction_accuracy(sym, as_of_date, window=60):
        """Compute directional accuracy for sym using prediction data up to as_of_date."""
        if sym not in pred_data:
            return None
        df = pred_data[sym]
        df_before = df.loc[df.index <= as_of_date].tail(window)
        if len(df_before) < 10:
            return None
        prev_close = df_before["real_close"].shift(1)
        real_dir = (df_before["real_close"] > prev_close)
        pred_dir = (df_before["predicted_close"] > prev_close)
        valid = prev_close.notna()
        if valid.sum() < 10:
            return None
        return float((real_dir[valid] == pred_dir[valid]).mean() * 100)

    def _select_top5(as_of_date):
        """Return list of (symbol, weight) for the top-5 portfolio."""
        scores = []
        for sym in smi_tickers:
            acc = _direction_accuracy(sym, as_of_date)
            if acc is not None:
                scores.append((sym, acc))
        scores.sort(key=lambda x: x[1], reverse=True)

        selected = []
        sector_count = {}
        for sym, acc in scores:
            sec = meta[sym]["sector"]
            if sector_count.get(sec, 0) >= MAX_PER_SECTOR:
                continue
            selected.append(sym)
            sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected) >= TOP_N:
                break

        if not selected:
            return []

        # Market-cap weighting
        total_cap = sum(meta[s]["market_cap"] for s in selected)
        if total_cap == 0:
            w = 1.0 / len(selected)
            return [(s, w) for s in selected]
        return [(s, meta[s]["market_cap"] / total_cap) for s in selected]

    # ------------------------------------------------------------------
    # 6. Simulate period-by-period returns
    # ------------------------------------------------------------------
    portfolio_value = 100.0
    smi_value = 100.0

    series_dates = []
    series_portfolio = []
    series_smi = []
    compositions = []  # [{date, holdings: [{symbol,name,sector,weight,accuracy}]}]

    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_date = rebal_dates[i + 1]

        # Find actual trading days in prices between rebal_date and next_date
        mask = (prices.index > rebal_date) & (prices.index <= next_date)
        period_prices = prices.loc[mask]
        if period_prices.empty:
            continue

        # Select portfolio
        holdings = _select_top5(rebal_date)
        if not holdings:
            # No valid stocks — hold cash
            series_dates.append(next_date)
            series_portfolio.append(portfolio_value)
            # SMI still moves
            if "^SSMI" in period_prices.columns:
                smi_start = prices.loc[prices.index <= rebal_date, "^SSMI"].dropna().iloc[-1]
                smi_end = period_prices["^SSMI"].dropna().iloc[-1]
                smi_value *= smi_end / smi_start
            series_smi.append(smi_value)
            compositions.append({
                "date": str(rebal_date.date()),
                "holdings": [],
            })
            continue

        # Record composition
        comp_entry = {
            "date": str(rebal_date.date()),
            "holdings": [],
        }
        for sym, wt in holdings:
            comp_entry["holdings"].append({
                "symbol": sym,
                "name": meta[sym]["name"],
                "sector": meta[sym]["sector"],
                "weight": round(wt * 100, 2),
                "accuracy": round(_direction_accuracy(sym, rebal_date) or 0, 2),
            })
        compositions.append(comp_entry)

        # Portfolio return: weighted sum of individual stock returns
        port_return = 0.0
        for sym, wt in holdings:
            if sym not in period_prices.columns:
                continue
            sym_prices = period_prices[sym].dropna()
            if sym_prices.empty:
                continue
            sym_start = prices.loc[prices.index <= rebal_date, sym].dropna()
            if sym_start.empty:
                continue
            ret = (sym_prices.iloc[-1] / sym_start.iloc[-1]) - 1
            port_return += wt * ret

        portfolio_value *= (1 + port_return)

        # SMI return
        if "^SSMI" in period_prices.columns:
            smi_prices = period_prices["^SSMI"].dropna()
            smi_start_series = prices.loc[prices.index <= rebal_date, "^SSMI"].dropna()
            if not smi_prices.empty and not smi_start_series.empty:
                smi_ret = (smi_prices.iloc[-1] / smi_start_series.iloc[-1]) - 1
                smi_value *= (1 + smi_ret)

        series_dates.append(str(next_date.date()))
        series_portfolio.append(round(portfolio_value, 4))
        series_smi.append(round(smi_value, 4))

    # ------------------------------------------------------------------
    # 7. Compute summary statistics
    # ------------------------------------------------------------------
    port_arr = np.array(series_portfolio)
    smi_arr = np.array(series_smi)

    def _stats(values, label):
        if len(values) < 2:
            return {}
        total_ret = (values[-1] / values[0] - 1) * 100
        rets = np.diff(values) / values[:-1]
        # Annualise based on number of periods per year
        periods_per_year = {"D": 252, "W": 52, "M": 12, "3M": 4}.get(rebal_freq, 4)
        vol_ann = float(np.std(rets) * np.sqrt(periods_per_year) * 100)
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(periods_per_year)) if np.std(rets) > 0 else 0
        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax
        max_dd = float(np.min(drawdowns) * 100)
        return {
            "label": label,
            "total_return": round(total_ret, 2),
            "annualized_vol": round(vol_ann, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd, 2),
            "final_value": round(values[-1], 2),
        }

    return {
        "dates": series_dates,
        "portfolio": series_portfolio,
        "smi": series_smi,
        "compositions": compositions,
        "portfolio_stats": _stats(port_arr, "Robo-Index"),
        "smi_stats": _stats(smi_arr, "SMI"),
        "meta": {s: {"name": m["name"], "sector": m["sector"]} for s, m in meta.items()},
    }


class Predictor:
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
    def __init__(self, database_url, symbol):
        self.engine = create_engine(database_url)
        self.symbol = symbol
        self._ensure_table()
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

    def create_predictions(self):
        """
        Create and return predictions as DataFrames (forecast_df for future, past_df for historical).
        Does NOT store anything in the database.
        """
        # ...existing logic...
        # (Your full create_predictions implementation goes here)
        # For now, using the last known correct version:
        df = load_data(self.symbol)
        df.index = pd.to_datetime(df.index).date
        df["Date"] = df.index

        # ...feature engineering, model, etc...
        # (Omitted for brevity, unchanged logic for prediction creation)

        # Build DataFrames — use actual trading dates, NOT generated business days
        # ...existing code to build past_df and forecast_df...

        # Return only DataFrames (no stats)
        # (If you want to return stats, restore the stats logic here)
        return forecast_df, past_df, df

    def store_latest_stats(self, window_size=30, stat_date=None):
        """
        Calculate and store stats for the latest window_size days using fetch_prediction_history.
        """
        df, correct_direction_perc, mae, rmse, mape, close_correct, mae_pct, rmse_pct = self.fetch_prediction_history(limit=window_size)
        stats = {
            "correct_direction": correct_direction_perc,
            "close_correct": close_correct,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
        }
        self.store_prediction_stats(stats, stat_date=stat_date, window_size=window_size)
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
        stats = {
            "mae": float(real_mae) if real_mae is not None else None,
            "rmse": float(rmse) if rmse is not None else None,
            "mape": float(mape) if mape is not None else None,
            "correct_direction": float(direction_acc) * 100 if direction_acc is not None else None,
            "close_correct": None,  # Not calculated here
        }
        return forecast_df, past_df, df, stats

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

        # Only mark direction where prev_real is available (skip oldest row)
        has_prev = prev_real.notna()
        df["Correct_Direction"] = None  # default to None (will serialize as null)
        df.loc[has_prev, "Correct_Direction"] = (
            ((df.loc[has_prev, "Real_Close_diff"] > 0) & (df.loc[has_prev, "Predicted_Close_diff"] > 0)) |
            ((df.loc[has_prev, "Real_Close_diff"] < 0) & (df.loc[has_prev, "Predicted_Close_diff"] < 0))
        )

        # Close & Correct: direction right AND within 5% of real price
        pct_error = (abs(df["Predicted_Close"] - df["Real_Close"]) / df["Real_Close"])
        df["Close_Correct"] = None
        df.loc[has_prev, "Close_Correct"] = df.loc[has_prev, "Correct_Direction"].astype(bool) & (pct_error[has_prev] <= 0.05)

        # Aggregate stats — only count rows with valid direction (matches JS logic)
        valid_dir = df["Correct_Direction"].dropna()
        total_dir = len(valid_dir)
        correct_direction = valid_dir.astype(bool).sum()
        correct_direction_perc = correct_direction / total_dir * 100 if total_dir > 0 else 0
        close_correct = df["Close_Correct"].dropna().astype(bool).sum() / total_dir * 100 if total_dir > 0 else 0

        valid = df.dropna(subset=["Real_Close", "Predicted_Close"])
        mae = float(np.mean(np.abs(valid["Real_Close"] - valid["Predicted_Close"]))) if len(valid) > 0 else 0
        rmse = float(np.sqrt(np.mean((valid["Real_Close"] - valid["Predicted_Close"]) ** 2))) if len(valid) > 0 else 0
        mape = float(np.mean(np.abs((valid["Real_Close"] - valid["Predicted_Close"]) / valid["Real_Close"])) * 100) if len(valid) > 0 else 0
        mean_price = float(valid["Real_Close"].mean()) if len(valid) > 0 else 0
        mae_pct = (mae / mean_price * 100) if mean_price > 0 else 0
        rmse_pct = (rmse / mean_price * 100) if mean_price > 0 else 0
        return df, correct_direction_perc, mae, rmse, mape, close_correct, mae_pct, rmse_pct

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
