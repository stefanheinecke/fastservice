import io
import os
import json
import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, send_file, Response, request
from data import Predictor
from sqlalchemy import create_engine, text
@app.route("/api/summary-predictions")
def api_summary_predictions():
    """Return next day prediction and last completed day's evaluation for all tickers."""
    engine = create_engine(DATABASE_URL)
    # Get all symbols
    with engine.connect() as conn:
        symbols = [r[0] for r in conn.execute(text("SELECT DISTINCT symbol FROM predictions ORDER BY symbol"))]

    results = []
    for symbol in symbols:
        predictor = Predictor(DATABASE_URL, symbol)
        # Next day prediction (for auction)
        next_pred = predictor.fetch_next_day_forecast()
        # Most recent completed day (with real_close)
        df, correct_direction_pct, mae = predictor.fetch_prediction_history(limit=30)
        if not df.empty:
            last_row = df.iloc[0]
            last_real_close = last_row["Real_Close"]
            last_pred_close = last_row["Predicted_Close"]
            # Calculate RMSE, MAPE for last 30 days
            y_true = df["Real_Close"].values
            y_pred = df["Predicted_Close"].values
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
            correct_direction = float(np.mean(df["Correct_Direction"])) * 100
        else:
            last_real_close = None
            last_pred_close = None
            rmse = None
            mape = None
            correct_direction = None
        results.append({
            "symbol": symbol,
            "next_pred_date": next_pred["date"] if next_pred else None,
            "next_pred_value": next_pred["predicted_close"] if next_pred else None,
            "last_real_close": last_real_close,
            "last_pred_close": last_pred_close,
            "mae": round(mae, 2) if mae is not None else None,
            "rmse": round(rmse, 2) if rmse is not None else None,
            "mape": round(mape, 2) if mape is not None else None,
            "correct_direction": round(correct_direction, 2) if correct_direction is not None else None,
        })
    return _json_response(results)

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
SYMBOL = "GC=F"


def _get_predictor(symbol=None):
    return Predictor(DATABASE_URL, symbol or SYMBOL)


def _json_response(data):
    """Serialize to JSON, converting NaN/Infinity to null."""
    body = json.dumps(data, default=str, allow_nan=False)
    return Response(body, mimetype="application/json")

@app.route("/")
def index():
    return render_template("index.html")


# -- API --

@app.route("/api/data-summary")
def api_data_summary():
    from sqlalchemy import create_engine, text
    engine = create_engine(DATABASE_URL)
    query = text("""
        SELECT symbol,
               MIN(date) AS min_date,
               MAX(date) AS max_date,
               COUNT(*) AS total_rows,
               COUNT(real_close) AS rows_with_real
        FROM predictions
        GROUP BY symbol
        ORDER BY symbol
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    rows = json.loads(df.to_json(orient="records", date_format="iso"))
    return _json_response(rows)


@app.route("/api/predictions")
def api_predictions():
    symbol = request.args.get("symbol", default=None, type=str)
    days = request.args.get("days", default=None, type=int)
    start = request.args.get("start", default=None, type=str)
    end = request.args.get("end", default=None, type=str)
    predictor = _get_predictor(symbol)
    df, correct_direction_perc, mae = predictor.fetch_prediction_history(
        limit=days, start_date=start, end_date=end
    )
    df["Date"] = df["Date"].astype(str)
    # Replace NaN/None so JSON serialization doesn't produce invalid tokens
    rows = json.loads(df.to_json(orient="records"))
    forecast = predictor.fetch_next_day_forecast()
    return _json_response({
        "correct_direction_pct": round(correct_direction_perc, 2),
        "mae": round(mae, 2),
        "rows": rows,
        "next_day": forecast,
    })


@app.route("/api/download")
def download_csv():
    predictor = _get_predictor()
    df, _, _ = predictor.fetch_prediction_history()
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    filename = f"gold_prediction_report_{datetime.date.today()}.csv"
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name=filename)


@app.route("/api/store_predictions", methods=["POST"])
def store_predictions():
    body = request.get_json(silent=True) or {}
    symbol = body.get("symbol", SYMBOL)
    predictor = _get_predictor(symbol)
    forecast_df, past_df, df = predictor.create_predictions()
    predictor.store_predictions(past_df)
    predictor.store_predictions(forecast_df)
    predictor.update_with_real_close(df)
    all_preds = pd.concat([past_df, forecast_df])
    min_date = str(all_preds["Date"].min())
    max_date = str(all_preds["Date"].max())
    total = len(all_preds)
    return _json_response({
        "message": f"Predictions for {symbol} stored successfully.",
        "min_date": min_date,
        "max_date": max_date,
        "total_rows": total,
    })


@app.route("/api/flush_predictions", methods=["POST"])
def flush_predictions():
    from sqlalchemy import create_engine, text as sa_text
    body = request.get_json(silent=True) or {}
    symbol = body.get("symbol", "").strip().upper()
    if not symbol:
        return _json_response({"error": "No symbol provided."}), 400
    engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        result = conn.execute(
            sa_text("DELETE FROM predictions WHERE symbol = :symbol"),
            {"symbol": symbol},
        )
        deleted = result.rowcount
    return _json_response({"message": f"Deleted {deleted} rows for {symbol}.", "deleted": deleted})


@app.route("/robots.txt")
def robots():
    return send_file("robots.txt", mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap():
    return send_file("sitemap.xml", mimetype="application/xml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)