@app.route("/api/summary-predictions/<symbol>")
def api_summary_predictions_symbol(symbol):
    """Return next day prediction and last 30-day evaluation for a single ticker."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        # Next-day forecast for this symbol
        next_day_row = conn.execute(text("""
            SELECT date, predicted_close
            FROM predictions
            WHERE real_close IS NULL AND symbol = :symbol
            ORDER BY date DESC
            LIMIT 1
        """), {"symbol": symbol}).fetchone()
        next_day = {"date": str(next_day_row[0]), "predicted_close": float(next_day_row[1])} if next_day_row else None

        # Stats from last 30 completed days for this symbol
        stats_row = conn.execute(text("""
            WITH ranked AS (
                SELECT date, real_close, predicted_close,
                       LAG(real_close) OVER (ORDER BY date) AS prev_real_close,
                       ROW_NUMBER() OVER (ORDER BY date DESC) AS rn
                FROM predictions
                WHERE real_close IS NOT NULL AND symbol = :symbol
            ),
            last30 AS (
                SELECT * FROM ranked WHERE rn <= 30
            )
            SELECT
                MAX(CASE WHEN rn = 1 THEN real_close END) AS last_real_close,
                MAX(CASE WHEN rn = 1 THEN predicted_close END) AS last_pred_close,
                AVG(ABS(real_close - predicted_close)) AS mae,
                SQRT(AVG(POWER(real_close - predicted_close, 2))) AS rmse,
                AVG(ABS(real_close - predicted_close) / NULLIF(real_close, 0)) * 100 AS mape,
                AVG(CASE WHEN prev_real_close IS NOT NULL AND (
                    (real_close - prev_real_close > 0 AND predicted_close - prev_real_close > 0) OR
                    (real_close - prev_real_close < 0 AND predicted_close - prev_real_close < 0)
                ) THEN 1.0 ELSE 0.0 END) * 100 AS correct_direction
            FROM last30
        """), {"symbol": symbol}).fetchone()

    result = {
        "symbol": symbol,
        "next_pred_date": next_day["date"] if next_day else None,
        "next_pred_value": next_day["predicted_close"] if next_day else None,
        "last_real_close": round(float(stats_row[0]), 2) if stats_row and stats_row[0] is not None else None,
        "last_pred_close": round(float(stats_row[1]), 2) if stats_row and stats_row[1] is not None else None,
        "mae": round(float(stats_row[2]), 2) if stats_row and stats_row[2] is not None else None,
        "rmse": round(float(stats_row[3]), 2) if stats_row and stats_row[3] is not None else None,
        "mape": round(float(stats_row[4]), 2) if stats_row and stats_row[4] is not None else None,
        "correct_direction": round(float(stats_row[5]), 2) if stats_row and stats_row[5] is not None else None,
    }
    return _json_response(result)
import io
import os
import json
import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, send_file, Response, request
from data import Predictor
from sqlalchemy import create_engine, text

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

@app.route("/api/summary-predictions")
def api_summary_predictions():
    """Return next day prediction and last 30-day evaluation for all tickers (two fast SQL queries)."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        # 1) Next-day forecasts (rows with no real_close yet)
        next_day_rows = conn.execute(text("""
            SELECT DISTINCT ON (symbol) symbol, date, predicted_close
            FROM predictions
            WHERE real_close IS NULL
            ORDER BY symbol, date DESC
        """)).fetchall()
        next_day_map = {r[0]: {"date": str(r[1]), "predicted_close": float(r[2])} for r in next_day_rows}

        # 2) Stats from last 30 completed days per ticker (single query with window functions)
        stats_rows = conn.execute(text("""
            WITH ranked AS (
                SELECT symbol, date, real_close, predicted_close,
                       LAG(real_close) OVER (PARTITION BY symbol ORDER BY date) AS prev_real_close,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
                FROM predictions
                WHERE real_close IS NOT NULL
            ),
            last30 AS (
                SELECT * FROM ranked WHERE rn <= 30
            )
            SELECT symbol,
                   MAX(CASE WHEN rn = 1 THEN real_close END) AS last_real_close,
                   MAX(CASE WHEN rn = 1 THEN predicted_close END) AS last_pred_close,
                   AVG(ABS(real_close - predicted_close)) AS mae,
                   SQRT(AVG(POWER(real_close - predicted_close, 2))) AS rmse,
                   AVG(ABS(real_close - predicted_close) / NULLIF(real_close, 0)) * 100 AS mape,
                   AVG(CASE WHEN prev_real_close IS NOT NULL AND (
                       (real_close - prev_real_close > 0 AND predicted_close - prev_real_close > 0) OR
                       (real_close - prev_real_close < 0 AND predicted_close - prev_real_close < 0)
                   ) THEN 1.0 ELSE 0.0 END) * 100 AS correct_direction
            FROM last30
            GROUP BY symbol
            ORDER BY symbol
        """)).fetchall()

    results = []
    for r in stats_rows:
        sym = r[0]
        nd = next_day_map.get(sym)
        results.append({
            "symbol": sym,
            "next_pred_date": nd["date"] if nd else None,
            "next_pred_value": nd["predicted_close"] if nd else None,
            "last_real_close": round(float(r[1]), 2) if r[1] is not None else None,
            "last_pred_close": round(float(r[2]), 2) if r[2] is not None else None,
            "mae": round(float(r[3]), 2) if r[3] is not None else None,
            "rmse": round(float(r[4]), 2) if r[4] is not None else None,
            "mape": round(float(r[5]), 2) if r[5] is not None else None,
            "correct_direction": round(float(r[6]), 2) if r[6] is not None else None,
        })
    return _json_response(results)

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