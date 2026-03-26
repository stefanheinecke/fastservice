import io
import os
import json
import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, send_file, Response, request
from data import Predictor

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
SYMBOL = "GC=F"


def _get_predictor():
    return Predictor(DATABASE_URL, SYMBOL)


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
    days = request.args.get("days", default=None, type=int)
    start = request.args.get("start", default=None, type=str)
    end = request.args.get("end", default=None, type=str)
    predictor = _get_predictor()
    df, correct_direction_perc, mae = predictor.fetch_prediction_history(
        limit=days, start_date=start, end_date=end
    )
    df["Date"] = df["Date"].astype(str)
    # Replace NaN/None so JSON serialization doesn't produce invalid tokens
    rows = json.loads(df.to_json(orient="records"))
    return _json_response({
        "correct_direction_pct": round(correct_direction_perc, 2),
        "mae": round(mae, 2),
        "rows": rows,
    })


@app.route("/api/download")
def download_csv():
    predictor = _get_predictor()
    df, _, _ = predictor.fetch_prediction_history()
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    filename = f"gold_prediction_report_{datetime.date.today()}.csv"
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name=filename)


@app.route("/api/store_predictions")
def store_predictions():
    predictor = _get_predictor()
    forecast_df, past_df, df = predictor.create_predictions()
    predictor.store_predictions(past_df)
    predictor.store_predictions(forecast_df)
    predictor.update_with_real_close(df)
    return _json_response({"message": "Predictions stored successfully."})


@app.route("/robots.txt")
def robots():
    return send_file("robots.txt", mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap():
    return send_file("sitemap.xml", mimetype="application/xml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)