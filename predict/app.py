import io
import os
import json
import datetime

import numpy as np
import pandas as pd
from flask import Flask, render_template, send_file, Response, request
from data import Predictor, get_symbol_name, robo_index_backtest
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

# Restore /api/summary-predictions endpoint
@app.route("/api/summary-predictions")
def api_summary_predictions():
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            symbols = pd.read_sql("SELECT DISTINCT symbol FROM predictions ORDER BY symbol", conn)["symbol"].tolist()
        results = []
        for symbol in symbols:
            try:
                predictor = _get_predictor(symbol)
                df, correct_direction_perc, mae, rmse, mape, close_correct, mae_pct, rmse_pct = predictor.fetch_prediction_history()
                forecast = predictor.fetch_next_day_forecast()
                next_pred_date = None
                next_pred_value = None
                if forecast:
                    next_pred_date = forecast.get("date")
                    next_pred_value = forecast.get("predicted_close")
                last_real_close = round(float(df.iloc[0]["Real_Close"]), 2) if not df.empty and pd.notna(df.iloc[0]["Real_Close"]) else None
                last_real_close_date = str(df.iloc[0]["Date"]) if not df.empty else None
                last_pred_close = round(float(df.iloc[0]["Predicted_Close"]), 2) if not df.empty and pd.notna(df.iloc[0]["Predicted_Close"]) else None
                results.append({
                    "symbol": symbol,
                    "name": get_symbol_name(symbol),
                    "next_pred_date": next_pred_date,
                    "next_pred_value": next_pred_value,
                    "last_real_close": last_real_close,
                    "last_real_close_date": last_real_close_date,
                    "last_pred_close": last_pred_close,
                    "mae": round(mae, 2) if mae is not None else None,
                    "mae_pct": round(mae_pct, 2) if mae_pct is not None else None,
                    "rmse": round(rmse, 2) if rmse is not None else None,
                    "rmse_pct": round(rmse_pct, 2) if rmse_pct is not None else None,
                    "mape": round(mape, 2) if mape is not None else None,
                    "correct_direction": round(correct_direction_perc, 2) if correct_direction_perc is not None else None,
                    "close_correct": round(close_correct, 2) if close_correct is not None else None,
                })
            except Exception as e:
                import traceback
                print(f"Error processing symbol {symbol}: {e}\n{traceback.format_exc()}")
        return _json_response(results)
    except Exception as e:
        import traceback
        print(f"Error in /api/summary-predictions: {e}\n{traceback.format_exc()}")
        return _json_response({"error": str(e), "trace": traceback.format_exc()}), 500

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
    for row in rows:
        row["name"] = get_symbol_name(row["symbol"]) if row.get("symbol") else None
    return _json_response(rows)


@app.route("/api/index-components/<index_name>")
def api_index_components(index_name):
    """Return ticker symbols for a known stock index."""
    INDEX_COMPONENTS = {
        "smi": [
            "ABBN.SW", "ALC.SW", "CFR.SW", "GEBN.SW", "GIVN.SW",
            "HOLN.SW", "KNIN.SW", "LONN.SW", "NESN.SW", "NOVN.SW",
            "PGHN.SW", "ROG.SW", "SCMN.SW", "SDZ.SW", "SIKA.SW",
            "SLHN.SW", "SOON.SW", "SREN.SW", "UBSG.SW", "ZURN.SW",
        ],
    }
    name = index_name.lower()
    if name not in INDEX_COMPONENTS:
        return _json_response({"error": f"Unknown index: {index_name}"}), 404
    return _json_response({"index": index_name, "tickers": INDEX_COMPONENTS[name]})


SMI_TICKERS = [
    "ABBN.SW", "ALC.SW", "CFR.SW", "GEBN.SW", "GIVN.SW",
    "HOLN.SW", "KNIN.SW", "LONN.SW", "NESN.SW", "NOVN.SW",
    "PGHN.SW", "ROG.SW", "SCMN.SW", "SDZ.SW", "SIKA.SW",
    "SLHN.SW", "SOON.SW", "SREN.SW", "UBSG.SW", "ZURN.SW",
]


@app.route("/api/robo-index")
def api_robo_index():
    """Run the Robo-Index backtest and return results."""
    weeks = request.args.get("weeks", default=52, type=int)
    weeks = min(max(weeks, 4), 156)  # clamp 4–156 weeks
    rebal = request.args.get("rebal", default="3M", type=str)
    if rebal not in ("D", "W", "M", "3M"):
        rebal = "3M"
    try:
        result = robo_index_backtest(DATABASE_URL, SMI_TICKERS, lookback_weeks=weeks, rebal_freq=rebal)
        return _json_response(result)
    except Exception as e:
        import traceback
        print(f"Error in /api/robo-index: {e}\n{traceback.format_exc()}")
        return _json_response({"error": str(e)}), 500


@app.route("/api/predictions")
def api_predictions():
    symbol = request.args.get("symbol", default=None, type=str)
    days = request.args.get("days", default=None, type=int)
    start = request.args.get("start", default=None, type=str)
    end = request.args.get("end", default=None, type=str)
    predictor = _get_predictor(symbol)
    df, correct_direction_perc, mae, rmse, mape, close_correct, mae_pct, rmse_pct = predictor.fetch_prediction_history(
        limit=days, start_date=start, end_date=end
    )
    df["Date"] = df["Date"].astype(str)
    # Replace NaN/None so JSON serialization doesn't produce invalid tokens
    rows = json.loads(df.to_json(orient="records"))
    forecast = predictor.fetch_next_day_forecast()
    return _json_response({
        "correct_direction_pct": round(correct_direction_perc, 2),
        "mae": round(mae, 2),
        "mae_pct": round(mae_pct, 2),
        "rmse": round(rmse, 2),
        "rmse_pct": round(rmse_pct, 2),
        "mape": round(mape, 2),
        "close_correct": round(close_correct, 2),
        "rows": rows,
        "next_day": forecast,
    })


@app.route("/api/download")
def download_csv():
    predictor = _get_predictor()
    df, *_ = predictor.fetch_prediction_history()
    buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    filename = f"gold_prediction_report_{datetime.date.today()}.csv"
    return send_file(buf, mimetype="text/csv", as_attachment=True,
                     download_name=filename)


@app.route("/api/store_predictions", methods=["POST"])
def store_predictions():
    body = request.get_json(silent=True) or {}
    symbol = body.get("symbol", SYMBOL)
    try:
        predictor = _get_predictor(symbol)
        forecast_df, past_df, df = predictor.create_predictions()
        predictor.store_predictions(past_df)
        predictor.store_predictions(forecast_df)
        predictor.update_with_real_close(df)
        # Calculate and store stats after storing predictions
        predictor.store_latest_stats()
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
    except Exception as e:
        return _json_response({"error": f"Failed to create predictions for {symbol}: {str(e)}"}), 500


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