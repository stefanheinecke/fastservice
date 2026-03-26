import io
import os
import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, send_file
from data import Predictor

app = Flask(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL")
SYMBOL = "GC=F"


def _get_predictor():
    return Predictor(DATABASE_URL, SYMBOL)


# -- Pages --

@app.route("/")
def index():
    return render_template("index.html")


# -- API --

@app.route("/api/predictions")
def api_predictions():
    predictor = _get_predictor()
    df, correct_direction_perc = predictor.fetch_prediction_history()
    df["Date"] = df["Date"].astype(str)
    return jsonify({
        "correct_direction_pct": round(correct_direction_perc, 2),
        "rows": df.to_dict(orient="records"),
    })


@app.route("/api/download")
def download_csv():
    predictor = _get_predictor()
    df, _ = predictor.fetch_prediction_history()
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
    return jsonify({"message": "Predictions stored successfully."})


@app.route("/robots.txt")
def robots():
    return send_file("robots.txt", mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap():
    return send_file("sitemap.xml", mimetype="application/xml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)