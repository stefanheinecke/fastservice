import data
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from data import Predictor
from enums import CloudProvider

app = FastAPI()

cloud_provider = CloudProvider.GCP
    
@app.get("/history")
def get_history(symbol: str = Query(...)):
    history_df = data.load_data(symbol)
    history_data = history_df.to_dict(orient="records")
    return JSONResponse(content={"history": list(history_data)})

@app.get("/predictions")
def get_predictions(symbol: str = Query(...)):
    forecast_df = data.create_predictions(symbol)
    forecast_data = forecast_df.to_dict(orient="records")
    return JSONResponse(content={"predictions": list(forecast_data)})

@app.get("/store_predictions")
def store_predictions(symbol: str = Query(...)):

    predict_obj = Predictor(cloud_provider.project_id, cloud_provider.dataset_id, cloud_provider.table_id, symbol)
    print(f"Storing predictions for {symbol}...")

    forecast_df, df = predict_obj.create_predictions()
    print(f"Forecast DataFrame for {symbol}:\n{forecast_df}")

    predict_obj.store_predictions(forecast_df)
    print(f"Stored Predictions for {symbol} in BigQuery.")

    predict_obj.update_with_real_close(df)

    return JSONResponse(content={"message": "Predictions stored successfully."})

@app.get("/dashboard")
def prediction_history(
    symbol: str = Query(...),
    date: str = Query(...)  # Format: YYYY-MM-DD
):
    predict_obj = Predictor(cloud_provider.project_id, cloud_provider.dataset_id, cloud_provider.table_id, symbol)
    df = predict_obj.fetch_prediction_history(date)
    df["Created_at"] = df["Created_at"].astype(str)
    df["Real_Close"] = df["Real_Close"].apply(lambda x: "NaN" if pd.isna(x) else x)
    return JSONResponse(content={"data": df.to_dict(orient="records")})
