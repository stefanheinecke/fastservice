import data
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/sum")
def sum_numbers(a: float = Query(...), b: float = Query(...)):
    result = a + b
    return JSONResponse(content={"sum": result})
    
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
    print(f"Storing predictions for {symbol}...")
    # Create predictions and store them in BigQuery
    forecast_df = data.create_predictions(symbol)
    print(f"Forecast DataFrame for {symbol}:\n{forecast_df}")
    # Ensure the DataFrame has the correct columns
    data.store_predictions(forecast_df)
    return JSONResponse(content={"message": "Predictions stored successfully."})
