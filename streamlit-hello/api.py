from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import subprocess
import threading
import time

app = FastAPI()

# --- Your baskets data here (copy from app.py) ---
import pandas as pd
import numpy as np

baskets = {
    "Portfolio A": {
        "values": [1.1, 2.3, 3.3],
        "timeseries": pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=10),
            "Value": np.random.rand(10)*100,
            "Volume": np.random.randint(100, 200, 10),
            "Category": np.random.choice(["Tech", "Finance", "Retail"], 10)
        }),
        "components": {
            "2023-01-01": ["AAPL", "Apple"],
            "2023-01-02": ["Orange", "Grapes"],
        }
    },
    "Portfolio B": {
        "values": [4.4, 5.2],
        "timeseries": pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=10),
            "Value": np.random.rand(10)*50,
            "Volume": np.random.randint(50, 150, 10),
            "Category": np.random.choice(["Food", "Beverage", "Other"], 10)
        }),
        "components": {
            "2023-01-01": ["Bread", "Butter"],
            "2023-01-02": ["Cheese", "Ham"],
        }
    }
}

@app.get("/composition/{basket}/{date}")
def get_composition(basket: str, date: str):
    if basket not in baskets:
        raise HTTPException(status_code=404, detail="Basket not found")
    components = baskets[basket]["components"].get(date)
    if not components:
        raise HTTPException(status_code=404, detail="No data for this date")
    return {"basket": basket, "date": date, "components": components}

@app.get("/timeseries/{basket}")
def get_timeseries(basket: str):
    if basket not in baskets:
        raise HTTPException(status_code=404, detail="Basket not found")
    ts = baskets[basket]["timeseries"]
    timeseries = ts.to_dict(orient="records")
    return {"basket": basket, "timeseries": timeseries}

# --- Streamlit integration ---
def run_streamlit():
    subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])

@app.on_event("startup")
def startup_event():
    # Start Streamlit in a separate thread if not already running
    threading.Thread(target=run_streamlit, daemon=True).start()
    time.sleep(2)  # Give Streamlit time to start

@app.get("/streamlit")
def redirect_to_streamlit():
    # Redirect to the Streamlit app
    return RedirectResponse(url="http://localhost:8501")
