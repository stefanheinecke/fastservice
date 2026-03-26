# this is fastservice – SX5E Stock Prediction

A deep-learning web app that forecasts EURO STOXX 50 (SX5E) closing prices using TensorFlow and technical indicators.

## Stack

- **Backend:** Flask + Gunicorn
- **Frontend:** HTML / CSS / JavaScript (no framework)
- **ML:** TensorFlow, scikit-learn, ta (technical analysis)
- **Data:** Yahoo Finance via yfinance

## Local Development

```bash
cd streamlit-hello
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8080` in your browser.

## Deploy to Railway

1. Connect your GitHub repo to [Railway](https://railway.app).
2. Set the **Root Directory** to `streamlit-hello`.
3. Railway auto-detects the `Procfile` and deploys with Gunicorn.

No Dockerfile or GitHub Actions needed — Railway handles the build.