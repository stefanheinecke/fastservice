import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enums import CloudProvider
from data import Predictor

def show_predictions():

    st.markdown(
        """
        <meta name="description" content="GoldPredicts – a machine learning powered gold price predictor. 
        Forecast tomorrow’s gold close, track accuracy, and explore trends in the world’s most trusted safe-haven asset.">
        """,
        unsafe_allow_html=True
    )

    # Tagline as a header
    st.title("✨ Gold Price Predictor")
    st.subheader("Forecast tomorrow’s gold price with data‑driven insights.")

    # Full description as markdown
    st.markdown("""
    Stay ahead of the market with our **Gold Price Predictor** — a smart tool that analyzes historical price movements and technical indicators to forecast the next trading day’s gold close.  

    By combining past trends with machine‑learning insights, it helps you spot potential shifts in direction, compare predicted vs. actual prices, and track accuracy over time.  

    Whether you’re an investor, trader, or simply curious about the world’s most trusted safe‑haven asset, this predictor gives you a quick, data‑driven glimpse into where gold might be heading next.
    """)

    st.markdown("---")

    cloud_provider = CloudProvider.GCP

    symbol = "GC=F"

    predict_obj = Predictor(cloud_provider.project_id, cloud_provider.dataset_id, cloud_provider.table_id, symbol)
    df, correct_direction_perc = predict_obj.fetch_prediction_history()
    df.index = df["Date"]
    df["Real_Close"] = pd.to_numeric(df["Real_Close"], errors="coerce")
    
    st.metric(label="Correct Direction (%)", value=f"{correct_direction_perc:.2f}")

    st.dataframe(df, use_container_width=True)  

    csv_df = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download as CSV", data=csv_df, file_name=f"historical_gold_prediction_report_{datetime.datetime.now()}.csv", mime="text/csv")

    # Plot with dates on x-axis
    actual = df["Real_Close"].head(100).values
    preds = df["Predicted_Close"].head(100).values
    future_preds = df["Predicted_Close"].head(1).values
    dates = pd.to_datetime(df["Date"].head(100))
    future_date = pd.to_datetime(df["Date"].head(1)).values[0]

    st.subheader("Actual vs. Predicted & 1-Day Forecast")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, actual, label="Actual", color="black")
    ax.plot(dates, preds, label="Predicted", color="orange")
    ax.plot([future_date], future_preds, linestyle="--", marker="o", label="Forecast", color="blue")
    ax.set_title("Forecast")
    ax.set_xlabel("Date")
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)

    # Display 1-Day Forecasted Prices
    st.subheader("Next 1-Day Forecast")
    future_dates = df["Date"].head(1).values
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": np.round(future_preds, 2)
    })
    st.dataframe(forecast_df, use_container_width=True)

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download as CSV", data=csv, file_name="future_gold_prediction_report.csv", mime="text/csv")

    adsense_code = """
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-9614544934238374"
        crossorigin="anonymous"></script>
    """

    st.markdown(adsense_code, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 📌 Disclaimer
    *The Gold Price Predictor is provided for informational and educational purposes only.  
    The forecasts shown are generated from historical data and machine‑learning models, and there is **no guarantee of accuracy or future performance**.  
    This tool should not be considered financial advice, and users should make investment decisions at their own discretion.*
    """)
