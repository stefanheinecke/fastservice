import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enums import CloudProvider
from data import Predictor


st.set_page_config(page_title="Forecast", layout="wide")
st.title("Prediction")

adsense_code = """
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-9614544934238374"
     crossorigin="anonymous"></script>
"""

st.markdown(adsense_code, unsafe_allow_html=True)

cloud_provider = CloudProvider.GCP

symbol = "GC=F"

predict_obj = Predictor(cloud_provider.project_id, cloud_provider.dataset_id, cloud_provider.table_id, symbol)
df, correct_direction_perc = predict_obj.fetch_prediction_history()
df.index = df["Date"]
df["Real_Close"] = pd.to_numeric(df["Real_Close"], errors="coerce")
st.dataframe(df, use_container_width=True)

st.metric(label="Correct Direction (%)", value=f"{correct_direction_perc:.2f}")

# Plot with dates on x-axis
actual = df["Real_Close"].head(10).values
preds = df["Predicted_Close"].head(10).values
future_preds = df["Predicted_Close"].head(1).values
dates = pd.to_datetime(df["Date"].head(10))
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
st.download_button("📥 Download CSV Report", data=csv, file_name="sx5e_prediction_report.csv", mime="text/csv")
