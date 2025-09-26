import streamlit as st
import os
from predictions import show_predictions

st.set_page_config(page_title="Gold Price Predictor", layout="centered")


if st.request.path == "/robots.txt":
    st.text(open("robots.txt").read())


def show_about():
    st.title("About")
    st.write(
        "This app predicts gold prices using a trained TensorFlow model. "
        "Your gold price predictor is a machine learning-powered tool designed to "
        "forecast the next day's gold price based on historical market data and technical indicators. "
        "It uses a sliding window approach to capture patterns from the past 100 trading days, "
        "transforming this data into a feature-rich input for a neural network built with TensorFlow. "
        "The model incorporates indicators like RSI, MACD, Bollinger Bands, EMA, and volume-based metrics "
        "to enhance its understanding of market momentum, volatility, and trend behavior.\n\n"
        "Once trained, the model predicts future prices and evaluates its performance using metrics "
        "such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), "
        "Mean Absolute Percentage Error (MAPE), and directional accuracy. These metrics help assess "
        "not just how close the predictions are, but whether the model correctly anticipates upward or downward movements.\n\n"
        "The predictor also generates rolling forecasts and a one-day-ahead prediction, which can be visualized "
        "or stored for further analysis. It’s designed to be modular, scalable, and deployable via Cloud Run, "
        "making it suitable for integration into dashboards or automated workflows. Overall, "
        "it’s a robust tool for traders, analysts, or curious minds looking to anticipate "
        "gold price movements with data-driven precision."
    )

def show_contact():
    st.title("Contact")
    st.write("For inquiries, reach out to info@goldpredicts.com")

def show_privacy():
    st.title("Privacy Policy")
    st.write("We respect your privacy. No personal data is stored...")

# Top menu
tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "About", "Contact", "Privacy Policy"])

with tab1:
    show_predictions()
with tab2:
    show_about()
with tab3:
    show_contact()
with tab4:
    show_privacy()