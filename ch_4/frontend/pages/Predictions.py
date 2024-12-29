import streamlit as st
import requests
import json


def show_predictions_page():
    st.header("Model Predictions")
    ticker = st.text_input("Enter Ticker (e.g., SBER)", "")
    forecast_days = st.number_input("Forecast Days", min_value=1, value=30)

    if st.button("Get Predictions"):
        if ticker:
            request_data = {"ticker": ticker, "forecast_days": forecast_days}
            response = requests.post("http://localhost:8000/predict/", json=request_data)
            if response.status_code == 200:
                prediction_data = response.json()
                st.write(f"Forecasted Dates: {prediction_data['forecast_dates']}")
                st.write(f"Forecasted Prices: {prediction_data['forecast_prices']}")
            else:
                st.error(f"Failed to fetch predictions: {response.text}")
        else:
            st.error("Please enter a ticker to get predictions.")
