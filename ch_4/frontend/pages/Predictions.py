import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go


def show_predictions_page():
    st.header("Model Predictions")
    ticker = st.text_input("Enter Ticker (e.g., SBER)", "")
    forecast_days = st.number_input("Forecast Days", min_value=1, value=30)

    if st.button("Get Predictions"):
        if ticker:
            request_data = {"ticker": ticker, "X_input": [[1] * forecast_days]}
            response = requests.post("http://backend:8000/predict/", json=request_data)
            if response.status_code == 200:
                prediction_data = response.json()
                forecast_dates = prediction_data["forecast_dates"]
                forecast_prices = prediction_data["forecast_prices"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_prices, mode='lines', name='Predicted Prices'))
                fig.update_layout(title="Stock Price Predictions", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig)
            else:
                st.error(f"Failed to fetch predictions: {response.text}")
        else:
            st.error("Please enter a ticker to get predictions.")

    st.subheader("Compare Models")
    comparison_file = st.file_uploader("Upload Model Comparison Data (CSV)", type=["csv"])
    if comparison_file:
        comparison_data = pd.read_csv(comparison_file)
        fig = go.Figure()
        for model in comparison_data['Model'].unique():
            model_data = comparison_data[comparison_data['Model'] == model]
            fig.add_trace(go.Scatter(x=model_data['Date'], y=model_data['Price'], mode='lines', name=f'Model: {model}'))
        fig.update_layout(title="Model Comparison", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig)