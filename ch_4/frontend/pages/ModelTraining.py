import streamlit as st
import requests
import json


def show_model_training_page():
    st.header("Model Training")

    ticker = st.text_input("Enter Ticker (e.g., SBER)", "")
    window_size = st.number_input("Window Size", min_value=1, value=10)

    if st.button("Start Training"):
        if ticker:
            request_data = {"ticker": ticker, "window_size": window_size}
            response = requests.post("http://localhost:8000/fit/", json=request_data)
            if response.status_code == 200:
                st.success("Training started successfully.")
            else:
                st.error(f"Failed to start training: {response.text}")
        else:
            st.error("Please enter a ticker to train the model.")
