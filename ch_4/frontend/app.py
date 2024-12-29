import streamlit as st
from pages import EDA, ModelTraining, Predictions
import logging
from logging.handlers import RotatingFileHandler
import json_log_formatter



formatter = json_log_formatter.JSONFormatter()
file_handler = RotatingFileHandler("logs/streamlit.log", maxBytes=10**6, backupCount=5)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)


def main():
    st.title("Stock Price Prediction Dashboard")
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Choose a page", ["EDA", "Model Training", "Predictions"])

    if options == "EDA":
        EDA.show_eda_page()
    elif options == "Model Training":
        ModelTraining.show_model_training_page()
    elif options == "Predictions":
        Predictions.show_predictions_page()

if __name__ == "__main__":
    main()
