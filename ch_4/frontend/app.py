import streamlit as st
from pages import EDA, ModelTraining, Predictions
import logging
from logging.handlers import RotatingFileHandler


log_handler = RotatingFileHandler("logs/streamlit.log", maxBytes=10**6, backupCount=5)
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, handlers=[log_handler], format=log_format)

logger = logging.getLogger(__name__)

logger.info("Streamlit app has started")


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
