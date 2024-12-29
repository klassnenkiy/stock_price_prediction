import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_eda_page():
    st.header("Exploratory Data Analysis (EDA)")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        st.subheader("Descriptive Statistics")
        st.write(data.describe())

        st.subheader("Closing Price over Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['TRADEDATE'], data['CLOSE'], label="Closing Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title("Stock Closing Price Over Time")
        st.pyplot(fig)

        ticker = st.selectbox("Select Ticker", data['TICKER'].unique())
        ticker_data = data[data['TICKER'] == ticker]
        st.write(f"Data for {ticker}")
        st.write(ticker_data)

        st.subheader(f"Closing Price for {ticker}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ticker_data['TRADEDATE'], ticker_data['CLOSE'], label=f"{ticker} Closing Price", color="orange")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title(f"{ticker} Stock Price Over Time")
        st.pyplot(fig)
