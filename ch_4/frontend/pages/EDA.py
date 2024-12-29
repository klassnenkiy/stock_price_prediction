import streamlit as st
import pandas as pd
import plotly.express as px

def show_eda_page():
    st.header("Exploratory Data Analysis (EDA)")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        st.subheader("Descriptive Statistics")
        st.write(data.describe())

        st.subheader("Closing Price over Time")
        fig = px.line(data, x='TRADEDATE', y='CLOSE', title="Stock Closing Price Over Time")
        st.plotly_chart(fig)

        ticker = st.selectbox("Select Ticker", data['TICKER'].unique())
        ticker_data = data[data['TICKER'] == ticker]
        st.write(f"Data for {ticker}")
        st.write(ticker_data)

        st.subheader(f"Closing Price for {ticker}")
        fig = px.line(ticker_data, x='TRADEDATE', y='CLOSE', title=f"{ticker} Stock Price Over Time")
        st.plotly_chart(fig)
