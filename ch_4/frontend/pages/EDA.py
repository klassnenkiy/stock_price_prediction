import streamlit as st
import pandas as pd
import plotly.express as px


def show_eda_page():
    st.header("Exploratory Data Analysis (EDA)")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data['TRADEDATE'] = pd.to_datetime(data['TRADEDATE'], errors='coerce')

        invalid_dates = data[data['TRADEDATE'].isnull()]
        if not invalid_dates.empty:
            st.warning(
                f"Некоторые значения в столбце 'TRADEDATE' не удалось преобразовать в даты. Это следующие строки:")
            st.write(invalid_dates)

        st.write("Первоначальные данные:")
        st.write(data.head())

        st.subheader("Descriptive Statistics")
        st.write(data.describe())

        st.subheader("Closing Price Over Time")
        if 'CLOSE' in data.columns:
            fig = px.line(data, x='TRADEDATE', y='CLOSE', title="Stock Closing Price Over Time")
            st.plotly_chart(fig)
        else:
            st.error("Столбец 'CLOSE' не найден в данных.")

        if 'TICKER' in data.columns:
            ticker = st.selectbox("Select Ticker", data['TICKER'].unique())
            ticker_data = data[data['TICKER'] == ticker]
            st.write(f"Data for {ticker}")
            st.write(ticker_data)

            st.subheader(f"Closing Price for {ticker}")
            fig = px.line(ticker_data, x='TRADEDATE', y='CLOSE', title=f"{ticker} Stock Price Over Time")
            st.plotly_chart(fig)
