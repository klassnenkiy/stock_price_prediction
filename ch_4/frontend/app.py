import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import logging
import joblib
from logging.handlers import RotatingFileHandler


api_url = "http://backend:8000"


log_filename = "/var/log/app/frontend/app.log"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def prepare_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    logging.info(f"Data prepared with window size {window_size}.")
    return np.array(X), np.array(y)


def load_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Загруженные данные:")
        st.write(data.head())
        logging.info("Data uploaded successfully.")
        return data
    logging.warning("No file uploaded.")
    return None


def load_model(ticker):
    model_filename = f"models/model_{ticker}.joblib"
    if os.path.exists(model_filename):
        logging.info(f"Loading model for ticker {ticker}.")
        return joblib.load(model_filename)
    else:
        st.error(f"Модель для {ticker} не найдена.")
        logging.error(f"Model for {ticker} not found.")
        return None


def compare_models(tickers):
    params = {"tickers": tickers}
    response = requests.get(f"{api_url}/compare_models", params=params)
    if response.status_code == 200:
        model_data = response.json()
        logging.info(f"Fetched model data for comparison: {model_data}")
        return model_data
    else:
        st.error("Ошибка при получении данных для сравнения моделей.")
        logging.error(f"Error fetching model comparison data: {response.text}")
        return []


def show_eda(data):
    st.header("Exploratory Data Analysis (EDA)")
    if data is not None:
        st.subheader("Основная статистика данных:")
        st.write(data.describe())
        st.subheader("Исторические цены закрытия для предзагруженных тикеров:")
        first_tier_tickers = ["SBER", "SBERP", "ROSN", "SIBN", "NVTK", "LKOH", "GAZP", "PLZL", "TATN", "TATNP", "SNGS",
                              "SNGSP", "CHMF", "NLMK"]
        first_tier_data = data[data['TICKER'].isin(first_tier_tickers)]
        first_tier_data['TRADEDATE'] = pd.to_datetime(first_tier_data['TRADEDATE'])
        first_tier_data.set_index('TRADEDATE', inplace=True)
        fig = go.Figure()
        for ticker in first_tier_tickers:
            ticker_data = first_tier_data[first_tier_data['TICKER'] == ticker]
            fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['CLOSE'], mode='lines', name=ticker))
        fig.update_layout(
            title='Цены закрытия для нескольких акций первого эшелона',
            xaxis_title='Дата',
            yaxis_title='Цена закрытия',
            template='plotly_dark'
        )
        st.plotly_chart(fig)
        logging.info("Displayed EDA and stock prices for first-tier tickers.")


def train_model(ticker, window_size, forecast_days):
    params = {"window_size": window_size, "forecast_days": forecast_days}
    logging.info(
        f"Training model for ticker {ticker} with window size {window_size} and forecast days {forecast_days}.")
    response = requests.post(f"{api_url}/fit?ticker={ticker}", json=params)
    if response.status_code == 200:
        st.success(f"Модель для {ticker} обучена успешно!")
        logging.info(f"Model for {ticker} trained successfully.")
        return True
    else:
        st.error(f"Ошибка при обучении модели для {ticker}: {response.text}")
        logging.error(f"Error training model for {ticker}: {response.text}")
        return False


def show_model_info(data):
    response = requests.get(f"{api_url}/predict")
    if response.status_code == 200:
        model_info = response.json()
        st.subheader(f"Информация о модели для {model_info['ticker']}")
        st.write(f"MAE: {model_info['mae']}")
        st.write(f"MSE: {model_info['mse']}")
        st.write(f"RMSE: {model_info['rmse']}")
        st.write(f"R²: {model_info['r2']}")
        forecast_dates = pd.to_datetime(model_info['forecast_dates'])
        forecast_prices = model_info['forecast_prices']
        ticker_data = data[data['TICKER'] == model_info['ticker']]
        ticker_data['TRADEDATE'] = pd.to_datetime(ticker_data['TRADEDATE'])
        ticker_data = ticker_data.sort_values(by='TRADEDATE')
        window_size = 10
        X, y = prepare_data(ticker_data['CLOSE'].values, window_size)
        train_size = int(len(X) * 0.8)
        predictions = np.array(model_info['predicted_prices'])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=ticker_data['TRADEDATE'], y=ticker_data['CLOSE'], mode='lines', name='Исторические данные',
                       line=dict(color='blue')))
        predicted_dates = ticker_data['TRADEDATE'].iloc[train_size + window_size:]
        predicted_dates = predicted_dates[:len(predictions)]
        fig.add_trace(
            go.Scatter(x=predicted_dates, y=predictions, mode='lines', name='Predicted Prices (Linear Regression)',
                       line=dict(color='green')))
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=forecast_prices, mode='lines', name='Future Forecast (Linear Regression)',
                       line=dict(color='red', dash='dash')))
        fig.update_layout(
            title=f'Прогноз цен акций для {model_info["ticker"]}',
            title_y=1.0,
            xaxis_title='Дата',
            yaxis_title='Цена закрытия',
            template='plotly_dark',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=0.99,
                xanchor='center',
                x=0.5
            )
        )
        st.plotly_chart(fig)
        logging.info(f"Displayed model information and forecast for {model_info['ticker']}.")
    else:
        st.error("Ошибка при получении информации о модели.")
        logging.error("Error getting model information.")


def list_models():
    response = requests.get(f"{api_url}/models")
    if response.status_code == 200:
        models = response.json()
        logging.info("Fetched list of available models.")
        return models
    else:
        st.error("Ошибка при получении списка моделей.")
        logging.error("Error fetching list of models.")
        return []


def set_active_model(ticker):
    response = requests.post(f"{api_url}/set?ticker={ticker}")
    if response.status_code == 200:
        st.success(f"Модель {ticker} установлена как активная.")
        logging.info(f"Set model {ticker} as active.")
    else:
        st.error(f"Ошибка при установке активной модели: {response.text}")
        logging.error(f"Error setting active model {ticker}: {response.text}")


def show_comparison(models_data):
    fig = go.Figure()

    for model in models_data:
        ticker = model['ticker']
        forecast_dates = pd.to_datetime(model['forecast_dates'])
        forecast_prices = model['forecast_prices']
        predicted_prices = model['predicted_prices']

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_prices,
                mode='lines',
                name=f'Forecast for {ticker}',
                line=dict(dash='dash')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=predicted_prices,
                mode='lines',
                name=f'Predicted for {ticker}',
                line=dict(color='green')
            )
        )

    fig.update_layout(
        title='Сравнение прогнозов для нескольких моделей',
        xaxis_title='Дата',
        yaxis_title='Цена закрытия',
        template='plotly_dark',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.99,
            xanchor='center',
            x=0.5
        )
    )
    st.plotly_chart(fig)


st.title("Stock Price Prediction with Linear Regression")

data = load_data()

if data is not None:
    show_eda(data)

    window_size = st.slider("Размер окна", min_value=5, max_value=30, value=10)
    forecast_days = st.slider("Количество дней для прогноза", min_value=5, max_value=30, value=10)

    ticker = st.selectbox("Выберите тикер для обучения модели", data['TICKER'].unique())

    if st.button("Обучить модель"):
        if train_model(ticker, window_size, forecast_days):
            st.experimental_rerun()

    available_models = list_models()
    if available_models:
        active_model_ticker = st.selectbox("Выберите тикер для установки активной модели", available_models)
        if st.button(f"Установить модель {active_model_ticker} как активную"):
            set_active_model(active_model_ticker)

    if st.button("Показать информацию о модели и Прогнозные цены"):
        show_model_info(data)

    if st.button("Показать все модели"):
        available_models = list_models()
        st.write("Список доступных моделей:")
        for model in available_models:
            st.write(f"- {model}")
        logging.info("Displayed list of all available models.")

    tickers = st.multiselect("Выберите тикеры для сравнения", data['TICKER'].unique())
    if tickers:
        if st.button("Сравнить модели"):
            models_data = compare_models(tickers)
            if models_data:
                show_comparison(models_data)
