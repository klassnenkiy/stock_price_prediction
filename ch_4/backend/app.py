import os
import logging
from io import StringIO
from logging.handlers import RotatingFileHandler
import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel
from typing import List, Optional
import asyncio


log_filename = "/var/log/app/backend/app.log"
log_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5)
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(log_handler)
logger.setLevel(logging.INFO)
logging.info("Log rotation initialized.")

app = FastAPI()

models = {}
active_model_id = None

TICKERS = ["SBER", "SBERP", "ROSN", "SIBN", "NVTK", "LKOH", "GAZP", "GMKN",
           "PLZL", "TATN", "TATNP", "SNGS", "SNGSP", "CHMF", "NLMK"]


class ModelInfo(BaseModel):
    """Информация о модели."""
    ticker: str
    mse: float
    mae: float
    rmse: float
    r2: float
    forecast_dates: List[str]
    forecast_prices: List[float]


class ModelParams(BaseModel):
    """Параметры модели для обучения и прогноза."""
    window_size: int
    forecast_days: int


def load_data():
    """Загружает данные с внешнего источника."""
    logging.info("Loading data from external source...")
    file_url = "https://drive.google.com/uc?id=1Rn3-XWfgK-fs7-8G2HM9bkLQ9utTA4wJ"

    try:
        response = requests.get(file_url)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error downloading data.")

        data = pd.read_csv(StringIO(response.text))
        data = data.fillna(method='ffill')
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail="Error loading data.")


def prepare_data(data, window_size):
    """Подготавливает данные для обучения модели."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def train_model(ticker_data, window_size, forecast_days):
    """Обучает модель линейной регрессии на данных тикера."""
    logging.info(f"Training model for ticker {ticker_data['TICKER'].iloc[0]}...")
    ticker_data = ticker_data.sort_values(by='TRADEDATE')
    ticker_data_values = ticker_data['CLOSE'].values
    dates = ticker_data['TRADEDATE'].values

    X, y = prepare_data(ticker_data_values, window_size)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    last_data = ticker_data_values[-window_size:]
    forecast_prices = []
    forecast_dates = pd.date_range(dates[-1], periods=forecast_days + 1, freq='B')[1:]

    for _ in range(forecast_days):
        forecast = model.predict(last_data.reshape(1, -1))
        forecast_prices.append(forecast[0])
        last_data = np.append(last_data[1:], forecast[0])

    model_info = {
        "ticker": ticker_data['TICKER'].iloc[0],
        "mse": mse,
        "mae": np.mean(np.abs(y_test - predictions)),
        "rmse": rmse,
        "r2": r2,
        "forecast_dates": list(forecast_dates),
        "forecast_prices": forecast_prices,
        "predicted_prices": list(predictions)
    }

    logging.info(f"Model for {ticker_data['TICKER'].iloc[0]} trained successfully.")

    model_filename = save_model(model, ticker_data['TICKER'].iloc[0])

    models[ticker_data['TICKER'].iloc[0]] = model_info

    return model, model_info, model_filename


def save_model(model, ticker):
    """Сохраняет модель в файл."""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model_filename = os.path.join(models_dir, f"model_{ticker}.joblib")
    joblib.dump(model, model_filename)
    logging.info(f"Model for {ticker} saved at {model_filename}.")
    return model_filename


def load_model(ticker):
    """Загружает модель из файла."""
    model_filename = f"models/model_{ticker}.joblib"
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
    return None


data = load_data()


@app.on_event("startup")
async def startup():
    """Инициализация моделей при старте приложения."""
    global models, active_model_id
    logging.info("Initializing models...")
    for ticker in TICKERS:
        ticker_data = data[data['TICKER'] == ticker]
        if len(ticker_data) > 10:
            model, model_info, model_filename = train_model(ticker_data, window_size=10, forecast_days=30)
            models[ticker] = model_info
            if not active_model_id:
                active_model_id = ticker
    logging.info("Server started and models loaded.")


@app.post("/fit")
async def fit(ticker: str, params: ModelParams, background_tasks: BackgroundTasks):
    """Запускает обучение модели на данных тикера в фоновом режиме."""
    ticker_data = data[data['TICKER'] == ticker]
    if len(ticker_data) <= 10:
        raise HTTPException(status_code=400, detail="Insufficient data for training")
    try:
        background_tasks.add_task(
            train_model_with_timeout, ticker_data, params.window_size, params.forecast_days, timeout=10
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Model training exceeded the timeout limit.")

    return {"message": "Model training started", "models": models}


async def train_model_with_timeout(ticker_data, window_size, forecast_days, timeout):
    """Запускает обучение модели с тайм-аутом."""
    try:
        result = await asyncio.to_thread(
            train_model, ticker_data, window_size, forecast_days
        )
        return result
    except asyncio.TimeoutError:
        logging.error(
            f"Model training for {ticker_data['TICKER'].iloc[0]} exceeded timeout limit of {timeout} seconds."
        )
        raise HTTPException(status_code=408, detail="Model training exceeded the timeout limit.")


@app.get("/predict")
async def predict():
    """Возвращает прогноз для активной модели."""
    if active_model_id is None:
        raise HTTPException(status_code=400, detail="No active model available")

    model_info = models.get(active_model_id)

    forecast_dates = pd.to_datetime(model_info['forecast_dates'])
    forecast_prices = model_info['forecast_prices']

    predicted_prices = model_info.get('predicted_prices', [])

    return {
        "ticker": model_info['ticker'],
        "mae": model_info['mae'],
        "mse": model_info['mse'],
        "rmse": model_info['rmse'],
        "r2": model_info['r2'],
        "forecast_dates": list(forecast_dates),
        "forecast_prices": forecast_prices,
        "predicted_prices": predicted_prices
    }


@app.get("/models")
async def list_models():
    """Возвращает список всех моделей."""
    return models


@app.post("/set")
async def set_active_model(ticker: str):
    """Устанавливает активную модель для прогнозирования."""
    global active_model_id
    if ticker in models:
        active_model_id = ticker
        return {"message": f"Active model set to {ticker}"}
    raise HTTPException(status_code=400, detail="Model not found")


@app.post("/retrain")
async def retrain(ticker: str, params: ModelParams):
    """Перетренирует модель на новых данных тикера."""
    ticker_data = data[data['TICKER'] == ticker]
    if len(ticker_data) <= 10:
        raise HTTPException(status_code=400, detail="Insufficient data for retraining")

    model_info = models.get(ticker)
    if not model_info:
        raise HTTPException(status_code=400, detail=f"No model found for ticker {ticker}")

    model = load_model(ticker)
    if not model:
        raise HTTPException(status_code=400, detail="Model could not be loaded")

    logging.info(f"Retraining model for {ticker}...")
    X, y = prepare_data(ticker_data['CLOSE'].values, params.window_size)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    last_data = ticker_data['CLOSE'].values[-params.window_size:]
    forecast_prices = []
    forecast_dates = pd.date_range(ticker_data['TRADEDATE'].iloc[-1], periods=params.forecast_days + 1, freq='B')[1:]

    for _ in range(params.forecast_days):
        forecast = model.predict(last_data.reshape(1, -1))
        forecast_prices.append(forecast[0])
        last_data = np.append(last_data[1:], forecast[0])

    model_info.update({
        "mse": mse,
        "mae": np.mean(np.abs(y_test - predictions)),
        "rmse": rmse,
        "r2": r2,
        "forecast_dates": list(forecast_dates),
        "forecast_prices": forecast_prices,
        "predicted_prices": list(predictions)
    })

    save_model(model, ticker)

    return {"message": f"Model for {ticker} retrained successfully", "models": models}


@app.get("/compare_models")
async def compare_models(tickers: Optional[List[str]] = Query(None)):
    """Сравнивает несколько моделей по прогнозам."""
    model_data = []
    for ticker in tickers:
        model_info = models.get(ticker)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model for {ticker} not found")

        forecast_dates = pd.to_datetime(model_info['forecast_dates'])
        forecast_prices = model_info['forecast_prices']
        predicted_prices = model_info.get('predicted_prices', [])

        model_data.append({
            "ticker": model_info['ticker'],
            "forecast_dates": list(forecast_dates),
            "forecast_prices": forecast_prices,
            "predicted_prices": predicted_prices
        })
    return model_data
