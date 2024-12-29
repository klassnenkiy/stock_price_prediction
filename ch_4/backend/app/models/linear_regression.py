import os
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

MODEL_DIR = "models"
models = {}

def train_and_save_model(ticker, window_size):
    model = LinearRegression()
    X = np.random.rand(100, window_size)
    y = np.random.rand(100)
    model.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{ticker}.joblib")
    joblib.dump(model, model_path)
    models[ticker] = model_path
    return model_path

def load_model(ticker):
    model_path = os.path.join(MODEL_DIR, f"{ticker}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def list_models():
    return [{"ticker": ticker, "path": path} for ticker, path in models.items()]

def predict_prices(model, ticker, forecast_days):
    return [f"2024-12-{day}" for day in range(1, forecast_days + 1)], [100 + day for day in range(forecast_days)]

def set_active_model(model_id):
    global active_model
    if model_id in models:
        active_model = models[model_id]
        print(f"Active model set to {model_id}")
    else:
        raise ValueError(f"Model with ID {model_id} does not exist.")
