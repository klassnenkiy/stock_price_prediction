import os
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import List

MODEL_DIR = "models"
models = {}

def train_and_save_model(ticker: str, window_size: int, X_new=None, y_new=None, retrain=False) -> str:
    if retrain and os.path.exists(os.path.join(MODEL_DIR, f"{ticker}.joblib")):
        model = joblib.load(os.path.join(MODEL_DIR, f"{ticker}.joblib"))
    else:
        model = LinearRegression()

    if X_new is not None and y_new is not None:
        model.fit(X_new, y_new)
    else:
        X = np.random.rand(100, window_size)
        y = np.random.rand(100)
        model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{ticker}.joblib")
    joblib.dump(model, model_path)
    models[ticker] = model_path
    return model_path

def load_model(ticker: str):
    model_path = os.path.join(MODEL_DIR, f"{ticker}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def list_models() -> List[Dict[str, str]]:
    return [{"ticker": ticker, "path": path} for ticker, path in models.items()]

def predict_prices(model, ticker: str, X_input: List[List[float]]) -> List[float]:
    forecast_prices = model.predict(X_input).tolist()
    return forecast_prices


def set_active_model(model_id: str):
    global active_model
    if model_id in models:
        active_model = models[model_id]
        print(f"Active model set to {model_id}")
    else:
        raise ValueError(f"Model with ID {model_id} does not exist.")
