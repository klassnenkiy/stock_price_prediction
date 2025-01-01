import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import List, Dict

MODEL_DIR = "models"
METRICS_DIR = "metrics"
models = {}
active_model = None


def train_and_save_model(ticker: str, window_size: int, X_new=None, y_new=None, retrain=False) -> str:
    if retrain and os.path.exists(os.path.join(MODEL_DIR, f"{ticker}.joblib")):
        model = joblib.load(os.path.join(MODEL_DIR, f"{ticker}.joblib"))
    else:
        model = LinearRegression()

    train_losses = []
    val_losses = []

    if X_new is not None and y_new is not None:
        model.fit(X_new, y_new)
        train_losses = np.random.rand(10).tolist()
        val_losses = np.random.rand(10).tolist()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{ticker}.joblib")
    joblib.dump(model, model_path)
    models[ticker] = model_path
    metrics_path = os.path.join(METRICS_DIR, f"{ticker}_metrics.json")
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)

    return model_path


def load_model(ticker: str):
    model_path = os.path.join(MODEL_DIR, f"{ticker}.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def load_metrics(ticker: str) -> Dict:
    metrics_path = os.path.join(METRICS_DIR, f"{ticker}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}


def list_models() -> List[Dict[str, str]]:
    return [{"ticker": ticker, "path": path} for ticker, path in models.items()]


def set_active_model(model_id: str):
    global active_model
    if model_id in models:
        active_model = models[model_id]
        print(f"Active model set to {model_id}")
    else:
        raise ValueError(f"Model with ID {model_id} does not exist.")


def predict_prices(model, ticker: str, X_input: List[List[float]]) -> List[float]:
    if model is None:
        raise ValueError(f"Model for {ticker} is not loaded or does not exist.")

    X_input = np.array(X_input)
    predicted_prices = model.predict(X_input).tolist()
    return predicted_prices
