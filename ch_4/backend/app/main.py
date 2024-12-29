from fastapi import FastAPI
from app.api.router import api_router
from app.utils.log_config import setup_logging

setup_logging()

app = FastAPI(
    title="Stock Price Prediction API",
    description="API for training and predicting stock prices using machine learning models",
    version="1.0.0",
)

app.include_router(api_router)
