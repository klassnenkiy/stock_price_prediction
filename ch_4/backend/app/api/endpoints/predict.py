from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.linear_regression import load_model, predict_prices
from typing import List
import logging
import pandas as pd

logger = logging.getLogger(__name__)

router = APIRouter()


class PredictRequest(BaseModel):
    ticker: str
    X_input: List[List[float]]


class PredictResponse(BaseModel):
    forecast_dates: List[str]
    forecast_prices: List[float]


@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    try:
        model = load_model(request.ticker)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        forecast_prices = predict_prices(model, request.ticker, request.X_input)
        forecast_dates = pd.date_range(start='2023-01-01', periods=len(forecast_prices), freq='B').strftime(
            '%Y-%m-%d').tolist()
        return {"forecast_dates": forecast_dates, "forecast_prices": forecast_prices}

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction process failed")
