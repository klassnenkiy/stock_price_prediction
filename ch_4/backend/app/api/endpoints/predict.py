from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.linear_regression import load_model, predict_prices
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class PredictRequest(BaseModel):
    ticker: str
    forecast_days: int

@router.post("/")
async def predict(request: PredictRequest):
    try:
        model = load_model(request.ticker)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        forecast_dates, forecast_prices = predict_prices(model, request.ticker, request.forecast_days)
        return {"forecast_dates": forecast_dates, "forecast_prices": forecast_prices}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction process failed")
