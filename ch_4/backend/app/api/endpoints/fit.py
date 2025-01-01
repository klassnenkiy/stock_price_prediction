from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.services.training import train_model
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class TrainRequest(BaseModel):
    window_size: int
    ticker: str
    retrain: bool = False
    X_new: List[List[float]] = None
    y_new: List[float] = None

class TrainResponse(BaseModel):
    status: str

@router.post("/", response_model=TrainResponse)
async def fit_model(request: TrainRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    logger.info(f"Received training request for ticker: {request.ticker}")
    try:
        background_tasks.add_task(
            train_model,
            request.ticker,
            request.window_size,
            request.X_new,
            request.y_new,
            request.retrain,
        )
        return {"status": "Training or retraining started in the background"}
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Training process failed")
