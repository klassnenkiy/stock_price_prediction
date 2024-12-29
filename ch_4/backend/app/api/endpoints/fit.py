from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.services.training import train_model
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class TrainRequest(BaseModel):
    window_size: int
    ticker: str

@router.post("/")
async def fit_model(request: TrainRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received training request for ticker: {request.ticker}")
    try:
        background_tasks.add_task(train_model, request.ticker, request.window_size)
        return {"status": "Training started in the background"}
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Training process failed")
