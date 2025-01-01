from fastapi import APIRouter, HTTPException
from app.models.linear_regression import list_models, load_metrics, set_active_model
from pydantic import BaseModel
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class Model(BaseModel):
    ticker: str
    path: str

class SetModelResponse(BaseModel):
    status: str

class ExperimentMetricsResponse(BaseModel):
    ticker: str
    train_losses: List[float]
    val_losses: List[float]

@router.get("/", response_model=List[Model])
async def get_models() -> List[Model]:
    try:
        models = list_models()
        return models
    except Exception as e:
        logger.error(f"Failed to fetch models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving models")

@router.post("/{model_id}", response_model=SetModelResponse)
async def set_model(model_id: str) -> Dict[str, str]:
    try:
        set_active_model(model_id)
        return {"status": f"Model {model_id} set as active"}
    except Exception as e:
        logger.error(f"Failed to set active model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error setting active model")

@router.get("/metrics", response_model=List[ExperimentMetricsResponse])
async def get_experiment_metrics(tickers: List[str]) -> List[ExperimentMetricsResponse]:
    metrics = []
    for ticker in tickers:
        try:
            experiment_metrics = load_metrics(ticker)
            metrics.append(ExperimentMetricsResponse(
                ticker=ticker,
                train_losses=experiment_metrics.get('train_losses', []),
                val_losses=experiment_metrics.get('val_losses', [])
            ))
        except Exception as e:
            logger.error(f"Failed to fetch metrics for ticker {ticker}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error retrieving metrics for {ticker}")
    return metrics
