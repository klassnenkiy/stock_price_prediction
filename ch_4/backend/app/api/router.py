from fastapi import APIRouter
from app.api.endpoints import fit, predict, models

api_router = APIRouter()

api_router.include_router(fit.router, prefix="/fit", tags=["Model Training"])
api_router.include_router(predict.router, prefix="/predict", tags=["Prediction"])
api_router.include_router(models.router, prefix="/models", tags=["Model Management"])