from fastapi import APIRouter
from app.api.endpoints import fit, predict, models

api_router = APIRouter()
api_router.include_router(fit.router, prefix="/fit", tags=["Fit Model"])
api_router.include_router(predict.router, prefix="/predict", tags=["Predict"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
