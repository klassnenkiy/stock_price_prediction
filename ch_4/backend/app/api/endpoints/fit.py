from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.training import train_model

router = APIRouter()

class FitRequest(BaseModel):
    model_type: str
    hyperparameters: dict

@router.post("/")
async def fit_model(request: FitRequest):
    try:
        result = train_model(request.model_type, request.hyperparameters)
        return {"status": "success", "details": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
