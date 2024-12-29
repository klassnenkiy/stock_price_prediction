from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.linear_regression import predict

router = APIRouter()

class PredictRequest(BaseModel):
    input_data: list

@router.post("/")
async def predict_model(request: PredictRequest):
    try:
        predictions = predict(request.input_data)
        return {"status": "success", "predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
