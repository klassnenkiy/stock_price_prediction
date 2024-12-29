from fastapi import APIRouter, HTTPException
from app.models.linear_regression import list_models, set_active_model
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def get_models():
    try:
        models = list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to fetch models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error retrieving models")

@router.post("/{model_id}")
async def set_model(model_id: str):
    try:
        set_active_model(model_id)
        return {"status": f"Model {model_id} set as active"}
    except Exception as e:
        logger.error(f"Failed to set active model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error setting active model")
