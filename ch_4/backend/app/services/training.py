import logging
from app.models.linear_regression import train_and_save_model

logger = logging.getLogger(__name__)

def train_model(ticker: str, window_size: int, X_new=None, y_new=None, retrain=False) -> None:
    try:
        model_path = train_and_save_model(ticker, window_size, X_new, y_new, retrain)
        logger.info(f"Training completed for ticker: {ticker}, model saved at {model_path}")
    except Exception as e:
        logger.error(f"Training failed for ticker: {ticker} - {str(e)}")

