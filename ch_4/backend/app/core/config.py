import os

class Settings:
    model_path: str = "models/linear_regression_model.pkl"
    train_timeout: int = 10
    model_dir: str = "models"


settings = Settings()
