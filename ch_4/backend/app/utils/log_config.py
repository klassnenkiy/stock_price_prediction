import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    log_handler = RotatingFileHandler("logs/app.log", maxBytes=10**6, backupCount=5)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, handlers=[log_handler], format=log_format)
