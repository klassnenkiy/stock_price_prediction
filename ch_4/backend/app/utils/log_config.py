import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    handler = RotatingFileHandler("logs/app.log", maxBytes=2000000, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
