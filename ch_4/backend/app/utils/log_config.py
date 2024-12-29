import logging
from logging.handlers import RotatingFileHandler
import json_log_formatter

def setup_logging():
    formatter = json_log_formatter.JSONFormatter()

    file_handler = RotatingFileHandler("logs/app.log", maxBytes=10**6, backupCount=5)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
    )
