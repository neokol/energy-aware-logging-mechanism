import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

# Define where logs will be saved
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / os.getenv("LOG_FILE", "app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logging():
    """
    Configures logging to write to Console AND File.
    """
    # Create a custom format
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    # 1. Configure the Root Logger
    logging.basicConfig(
        level=LOG_LEVEL,
        format=log_format,
        handlers=[
            # Handler 1: Write to File (rotates if it gets huge, optional but good)
            logging.FileHandler(LOG_FILE),
            # Handler 2: Write to Terminal (Standard Output)
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    
    # Prevent extensive logs from libraries 
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)