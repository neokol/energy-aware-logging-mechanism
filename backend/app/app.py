from fastapi import  FastAPI 
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import logging

from backend.app.core.logging import setup_logging
from backend.app.database.db import  create_db_and_tables
from backend.app.routers import dataset
from backend.app.routers import experiments

load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield

app = FastAPI(
        title="Energy Aware Logging Mechanism",
        description="API for Energy Aware Logging Mechanism",
        version="1.0.0",
        lifespan=lifespan
    )


@app.get("/", summary="Home Endpoint", tags=["Health Check"])
async def read_root():
    logger.info("Welcome to Energy Aware Logging Mechanism API.")
    return {"message": "Welcome to Energy Aware Logging Mechanism API"}

@app.get("/status", summary="Status Check", tags=["Health Check"])
async def status():
    logger.info("API status checked.")
    return {"status": "API is running smoothly!"}

app.include_router(dataset.router, tags=["datasets"])
app.include_router(experiments.router, tags=["experiments"])

