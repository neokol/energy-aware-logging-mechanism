import logging
import shutil
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from dotenv import load_dotenv
import os

from backend.app.database.db import get_async_session
from backend.app.models.datasets import Dataset

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR")

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/datasets")
async def create_dataset(
        file:UploadFile = File(...), 
        description: str= "", 
        ai_model: str = "",
        session: AsyncSession = Depends(get_async_session)
    ):
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                logger.info(f"File '{file.filename}' saved successfully at '{file_path}'")
        except Exception as e:
            logger.error(f"Failed to save file '{file.filename}'. Error: {e}")
            raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
        
        new_dataset = Dataset(
            filename=file.filename,
            filepath=file_path,
            description=description,
            ai_model=ai_model
        )
        session.add(new_dataset)
        await session.commit()
        await session.refresh(new_dataset)
        
        logger.info(f"Dataset uploaded successfully. DB ID: {new_dataset.id}")
        
        return new_dataset
    except Exception as e:
        logger.error(f"Error during dataset upload: {e}")
        raise HTTPException(status_code=500, detail="Dataset upload failed")

@router.get("/datasets")
async def get_datasets(session: AsyncSession = Depends(get_async_session)):
    try:
        logger.info("Fetching all datasets from the database.")
        result = await session.execute(select(Dataset))
        datasets = [row[0] for row in result.all()]

        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error fetching datasets: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch datasets")