import logging
import shutil
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from dotenv import load_dotenv
import os

from app.database.db import get_async_session
from app.models.model import Model
from app.models.enums import AlgorithmType

load_dotenv()

UPLOAD_MODEL_DIR = os.getenv("UPLOAD_MODEL_DIR")

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/models")
async def upload_model(
        file:UploadFile = File(...), 
        description: str= "", 
        algorithm: AlgorithmType = AlgorithmType,
        session: AsyncSession = Depends(get_async_session)
    ):
    try:
        logger.info(f"Received upload request for model file: {file.filename}")
        
        file_path = os.path.join(UPLOAD_MODEL_DIR, file.filename)
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
                logger.info(f"Model file '{file.filename}' saved successfully at '{file_path}'")
        except Exception as e:
            logger.error(f"Failed to save model file '{file.filename}'. Error: {e}")
            raise HTTPException(status_code=500, detail=f"Could not save model file: {e}")
        
        new_model = Model(
            filename=file.filename,
            filepath=file_path,
            description=description,
            algorithm=algorithm
        )
        session.add(new_model)
        await session.commit()
        await session.refresh(new_model)
        
        logger.info(f"Model uploaded successfully. DB ID: {new_model.id}")
        
        return new_model
    except Exception as e:
        logger.error(f"Error during model upload: {e}")
        raise HTTPException(status_code=500, detail="Model upload failed")
    
    
@router.get("/models")
async def list_models(session: AsyncSession = Depends(get_async_session)):
    try:
        logger.info("Received request to list all models")
        result = await session.execute(select(Model))
        models = result.scalars().all()
        logger.info(f"Returning {len(models)} models")
        return models
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch models")