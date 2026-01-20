import os
import logging
import pandas as pd
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.database.db import get_async_session
from backend.app.models.datasets import Dataset
from backend.app.models.experiments import Experiment
from backend.app.schemas.experiments import ExperimentResponse
from backend.app.services.experiment_service import execute_experiment
from backend.app.services.model_factory import ModelFactory
from backend.app.models.enums import PrecisionType

logger = logging.getLogger(__name__)

router = APIRouter()


async def _get_dataset_and_model(session: AsyncSession, dataset_id: str):
    """
    Fetches dataset from DB, loads CSV, and instantiates the Model Service.
    """
    # 1. Fetch from DB
    result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        logger.error(f"Dataset with ID {dataset_id} not found")
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 2. Load CSV
    if not os.path.exists(dataset.filepath):
        logger.error(f"File not found at path: {dataset.filepath}")
        raise HTTPException(status_code=404, detail="File not found on disk")
    try:
        df = pd.read_csv(dataset.filepath)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise HTTPException(status_code=500, detail=f"Could not read CSV: {e}")

    # 3. Get Model Service
    service_key = dataset.ai_model.upper()
    try:
        model_service = ModelFactory.get_model_service(service_key)
    except ValueError:
        logger.error(f"Model '{dataset.ai_model}' not supported")
        raise HTTPException(status_code=400, detail=f"Model '{dataset.ai_model}' not supported")

    return dataset, df, model_service


@router.post("/run-experiment", response_model=ExperimentResponse)
async def run_experiment(
    dataset_id: str,
    precision: PrecisionType,
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info(f"Received experiment request for dataset ID: {dataset_id}")
        dataset, df, model_service = await _get_dataset_and_model(session, dataset_id)

        experiment = await execute_experiment(
        session=session,
        dataset=dataset,
        df=df,
        model_service=model_service,
        precision=precision.value
    )
        logger.info(f"Experiment completed for dataset ID: {dataset_id} with model type: {precision.value}")
        return experiment
    except HTTPException as he:
        logger.error(f"HTTP error during experiment: {he.detail}")
        raise he
    

@router.get("/experiments/", response_model=List[ExperimentResponse])
async def get_experiments(
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info("Fetching all experiments from the database.")
        result = await session.execute(select(Experiment))
        experiments = [row[0] for row in result.all()]
        logger.info(f"Fetched {len(experiments)} experiments.")
        return experiments
    except Exception as e:
        logger.error(f"Error fetching experiments: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch experiments")
    
@router.delete("/experiments/{experiment_id}")
async def delete_experiment(
    experiment_id: str, 
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info(f"Received request to delete experiment with ID: {experiment_id}")
        result = await session.execute(select(Experiment).where(Experiment.id == experiment_id))
        experiment = result.scalar_one_or_none()
        
        if not experiment:
            logger.warning(f"Experiment with ID {experiment_id} not found for deletion")
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        await session.delete(experiment)
        await session.commit()
        
        logger.info(f"Experiment with ID {experiment_id} deleted successfully from database")
        
        return {"detail": "Experiment deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting experiment with ID {experiment_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not delete experiment")
    
@router.get("/compare/{dataset_id}")
async def compare_models(
    dataset_id: str, 
    session: AsyncSession = Depends(get_async_session)
):
    try:
        """
        Runs BOTH fp32 and int8 sequentially and returns the difference.
        """
        logger.info(f"Starting model comparison for dataset ID: {dataset_id}")
        dataset, df, model_service = await _get_dataset_and_model(session, dataset_id)


        exp_fp32 = await execute_experiment(session, dataset, df, model_service, PrecisionType.FP32)
        exp_int8 = await execute_experiment(session, dataset, df, model_service, PrecisionType.INT8)

        # Calculate Logic
        energy_saved_kwh = exp_fp32.energy_consumed_kwh - exp_int8.energy_consumed_kwh
        energy_saved_pct = (energy_saved_kwh / exp_fp32.energy_consumed_kwh * 100) if exp_fp32.energy_consumed_kwh > 0 else 0
        latency_saved_sec = exp_fp32.latency_seconds - exp_int8.latency_seconds
        latency_saved_pct = (latency_saved_sec / exp_fp32.latency_seconds * 100) if exp_fp32.latency_seconds > 0 else 0
        logger.info(f"Model comparison completed for dataset ID: {dataset_id}")
        return {
            "dataset_id": dataset.id,
            "model_type": dataset.ai_model,
            "fp32_results": exp_fp32, 
            "int8_results": exp_int8,
            "improvement": {
                "energy_saved_kwh": energy_saved_kwh,
                "energy_saved_percentage": round(energy_saved_pct, 2),
                "latency_reduced_percentage": round(latency_saved_pct, 2),
                "accuracy_loss": round(exp_fp32.accuracy - exp_int8.accuracy, 4)
            }
        }
    except HTTPException as he:
        logger.error(f"HTTP error during model comparison: {he.detail}")
        raise he