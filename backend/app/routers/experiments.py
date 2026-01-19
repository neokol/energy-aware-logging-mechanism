import os
import logging
import pandas as pd
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from codecarbon import EmissionsTracker

from backend.app.database.db import get_async_session
from backend.app.models.datasets import Dataset
from backend.app.models.experiments import Experiment
from backend.app.schemas.experiments import ExperimentResponse, ExperimentCreate
from backend.app.services.experiment_service import execute_experiment
from backend.app.services.model_factory import ModelFactory

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
    service_key = "mlp" if dataset.ai_model == "MLP" else dataset.ai_model.lower()
    try:
        model_service = ModelFactory.get_model_service(service_key)
    except ValueError:
        logger.error(f"Model '{dataset.ai_model}' not supported")
        raise HTTPException(status_code=400, detail=f"Model '{dataset.ai_model}' not supported")

    return dataset, df, model_service


@router.post("/run-experiment", response_model=ExperimentResponse)
async def run_experiment(
    request: ExperimentCreate,
    session: AsyncSession = Depends(get_async_session)
):
    try:
        logger.info(f"Received experiment request for dataset ID: {request.dataset_id}")
        dataset, df, model_service = await _get_dataset_and_model(session, request.dataset_id)

        experiment = await execute_experiment(
        session=session,
        dataset=dataset,
        df=df,
        model_service=model_service,
        precision=request.model_type
    )
    
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

        return experiments
    except Exception as e:
        logger.error(f"Error fetching experiments: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch experiments")
    
@router.get("/compare/{dataset_id}")
async def compare_models(
    dataset_id: str, 
    session: AsyncSession = Depends(get_async_session)
):
    """
    Runs BOTH fp32 and int8 sequentially and returns the difference.
    """
    dataset, df, model_service = await _get_dataset_and_model(session, dataset_id)

    # Call the Service TWICE
    exp_fp32 = await execute_experiment(session, dataset, df, model_service, "fp32")
    exp_int8 = await execute_experiment(session, dataset, df, model_service, "int8")

    # Calculate Logic
    energy_saved_kwh = exp_fp32.energy_consumed_kwh - exp_int8.energy_consumed_kwh
    energy_saved_pct = (energy_saved_kwh / exp_fp32.energy_consumed_kwh * 100) if exp_fp32.energy_consumed_kwh > 0 else 0
    latency_saved_sec = exp_fp32.latency_seconds - exp_int8.latency_seconds
    latency_saved_pct = (latency_saved_sec / exp_fp32.latency_seconds * 100) if exp_fp32.latency_seconds > 0 else 0

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